#!/usr/bin/env python3
"""
footage_api.py

FastAPI app to:
- list recorded segments (query by camera + time range)
- stream recorded segment files with Range support
- generate / serve thumbnails (cached)
- extract an exact clip for a requested time window (uses ffmpeg)

Assumptions:
- The recorder writes files into OUTPUT_DIR and inserts filename (full path) into the SQLite DB.
- SQLite DB has table 'segments' with columns: id, camera_id, filename, start_ts, end_ts (ISO datetimes).
"""

import os
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Query, Body, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime, timezone
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# -------- CONFIG (match your recorder) --------
load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./recordings"))
DB_PATH = OUTPUT_DIR / "index.db"
THUMB_DIR = OUTPUT_DIR / "thumbs"
TMP_DIR = OUTPUT_DIR / "tmp"
CHUNK_SIZE = 1024 * 1024
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

# ensure dirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# -------- Logging --------
log_path = "./logs/api.log"
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[handler, logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RADR Beacon Footage API (Advanced)")

# allow your portal origin(s) ? adjust for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your portal domain in production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------- DB helper functions ----------
_db_lock = None  # no global lock here; sqlite will handle simple concurrency

def iso_to_dt(iso_str: str) -> datetime:
    # support both naive and with tz
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def query_segments(camera_id: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, page: int = 1, per_page: int = 100):
    """
    Return list of segments that *overlap* [start_time, end_time].
    If no start_time/end_time provided, returns newest segments (paginated).
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        c = conn.cursor()
        params = []
        where_clauses = []
        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        if start_time and end_time:
            # overlap condition: segment_end >= start_time AND segment_start <= end_time
            where_clauses.append("end_ts >= ? AND start_ts <= ?")
            params.append(start_time.isoformat())
            params.append(end_time.isoformat())
        where_clause = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        order_clause = "ORDER BY start_ts DESC"
        offset = (page - 1) * per_page
        sql = f"SELECT id, camera_id, filename, start_ts, end_ts FROM segments {where_clause} {order_clause} LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        c.execute(sql, params)
        rows = c.fetchall()
    segments = []
    for r in rows:
        seg = {
            "id": r[0],
            "camera_id": r[1],
            "filename": r[2],
            "start_ts": r[3],
            "end_ts": r[4],
            # stream_url/thumbnail_url are UI-friendly helpers
            "stream_url": f"/api/stream/{Path(r[2]).name}",
            "thumbnail_url": f"/api/thumbnail/{Path(r[2]).name}"
        }
        segments.append(seg)
    return segments

# --------- Helper: find file path for filename or basename ----------
def resolve_file_path(filename_or_basename: str) -> Optional[Path]:
    """If stored filenames in DB are full paths, we still accept basenames from client and resolve."""
    p = Path(filename_or_basename)
    if p.exists():
        return p
    candidate = OUTPUT_DIR / filename_or_basename
    if candidate.exists():
        return candidate
    with sqlite3.connect(str(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("SELECT filename FROM segments WHERE filename LIKE ? LIMIT 1", (f"%{filename_or_basename}",))
        row = c.fetchone()
        if row:
            found = Path(row[0])
            if found.exists():
                return found
    return None

# --------- Endpoint: list segments ----------
@app.get("/api/segments")
def api_segments(
    camera_id: Optional[str] = Query(None),
    start: Optional[str] = Query(None, description="ISO datetime (UTC)"),
    end: Optional[str] = Query(None, description="ISO datetime (UTC)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=500),
):
    try:
        start_dt = iso_to_dt(start) if start else None
        end_dt = iso_to_dt(end) if end else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    segs = query_segments(camera_id=camera_id, start_time=start_dt, end_time=end_dt, page=page, per_page=per_page)
    return {"segments": segs, "page": page, "per_page": per_page, "count": len(segs)}

# --------- Endpoint: thumbnail ----------
@app.get("/api/thumbnail/{filename}")
def api_thumbnail(filename: str):
    fp = resolve_file_path(filename)
    if not fp:
        raise HTTPException(status_code=404, detail="File not found")
    thumb_path = THUMB_DIR / (fp.name + ".jpg")
    if not thumb_path.exists():
        try:
            cmd = [
                FFMPEG_BIN, "-y", "-i", str(fp),
                "-ss", "00:00:01.000",
                "-vframes", "1",
                "-q:v", "2",
                str(thumb_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            try:
                import cv2
                cap = cv2.VideoCapture(str(fp))
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Unable to read frame for thumbnail")
                frame = frame[:, :, ::-1]
                img = Image.fromarray(frame)
                img.thumbnail((320, 180))
                img.save(thumb_path, format="JPEG", quality=80)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Thumbnail generation failed: {e}")
    return FileResponse(str(thumb_path), media_type="image/jpeg")

# --------- Endpoint: stream video with Range support ----------
# Updated stream endpoint in your FastAPI app
@app.get("/api/stream/{filename}")
async def stream_video(request: Request, filename: str):
    """
    Streams a file with Range support. Client should pass basename (filename only).
    """
    fp = resolve_file_path(filename)
    if not fp:
        raise HTTPException(status_code=404, detail="File not found")

    file_size = fp.stat().st_size
    range_header = request.headers.get("range")

    def iterfile(start=0, end=None):
        with open(fp, "rb") as f:
            f.seek(start)
            remaining = (end - start) if end is not None else file_size - start
            while remaining > 0:
                chunk = f.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)

    if range_header:
        # parse "bytes=start-end"
        try:
            bytes_range = range_header.strip().split("=")[1]
            start_str, end_str = bytes_range.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Range header")
        if start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
            # Add CORS headers
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
            # Cache control for better streaming
            "Cache-Control": "public, max-age=3600",
        }
        return StreamingResponse(iterfile(start, end + 1), status_code=206, headers=headers)
    else:
        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
            # Add CORS headers
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
            # Cache control for better streaming
            "Cache-Control": "public, max-age=3600",
        }
        return StreamingResponse(iterfile(), headers=headers)

# Also add OPTIONS endpoint for CORS preflight
@app.options("/api/stream/{filename}")
async def stream_video_options(filename: str):
    """Handle CORS preflight for video streaming"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Max-Age": "86400",
        }
    )

# --------- Utility: build an extracted clip using ffmpeg ----------
def build_clip_for_range(camera_id: Optional[str], desired_start: datetime, desired_end: datetime) -> Path:
    """
    Finds segments overlapping requested range and produces a single mp4 clip trimmed to exact desired_start->desired_end.
    Returns path to produced clip.
    """
    # query DB for segments overlapping requested time window
    segments = query_segments(camera_id=camera_id, start_time=desired_start, end_time=desired_end, page=1, per_page=1000)
    if not segments:
        raise FileNotFoundError("No segments cover requested time range")

    # convert segments to absolute file paths
    seg_files = []
    for s in reversed(segments):  # reverse to lowest-to-highest start_ts
        p = resolve_file_path(Path(s["filename"]).name) or Path(s["filename"])
        if not p or not p.exists():
            continue
        seg_files.append({
            "path": p,
            "start": iso_to_dt(s["start_ts"]),
            "end": iso_to_dt(s["end_ts"]),
        })
    if not seg_files:
        raise FileNotFoundError("No segment files found on disk for requested range")

    # If single file covers range -> trim directly
    if len(seg_files) == 1:
        seg = seg_files[0]
        file_start = seg["start"]
        ss = max(0, (desired_start - file_start).total_seconds())
        to = max(0, (desired_end - file_start).total_seconds())
        out_path = TMP_DIR / f"extract_{camera_id or 'all'}_{desired_start.strftime('%Y%m%dT%H%M%SZ')}_{desired_end.strftime('%Y%m%dT%H%M%SZ')}.mp4"
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(seg["path"]),
            "-ss", str(ss),
            "-to", str(to),
            "-c", "copy",
            str(out_path)
        ]
        subprocess.run(cmd, check=True)
        return out_path

    # multiple segments: concat then trim
    # build a filelist for concat demuxer
    filelist = TMP_DIR / f"filelist_{desired_start.strftime('%Y%m%dT%H%M%SZ')}.txt"
    with open(filelist, "w") as f:
        for s in seg_files:
            f.write(f"file '{str(s['path'])}'\n")

    concat_temp = TMP_DIR / f"concat_{desired_start.strftime('%Y%m%dT%H%M%SZ')}.mp4"
    out_path = TMP_DIR / f"extract_{camera_id or 'all'}_{desired_start.strftime('%Y%m%dT%H%M%SZ')}_{desired_end.strftime('%Y%m%dT%H%M%SZ')}.mp4"

    # concat without re-encoding
    subprocess.run([FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(filelist),
                    "-c", "copy", str(concat_temp)], check=True)

    # Now trim the concatenated file:
    duration = (desired_end - desired_start).total_seconds()
    subprocess.run([FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(concat_temp),
                    "-ss", "0",
                    "-t", str(int(duration)),
                    "-c", "copy",
                    str(out_path)], check=True)

    # clean up concat_temp & filelist (optional)
    try:
        concat_temp.unlink(missing_ok=True)
        filelist.unlink(missing_ok=True)
    except Exception:
        pass

    return out_path

# --------- Endpoint: extract clip for exact requested time window ----------
@app.post("/api/extract")
def api_extract(payload: dict = Body(...)):
    """
    Request JSON:
    {
      "camera_id": "cam1" or null,
      "start": "2025-08-13T14:05:00+00:00",
      "end":   "2025-08-13T14:07:30+00:00"
    }
    Returns: a FileResponse containing the produced clip (application/octet-stream / video/mp4).
    """
    try:
        camera_id = payload.get("camera_id")
        start_iso = payload.get("start")
        end_iso = payload.get("end")
        if not start_iso or not end_iso:
            raise HTTPException(status_code=400, detail="start and end are required")
        start_dt = iso_to_dt(start_iso)
        end_dt = iso_to_dt(end_iso)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad payload: {e}")

    # sanity check
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    try:
        out_path = build_clip_for_range(camera_id, start_dt, end_dt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")

    # return the file (client can stream or download)
    return FileResponse(str(out_path), media_type="video/mp4", filename=out_path.name)

# --------- Health check ----------
@app.get("/api/health")
def health():
    """System health for monitoring"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        raise HTTPException(status_code=500, detail="Database not accessible")

    st = os.statvfs("/")
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    if free_gb < 0.5:
        logger.warning("Low disk space detected.")
    return {"status": "ok", "free_gb": round(free_gb, 2)}

# --------- Run with: uvicorn footage_api:app --host 0.0.0.0 --port 8000 ---------
