#!/usr/bin/env python3
"""
- Fetches *all sensors* for an institution from RADR backend (Supabase API)
- Builds RTSP URLs for each camera
- Spawns recording threads (with time-chunked local storage + timestamp overlay + SQLite index)
- Tags clips as motion/static and auto-cleans old static ones.
"""

import os
import cv2
import time
import signal
import threading
import sqlite3
import requests
import logging
import subprocess
from datetime import datetime, timezone, date
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()

INSTITUTION_ID = os.getenv("INSTITUTION_ID")
BEACON_ID = int(os.getenv("BEACON_ID")
API_BASE_URL = os.getenv("API_BASE_URL")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", 300))
OUTPUT_DIR = "./recordings"
DB_PATH = os.path.join(OUTPUT_DIR, "index.db")

DEFAULT_FPS = 15
FOURCC = "mp4v"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_RETRY = 5
MOTION_THRESHOLD = 400000
STATIC_CLEANUP_ENABLED = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_db_lock = threading.Lock()

# ---------------- DATABASE ----------------
def init_db():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                filename TEXT,
                start_ts TEXT,
                end_ts TEXT,
                is_motion INTEGER DEFAULT 0
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_camera_time ON segments(camera_id, start_ts, end_ts)")
        conn.commit()
        conn.close()

def add_segment(camera_id, filename, start_ts, end_ts, is_motion):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO segments(camera_id, filename, start_ts, end_ts, is_motion) VALUES (?, ?, ?, ?, ?)",
            (camera_id, filename, start_ts.isoformat(), end_ts.isoformat(), is_motion),
        )
        conn.commit()
        conn.close()

# ---------------- CLEANUP ----------------
def cleanup_old_static_clips():
    if not STATIC_CLEANUP_ENABLED:
        return
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT filename, start_ts FROM segments WHERE is_motion=0")
        rows = c.fetchall()
        for filename, start_ts in rows:
            try:
                ts_date = datetime.fromisoformat(start_ts).date()
                if ts_date < date.today():
                    if os.path.exists(filename):
                        os.remove(filename)
                        logging.info(f"[cleanup] Removed old static clip: {filename}")
                        c.execute("DELETE FROM segments WHERE filename=?", (filename,))
            except Exception as e:
                logging.warning(f"[cleanup] Failed to remove {filename}: {e}")
        conn.commit()
        conn.close()

# ---------------- MOTION DETECTION ----------------
def detect_motion_simple(video_path):
    """
    Lightweight motion detection using frame differencing.
    Returns 1 if motion detected, else 0.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        ret, prev = cap.read()
        if not ret:
            cap.release()
            return 0
        motion_score = 0
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            diff = cv2.absdiff(prev, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            motion_score += cv2.countNonZero(gray)
            frame_count += 1
            prev = frame
        cap.release()
        avg_motion = motion_score / max(frame_count, 1)
        is_motion = int(avg_motion > MOTION_THRESHOLD)
        logging.info(f"[motion] {video_path}: avg_pixel_diff={avg_motion:.1f}, motion={is_motion}")
        return is_motion
    except Exception as e:
        logging.warning(f"[motion] Failed for {video_path}: {e}")
        return 0

# ---------------- API ----------------
def fetch_all_sensor_connection_info():
    url = f"{API_BASE_URL}/beacons/{BEACON_ID}/sensors/connection_data"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def build_rtsp_url(sensor_data):
    host = sensor_data["general_info"]["host"]
    path = sensor_data["general_info"]["path"]
    port = sensor_data["general_info"]["port"]
    username = sensor_data["credentials"]["username"] or "admin"
    password = sensor_data["credentials"]["password"] or "admin"
    return f"rtsp://{username}:{password}@{host}:{port}{path}"

# ---------------- RECORDING CORE ----------------
running = True
def stop_all(signum=None, frame=None):
    global running
    running = False
    logging.info("Stopping all recorders...")

signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

def make_filename(camera_id, start_ts):
    ts = start_ts.strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(OUTPUT_DIR, f"{camera_id}_{ts}.mp4")

def create_opencv_capture(rtsp_url):
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    cap.open(rtsp_url)
    return cap

def test_stream_connectivity(camera_id, rtsp_url):
    if '/udp/' in rtsp_url.lower():
        transports = ['udp', 'tcp']
    else:
        transports = ['tcp', 'udp']
    for transport in transports:
        try:
            cmd = [
                'ffprobe', '-rtsp_transport', transport,
                '-analyzeduration', '5000000', '-probesize', '5000000', '-i', rtsp_url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and 'Video:' in result.stderr:
                logging.info(f"[{camera_id}] Stream accessible via {transport}")
                return transport
        except subprocess.TimeoutExpired:
            logging.warning(f"[{camera_id}] {transport} transport test timed out")
            continue
    logging.error(f"[{camera_id}] Stream not accessible via any transport")
    return None

def record_with_ffmpeg(camera_id, rtsp_url, filename, duration_seconds, transport='tcp'):
    try:
        cmd = [
            'ffmpeg', '-rtsp_transport', transport,
            '-fflags', '+genpts',
            '-analyzeduration', '10000000',
            '-probesize', '10000000',
            '-max_delay', '500000',
            '-reorder_queue_size', '0',
            '-i', rtsp_url,
            '-t', str(duration_seconds),
            '-an',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-r', '15',
            '-g', '30',
            '-keyint_min', '15',
            '-sc_threshold', '0',
            '-avoid_negative_ts', 'make_zero',
            '-vsync', 'cfr',
            '-pix_fmt', 'yuv420p',
            '-f', 'mp4',
            '-movflags', '+faststart',
            '-y', filename
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 30)
        if process.returncode == 0 and os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 50000:
                logging.info(f"[{camera_id}] Recorded {filename} ({size} bytes)")
                return True
        logging.error(f"[{camera_id}] ffmpeg failed: {process.stderr[:200]}")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"[{camera_id}] ffmpeg recording timed out")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except Exception as e:
        logging.error(f"[{camera_id}] ffmpeg recording failed: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def should_use_ffmpeg_fallback(camera_id, rtsp_url):
    if '/udp/' in rtsp_url.lower():
        logging.info(f"[{camera_id}] Using ffmpeg fallback for UDP stream")
        return True
    if 'admin:test123456' in rtsp_url:
        return True
    return False

def recorder_thread(camera_id, rtsp_url):
    retry = 0
    use_ffmpeg_fallback = should_use_ffmpeg_fallback(camera_id, rtsp_url)
    stream_transport = 'tcp'
    broken_stream = False

    logging.info(f"[{camera_id}] Testing stream connectivity...")
    detected_transport = test_stream_connectivity(camera_id, rtsp_url)
    if detected_transport is None:
        broken_stream = True
    else:
        stream_transport = detected_transport

    while running and not broken_stream:
        if use_ffmpeg_fallback:
            chunk_start = datetime.now(timezone.utc).replace(microsecond=0)
            filename = make_filename(camera_id, chunk_start)
            success = record_with_ffmpeg(camera_id, rtsp_url, filename, CHUNK_SECONDS, stream_transport)
            if success and os.path.exists(filename) and os.path.getsize(filename) > 50000:
                chunk_end = datetime.now(timezone.utc).replace(microsecond=0)
                motion_flag = detect_motion_simple(filename)
                add_segment(camera_id, filename, chunk_start, chunk_end, motion_flag)
                cleanup_old_static_clips()
                retry = 0
            else:
                retry += 1
                if os.path.exists(filename):
                    os.remove(filename)
                if retry >= MAX_RETRY * 3:
                    broken_stream = True
                    break
                if retry >= MAX_RETRY:
                    time.sleep(30)
                    retry = 0
                else:
                    time.sleep(5)
            continue

        cap = create_opencv_capture(rtsp_url)
        if not cap.isOpened():
            cap.release()
            use_ffmpeg_fallback = True
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)

        chunk_start = datetime.now(timezone.utc).replace(microsecond=0)
        filename = make_filename(camera_id, chunk_start)
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            use_ffmpeg_fallback = True
            continue

        frames_written = 0
        consecutive_failures = 0

        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                time.sleep(0.1)
                continue
            consecutive_failures = 0
            now_ts = datetime.now(timezone.utc).replace(microsecond=0)
            ts_text = now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            cv2.rectangle(frame, (10, 10), (350, 35), (0, 0, 0), -1)
            cv2.putText(frame, ts_text, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            frames_written += 1

            if (now_ts - chunk_start).total_seconds() >= CHUNK_SECONDS:
                chunk_end = now_ts
                writer.release()
                if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                    motion_flag = detect_motion_simple(filename)
                    add_segment(camera_id, filename, chunk_start, chunk_end, motion_flag)
                    cleanup_old_static_clips()
                chunk_start = now_ts
                filename = make_filename(camera_id, chunk_start)
                writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                if not writer.isOpened():
                    break
                frames_written = 0

        if writer and writer.isOpened():
            chunk_end = datetime.now(timezone.utc).replace(microsecond=0)
            writer.release()
            if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                motion_flag = detect_motion_simple(filename)
                add_segment(camera_id, filename, chunk_start, chunk_end, motion_flag)
                cleanup_old_static_clips()
            else:
                if os.path.exists(filename):
                    os.remove(filename)
        cap.release()

        if running and not broken_stream:
            retry += 1
            if retry >= MAX_RETRY and not use_ffmpeg_fallback:
                use_ffmpeg_fallback = True
                retry = 0
            else:
                time.sleep(5)
    logging.info(f"[{camera_id}] Recorder stopped.")

# ---------------- MAIN ----------------
def main():
    init_db()
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logging.info("ffmpeg available")
    except Exception:
        logging.warning("ffmpeg not found")

    logging.info("Fetching sensor connection info from backend...")
    sensors = fetch_all_sensor_connection_info()
    camera_list = []
    for s in sensors:
        cam_id = s["id"]
        info = s.get("general_info", {})
        if not info or "host" not in info or not info["host"]:
            logging.warning(f"[{cam_id}] Missing host info, skipping.")
            continue
        rtsp_url = build_rtsp_url(s)
        camera_list.append((cam_id, rtsp_url))
        logging.info(f"[{cam_id}] URL built.")

    threads = []
    for cam_id, rtsp_url in camera_list:
        t = threading.Thread(target=recorder_thread, args=(cam_id, rtsp_url), daemon=True)
        t.start()
        threads.append(t)

    while running:
        time.sleep(1)
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
