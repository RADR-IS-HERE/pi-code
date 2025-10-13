#!/usr/bin/env python3
"""
- Fetches *all sensors* for an institution from RADR backend (Supabase API)
- Builds RTSP URLs for each camera
- Spawns recording threads (with time-chunked local storage + timestamp overlay + SQLite index)
- Does NOT upload footage automatically (only stores locally).
  Uploading is handled later when a user requests footage.
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
from datetime import datetime, timezone

# --------------- CONFIG -----------------
from dotenv import load_dotenv
load_dotenv()

INSTITUTION_ID = os.getenv("INSTITUTION_ID")
BEACON_ID = int(os.getenv("BEACON_ID"))
API_BASE_URL = os.getenv("API_BASE_URL")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", 300))
OUTPUT_DIR = "./recordings"
DB_PATH = os.path.join(OUTPUT_DIR, "index.db")

DEFAULT_FPS = 15
FOURCC = "mp4v"         # Changed from XVID to mp4v for better MP4 compatibility
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_RETRY = 5

# --------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------- DB Helpers ----------------
_db_lock = threading.Lock()

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
                end_ts TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_camera_time ON segments(camera_id, start_ts, end_ts)")
        conn.commit()
        conn.close()

def add_segment(camera_id, filename, start_ts, end_ts):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO segments(camera_id, filename, start_ts, end_ts) VALUES (?, ?, ?, ?)",
                  (camera_id, filename, start_ts.isoformat(), end_ts.isoformat()))
        conn.commit()
        conn.close()

# --------------- Supabase API Helpers ----------------
def fetch_all_sensor_connection_info():
    """
    Call the backend API to get ALL sensor connection details for this institution.
    Endpoint: /institution/{INSTITUTION_ID}/sensors/connection_data
    """

    url = f"{API_BASE_URL}/beacons/{BEACON_ID}/sensors/connection_data"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()  # expected: list of sensors with {general_info, credentials}

    logging.info("Fetched all sensor info!")

def build_rtsp_url(sensor_data):
    """
    Construct RTSP URL from a single sensor's API data.
    """
    host = sensor_data["general_info"]["host"]
    path = sensor_data["general_info"]["path"]
    port = sensor_data["general_info"]["port"]
    username = sensor_data["credentials"]["username"] or "admin"
    password = sensor_data["credentials"]["password"] or "admin"

    print("credentials test", sensor_data)

    return f"rtsp://{username}:{password}@{host}:{port}{path}"

# --------------- Recorder Logic ----------------
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
    """
    Create OpenCV VideoCapture with proper RTSP settings
    """
    cap = cv2.VideoCapture()
    # Set RTSP transport to TCP (equivalent to ffmpeg's -rtsp_transport tcp)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # 10 second read timeout

    # Try to force TCP transport (may not work on all OpenCV builds)
    try:
        # This property might not be available in all OpenCV builds
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    except:
        pass

    # Open the stream
    cap.open(rtsp_url)
    return cap

def test_stream_connectivity(camera_id, rtsp_url):
    """
    Test if an RTSP stream is accessible and has valid video data
    """
    # Try UDP first if URL path contains 'udp', otherwise try TCP first
    if '/udp/' in rtsp_url.lower():
        transports = ['udp', 'tcp']
    else:
        transports = ['tcp', 'udp']

    for transport in transports:
        try:
            cmd = ['ffprobe', '-rtsp_transport', transport, '-analyzeduration', '5000000', '-probesize', '5000000', '-i', rtsp_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            # Check for successful connection and video stream
            if result.returncode == 0 and 'Video:' in result.stderr:
                logging.info(f"[{camera_id}] Stream accessible via {transport} transport")
                return transport
            elif 'Video:' in result.stderr and 'unspecified size' not in result.stderr:
                # Sometimes ffprobe returns non-zero but still finds the stream
                logging.info(f"[{camera_id}] Stream found via {transport} transport (with warnings)")
                return transport

        except subprocess.TimeoutExpired:
            logging.warning(f"[{camera_id}] {transport} transport test timed out")
            continue
        except Exception as e:
            logging.warning(f"[{camera_id}] {transport} transport test failed: {e}")
            continue

    logging.error(f"[{camera_id}] Stream not accessible via any transport method")
    return None

def record_with_ffmpeg(camera_id, rtsp_url, filename, duration_seconds, transport='tcp'):
    """
    Alternative recording method using ffmpeg subprocess (more reliable for RTSP)
    """
    try:
        # Enhanced ffmpeg command with better error handling for problematic streams
        cmd = [
            'ffmpeg',
            '-rtsp_transport', transport,
            '-fflags', '+genpts',            # Generate timestamps
            '-analyzeduration', '10000000',   # Reduced from 30M to 10M for faster startup
            '-probesize', '10000000',         # Reduced from 50M to 10M
            '-max_delay', '500000',           # Max 0.5s delay
            '-reorder_queue_size', '0',       # Disable reordering queue
            '-i', rtsp_url,
            '-t', str(duration_seconds),
            '-an',  # no audio
            '-c:v', 'libx264',                # Force re-encoding instead of copy
            '-preset', 'ultrafast',           # Fastest encoding
            '-crf', '28',                     # Slightly lower quality for speed
            '-r', '15',                       # Force framerate to 15 fps
            '-g', '30',                       # GOP size
            '-keyint_min', '15',              # Minimum keyframe interval
            '-sc_threshold', '0',             # Scene change threshold
            '-avoid_negative_ts', 'make_zero', # Handle timestamp issues
            '-vsync', 'cfr',                  # Constant frame rate
            '-pix_fmt', 'yuv420p',            # Ensure compatible pixel format
            '-f', 'mp4',                      # Force MP4 format
            '-movflags', '+faststart',        # Optimize for streaming
            '-y',                             # overwrite output file
            filename
        ]

        logging.info(f"[{camera_id}] Starting ffmpeg recording with re-encoding...")

        # Run ffmpeg with timeout
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 30)

        # Check if recording was successful
        if process.returncode == 0 and os.path.exists(filename):
            file_size = os.path.getsize(filename)
            if file_size > 50000:  # At least 50KB for a valid 10-second video
                logging.info(f"[{camera_id}] Successfully recorded {filename} ({file_size} bytes)")
                return True
            else:
                logging.warning(f"[{camera_id}] File too small ({file_size} bytes), likely corrupted")
                if os.path.exists(filename):
                    os.remove(filename)
                return False
        else:
            logging.error(f"[{camera_id}] ffmpeg failed. Return code: {process.returncode}")
            logging.error(f"[{camera_id}] stderr: {process.stderr}")
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
    """
    Determine if we should use ffmpeg fallback based on URL characteristics
    """
    # Always use ffmpeg for UDP streams or streams that had MPEG4 timebase issues
    if '/udp/' in rtsp_url.lower():
        logging.info(f"[{camera_id}] Using ffmpeg fallback for UDP stream")
        return True

    # Use ffmpeg for known problematic credentials/cameras
    if 'admin:test123456' in rtsp_url:
        logging.info(f"[{camera_id}] Using ffmpeg fallback for known problematic stream")
        return True

    return False

def recorder_thread(camera_id, rtsp_url):
    retry = 0
    use_ffmpeg_fallback = should_use_ffmpeg_fallback(camera_id, rtsp_url)
    stream_transport = 'tcp'  # Default transport
    broken_stream = False

    # Test stream connectivity first
    logging.info(f"[{camera_id}] Testing stream connectivity...")
    detected_transport = test_stream_connectivity(camera_id, rtsp_url)

    if detected_transport is None:
        logging.error(f"[{camera_id}] Stream appears to be completely inaccessible, marking as broken")
        broken_stream = True
    else:
        stream_transport = detected_transport

    while running and not broken_stream:
        if use_ffmpeg_fallback:
            # Use ffmpeg for more reliable RTSP handling
            logging.info(f"[{camera_id}] Using ffmpeg fallback for {rtsp_url}")
            chunk_start = datetime.now(timezone.utc).replace(microsecond=0)
            filename = make_filename(camera_id, chunk_start)

            success = record_with_ffmpeg(camera_id, rtsp_url, filename, CHUNK_SECONDS, stream_transport)

            if success and os.path.exists(filename) and os.path.getsize(filename) > 50000:
                chunk_end = datetime.now(timezone.utc).replace(microsecond=0)
                add_segment(camera_id, filename, chunk_start, chunk_end)
                retry = 0
            else:
                retry += 1
                if os.path.exists(filename):
                    os.remove(filename)  # Remove empty/corrupted file

                # Check if this is a permanently broken stream
                if retry >= MAX_RETRY * 3:
                    logging.error(f"[{camera_id}] Stream appears permanently broken, stopping recorder")
                    broken_stream = True
                    break

                if retry >= MAX_RETRY:
                    logging.error(f"[{camera_id}] Max retries reached. Backing off 30s.")
                    time.sleep(30)
                    retry = 0
                else:
                    time.sleep(5)
            continue

        # Try OpenCV approach for non-problematic streams
        logging.info(f"[{camera_id}] Connecting to {rtsp_url}")
        cap = create_opencv_capture(rtsp_url)

        if not cap.isOpened():
            logging.warning(f"[{camera_id}] Failed to open stream with OpenCV, trying ffmpeg fallback")
            cap.release()
            use_ffmpeg_fallback = True
            continue

        retry = 0

        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_HEIGHT)

        # Use mp4v codec instead of XVID for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)

        # start a new chunk
        chunk_start = datetime.now(timezone.utc).replace(microsecond=0)
        filename = make_filename(camera_id, chunk_start)
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        if not writer.isOpened():
            logging.error(f"[{camera_id}] Failed to initialize VideoWriter, switching to ffmpeg")
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
                if consecutive_failures > 10:  # Allow some frame drops before reconnecting
                    logging.warning(f"[{camera_id}] Too many frame read failures, reconnecting...")
                    break
                time.sleep(0.1)  # Brief pause before trying again
                continue

            consecutive_failures = 0  # Reset failure counter on successful read

            now_ts = datetime.now(timezone.utc).replace(microsecond=0)
            # overlay timestamp
            ts_text = now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            cv2.rectangle(frame, (10, 10), (350, 35), (0, 0, 0), -1)
            cv2.putText(frame, ts_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

            writer.write(frame)
            frames_written += 1

            # check chunk duration
            if (now_ts - chunk_start).total_seconds() >= CHUNK_SECONDS:
                chunk_end = now_ts
                writer.release()

                # Verify file was created successfully
                if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                    add_segment(camera_id, filename, chunk_start, chunk_end)
                    logging.info(f"[{camera_id}] Saved {filename}")
                else:
                    logging.warning(f"[{camera_id}] Failed to create valid video file, switching to ffmpeg")
                    use_ffmpeg_fallback = True
                    break

                # start new chunk
                chunk_start = now_ts
                filename = make_filename(camera_id, chunk_start)
                writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

                if not writer.isOpened():
                    logging.error(f"[{camera_id}] Failed to reinitialize VideoWriter")
                    break

                frames_written = 0

        # close partial chunk
        if writer and writer.isOpened():
            chunk_end = datetime.now(timezone.utc).replace(microsecond=0)
            writer.release()

            # Only add to database if file exists and has content
            if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                add_segment(camera_id, filename, chunk_start, chunk_end)
            else:
                logging.warning(f"[{camera_id}] Removing empty/corrupted file: {filename}")
                if os.path.exists(filename):
                    os.remove(filename)

        cap.release()

        if running and not broken_stream:
            retry += 1
            if retry >= MAX_RETRY and not use_ffmpeg_fallback:
                logging.info(f"[{camera_id}] Switching to ffmpeg fallback after max retries")
                use_ffmpeg_fallback = True
                retry = 0
            else:
                logging.info(f"[{camera_id}] Reconnecting in 5s... (retry {retry})")
                time.sleep(5)

    if broken_stream:
        logging.error(f"[{camera_id}] Recorder stopped due to permanently broken stream.")
    else:
        logging.info(f"[{camera_id}] Recorder stopped.")

# --------------- Main ----------------
def main():
    init_db()

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logging.info("ffmpeg is available for fallback recording")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("ffmpeg not found - only OpenCV recording will be used")

    # fetch all sensors + RTSP urls
    logging.info("Fetching sensor connection info from backend...")
    sensors = fetch_all_sensor_connection_info()
    camera_list = []
    for s in sensors:
        cam_id = s["id"]
        info = s.get("general_info", {})
        if not info or "host" not in info or not info["host"]:
            logging.warning(f"[{cam_id}] Missing host info. Skipping sensor.")
            continue
        rtsp_url = build_rtsp_url(s)
        camera_list.append((cam_id, rtsp_url))
        logging.info(f"[{cam_id}] URL built.")

    # start recorder threads
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
