##This has full motion-based clip saving functionality, and all logging stuff implemented. Just this doesn't fetch the cam data from backend, uses hardcoded just for debugging. 
#!/usr/bin/env python3
"""
- Fetches *all sensors* for an institution from RADR backend (Supabase API)
- Builds RTSP URLs for each camera
- Spawns recording threads (with time-chunked local storage + timestamp overlay + SQLite index)
- Saves a chunk only when motion is detected to save space and compute. 
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
from datetime import datetime, timezone, timedelta

# --------------- CONFIG -----------------
from dotenv import load_dotenv
load_dotenv()

INSTITUTION_ID = os.getenv("INSTITUTION_ID")
BEACON_ID = int(os.getenv("BEACON_ID"))
API_BASE_URL = os.getenv("API_BASE_URL")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", 300))
OUTPUT_DIR = "./recordings"
DB_PATH = os.path.join(OUTPUT_DIR, "index.db")

# Motion-triggered recording (OpenCV path only)
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", 400_000))  # tune per cam
MOTION_MIN_FRAMES = int(os.getenv("MOTION_MIN_FRAMES", 2))      # avoid single-frame spikes
RECORD_SECONDS_AFTER_MOTION = int(os.getenv("RECORD_SECONDS_AFTER_MOTION", 60))  # ?solid minute?
MOTION_COOLDOWN_SEC = int(os.getenv("MOTION_COOLDOWN_SEC", 2))  # allow quick re-triggers

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
    Debug version ? skips backend call and returns hardcoded camera data
    identical in structure to backend response.
    """
    logging.info("?? Using hardcoded sensor data (debug mode, no backend)")
    return [
        {
            "id": 1,
            "general_info": {
                "host": "192.168.1.179",
                "path": "/stream1",
                "port": 554,
                "type": "RTSP",
                "params": {"fps": 30, "resolution": "1080p"}
            },
            "credentials": {"username": "radrbonappletea", "password": "Testing123!"}
        },
        {
            "id": 5,
            "general_info": {
                "host": "192.168.86.31",
                "path": "/udp/av0_0",
                "port": "10554",
                "type": "RTSP"
            },
            "credentials": {"username": "admin", "password": "test123456"}
        },
        {
            "id": 6,
            "general_info": {
                "host": "192.168.1.179",
                "path": "/stream1",
                "port": 554,
                "type": "RTSP",
                "params": {"fps": 30, "resolution": "1080p"}
            },
            "credentials": {"username": "radrbonappletea", "password": "Testing123!"}
        },
        {
            "id": 7,
            "general_info": {
                "host": "192.168.1.179",
                "path": "/stream1",
                "port": 554,
                "type": "RTSP",
                "params": {"fps": 30, "resolution": "1080p"}
            },
            "credentials": {"username": "radrbonappletea", "password": "Testing123!"}
        }
    ]
def build_rtsp_url(sensor_data):
    """
    Construct RTSP URL from a single sensor's API data.
    """
    host = sensor_data["general_info"]["host"]
    path = sensor_data["general_info"]["path"]
    port = sensor_data["general_info"]["port"]
    username = sensor_data["credentials"].get("username") or "admin"
    password = sensor_data["credentials"].get("password") or "admin"

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

def motion_score(prev_bgr, curr_bgr):
    """Return a simple per-frame motion score using absdiff -> gray -> nonzeros."""
    diff = cv2.absdiff(prev_bgr, curr_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return int(cv2.countNonZero(gray))

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
            logging.info(f"[{camera_id}] (FFmpeg mode) Waiting for motion before recording...")

            cap = create_opencv_capture(rtsp_url)
            if not cap.isOpened():
                logging.error(f"[{camera_id}] Cannot open stream even for motion detection.")
                time.sleep(10)
                continue

            prev_frame = None
            motion_hits = 0
            recording = False
            record_proc = None
            record_until_ts = 0
            consecutive_failures = 0

            last_log_time = 0  # persist across frames

            while running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logging.warning(f"[{camera_id}] Too many frame read failures in motion loop.")
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0

                now_ts = datetime.now(timezone.utc).replace(microsecond=0)

                # --- Motion detection logic ---
                if prev_frame is not None:
                    score = motion_score(prev_frame, frame)

                    # ? Heartbeat log every 10 seconds per camera
                    if time.time() - last_log_time >= 10:
                        logging.info(f"[{camera_id}] Motion score: {score}")
                        last_log_time = time.time()

                    # --- Motion detection decision logic ---
                    if not recording and score > MOTION_THRESHOLD:
                        motion_hits += 1
                    elif recording and score > MOTION_THRESHOLD:
                        record_until_ts = now_ts.timestamp() + MOTION_COOLDOWN_SEC
                    else:
                        motion_hits = max(0, motion_hits - 1)

                    if not recording and motion_hits >= MOTION_MIN_FRAMES:
                                    recording = True
                                    motion_hits = 0
                                    start_ts = now_ts
                                    filename = make_filename(camera_id, start_ts)

                                    logging.info(f"[{camera_id}] ? Motion detected (score?{score}). Starting FFmpeg recording -> {filename}")

                                    # Stop reading RTSP (free the stream)
                                    cap.release()

                                    # Run ffmpeg to record 1 full minute
                                    try:
                                        subprocess.run([
                                            'ffmpeg', '-rtsp_transport', stream_transport,
                                            '-fflags', '+genpts',
                                            '-analyzeduration', '10000000',
                                            '-probesize', '10000000',
                                            '-max_delay', '500000',
                                            '-reorder_queue_size', '0',
                                            '-i', rtsp_url,
                                            '-t', str(RECORD_SECONDS_AFTER_MOTION),
                                            '-an',
                                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                                            '-r', '15', '-g', '30', '-sc_threshold', '0',
                                            '-pix_fmt', 'yuv420p', '-f', 'mp4',
                                            '-movflags', '+faststart', '-y', filename
                                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           timeout=RECORD_SECONDS_AFTER_MOTION + 20)
                                    except subprocess.TimeoutExpired:
                                        logging.error(f"[{camera_id}] FFmpeg timed out; no clip created")

                                    # Validate saved file
                                    end_ts = start_ts + timedelta(seconds=RECORD_SECONDS_AFTER_MOTION)
                                    if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                                        add_segment(camera_id, filename, start_ts, end_ts)
                                        logging.info(f"[{camera_id}] ? Saved motion-triggered clip: {filename}")
                                    else:
                                        if os.path.exists(filename):
                                            os.remove(filename)
                                        logging.warning(f"[{camera_id}] Discarded empty FFmpeg clip.")

                                    # Reopen stream for further motion detection
                                    cap = create_opencv_capture(rtsp_url)
                                    if not cap.isOpened():
                                        logging.error(f"[{camera_id}] Failed to reopen stream; will retry outer loop.")
                                        break

                                    prev_frame = None
                                    last_log_time = 0
                                    consecutive_failures = 0
                                    recording = False

                                    continue  # resume detection

                prev_frame = frame

            cap.release()

            continue
            # Try OpenCV approach for non-problematic streams (MOTION-GATED)
            logging.info(f"[{camera_id}] (OpenCV mode) Waiting for motion before recording...")
            cap = create_opencv_capture(rtsp_url)

            if not cap.isOpened():
                logging.warning(f"[{camera_id}] Failed to open stream with OpenCV, trying ffmpeg fallback")
                cap.release()
                use_ffmpeg_fallback = True
                continue

            retry = 0

            # Get stream properties for potential recording
            fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_WIDTH)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_HEIGHT)
            fourcc = cv2.VideoWriter_fourcc(*FOURCC)

            # --- Motion-gated state ---
            prev_frame = None
            motion_hits = 0
            recording = False
            record_until_ts = 0
            last_log_time = 0
            consecutive_failures = 0
            writer = None

            while running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logging.warning(f"[{camera_id}] Too many frame read failures in motion loop (OpenCV). Reconnecting...")
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0

                now_ts = datetime.now(timezone.utc).replace(microsecond=0)

                # --- Motion detection + heartbeat ---
                if prev_frame is not None:
                    score = motion_score(prev_frame, frame)

                    # Heartbeat every 10s
                    if time.time() - last_log_time >= 10:
                        logging.info(f"[{camera_id}] Motion score: {score} [OpenCV]")
                        last_log_time = time.time()

                    # Decision logic
                    if not recording and score > MOTION_THRESHOLD:
                        motion_hits += 1
                    elif recording and score > MOTION_THRESHOLD:
                        record_until_ts = now_ts.timestamp() + MOTION_COOLDOWN_SEC
                    else:
                        motion_hits = max(0, motion_hits - 1)

                    # Start recording on confirmed motion
                    if not recording and motion_hits >= MOTION_MIN_FRAMES:
                        recording = True
                        motion_hits = 0
                        start_ts = now_ts
                        filename = make_filename(camera_id, start_ts)
                        record_until_ts = now_ts.timestamp() + RECORD_SECONDS_AFTER_MOTION
                        logging.info(f"[{camera_id}] ? Motion detected (score?{score}). Starting OpenCV recording -> {filename}")

                        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                        if not writer.isOpened():
                            logging.error(f"[{camera_id}] Failed to initialize VideoWriter; switching to ffmpeg fallback next cycle")
                            writer = None
                            # bail from this OpenCV attempt and let outer loop switch to ffmpeg on next iteration
                            break

                # Write frame only if recording
                if recording and writer is not None:
                    # overlay timestamp
                    ts_text = now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
                    cv2.rectangle(frame, (10, 10), (350, 35), (0, 0, 0), -1)
                    cv2.putText(frame, ts_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                    writer.write(frame)

                # Stop recording when cooldown expires
                if recording and now_ts.timestamp() >= record_until_ts:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    end_ts = now_ts

                    # validate file, then index or discard
                    if os.path.exists(filename) and os.path.getsize(filename) > 50000:
                        add_segment(camera_id, filename, start_ts, end_ts)
                        logging.info(f"[{camera_id}] ? Saved motion-triggered clip: {filename}")
                    else:
                        if os.path.exists(filename):
                            os.remove(filename)
                        logging.warning(f"[{camera_id}] Discarded empty/invalid OpenCV clip.")

                prev_frame = frame

            # If an OpenCV writer is still open due to loop exit, close safely
            if writer:
                writer.release()
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
