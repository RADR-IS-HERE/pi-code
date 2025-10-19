#!/usr/bin/env python3
"""
Optimized Multi-Camera RTSP Recorder (RADR)
-------------------------------------------
- Fetches all sensors for this beacon
- Builds RTSP URLs
- Spawns lightweight ffmpeg subprocess recorders (copy stream mode)
- Stores short MP4 chunks + JSON/SQLite index
- Designed for Raspberry Pi 4 (8 cams) or Pi Zero 2 W (1-2 cams)
"""

import os
import time
import json
import signal
import sqlite3
import threading
import subprocess
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

# ---------- CONFIG ----------
load_dotenv()
INSTITUTION_ID = os.getenv("INSTITUTION_ID")
BEACON_ID = int(os.getenv("BEACON_ID"))
API_BASE_URL = os.getenv("API_BASE_URL")

OUTPUT_DIR = "./recordings"
DB_PATH = os.path.join(OUTPUT_DIR, "index.db")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", 60))
MAX_RETRY = 5

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- DB ----------
_db_lock = threading.Lock()

def init_db():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS segments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                filename TEXT,
                start_ts TEXT,
                end_ts TEXT
            )
        """)
        conn.commit()
        conn.close()

def add_segment(camera_id, filename, start_ts, end_ts):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute(
            "INSERT INTO segments(camera_id, filename, start_ts, end_ts) VALUES (?,?,?,?)",
            (camera_id, filename, start_ts.isoformat(), end_ts.isoformat())
        )
        conn.commit()
        conn.close()

# ---------- API ----------
def fetch_all_sensor_connection_info():
    url = f"{API_BASE_URL}/beacons/{BEACON_ID}/sensors/connection_data"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def build_rtsp_url(sensor):
    g = sensor.get("general_info", {})
    c = sensor.get("credentials", {})
    host, path, port = g.get("host"), g.get("path", ""), g.get("port", 554)
    user = c.get("username") or "admin"
    pwd = c.get("password") or "admin"
    return f"rtsp://{user}:{pwd}@{host}:{port}{path}"

# ---------- GLOBAL CONTROL ----------
running = True
def stop_all(signum=None, frame=None):
    global running
    running = False
    logging.warning("Stopping all recorders...")
signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

def make_filename(cam_id, start_ts):
    ts = start_ts.strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(OUTPUT_DIR, f"{cam_id}_{ts}.mp4")

# ---------- RECORDING ----------
def record_loop(cam_id, rtsp_url):
    retry = 0
    while running:
        start = datetime.now(timezone.utc).replace(microsecond=0)
        filename = make_filename(cam_id, start)
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-stimeout", "5000000",
            "-i", rtsp_url,
            "-t", str(CHUNK_SECONDS),
            "-an",
            "-c", "copy",                # ⚡ no re-encoding
            "-f", "mp4",
            "-movflags", "+faststart",
            "-y",
            filename
        ]
        try:
            logging.info(f"[{cam_id}] Recording chunk → {filename}")
            proc = subprocess.run(cmd, capture_output=True, text=False,
                                  timeout=CHUNK_SECONDS + 10)
            if proc.returncode == 0 and os.path.exists(filename):
                if os.path.getsize(filename) > 50_000:
                    end = datetime.now(timezone.utc).replace(microsecond=0)
                    add_segment(cam_id, filename, start, end)
                    retry = 0
                    logging.info(f"[{cam_id}] Chunk saved ({os.path.getsize(filename)//1024} KB)")
                else:
                    logging.warning(f"[{cam_id}] Empty file; deleting")
                    os.remove(filename)
                    retry += 1
            else:
                logging.error(f"[{cam_id}] ffmpeg error (code {proc.returncode})")
                retry += 1
        except subprocess.TimeoutExpired:
            logging.error(f"[{cam_id}] ffmpeg timeout")
            retry += 1
        except Exception as e:
            logging.error(f"[{cam_id}] Exception: {e}")
            retry += 1

        if retry >= MAX_RETRY:
            logging.error(f"[{cam_id}] Too many errors → back-off 30 s")
            time.sleep(30)
            retry = 0
        else:
            time.sleep(1)

    logging.info(f"[{cam_id}] Recorder stopped")

# ---------- MAIN ----------
def main():
    init_db()
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except FileNotFoundError:
        logging.error("ffmpeg not installed; exiting.")
        return

    logging.info("Fetching sensor data...")
    sensors = fetch_all_sensor_connection_info()
    cameras = []
    for s in sensors:
        info = s.get("general_info", {})
        if not info or not info.get("host"):
            logging.warning(f"[{s.get('id')}] Missing host info → skip")
            continue
        rtsp = build_rtsp_url(s)
        cameras.append((s["id"], rtsp))
        logging.info(f"[{s['id']}] RTSP built.")

    threads = []
    for cam_id, url in cameras:
        t = threading.Thread(target=record_loop, args=(cam_id, url), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)  # small stagger to avoid I/O spike

    logging.info(f"Recording started for {len(threads)} cameras.")
    while running:
        time.sleep(1)
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
