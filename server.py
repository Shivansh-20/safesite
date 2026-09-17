"""
SafeSite AI — Construction Safety Monitor Application Server
FastAPI backend for camera streaming, AI vision pipeline control,
multilingual audio voice prompts, and contractor safety reports.
"""

import os
import cv2
import time
import json
import logging
import numpy as np
from fastapi import FastAPI, Response, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from audio_engine import audio_engine
from vision_engine import vision_engine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeSiteServer")

app = FastAPI(
    title="SafeSite AI — Construction Safety Kiosk",
    description="Low-Cost AI PPE Compliance & Contractor Analytics System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Tracking Software Server Uptime (Accumulates active running seconds only)
SERVER_START_TIME = time.time()
TOTAL_ACTIVE_SECONDS = 0.0
LAST_ACTIVE_START = time.time()
camera = None

def update_active_uptime(is_halting: bool):
    """Updates accumulated active uptime seconds."""
    global TOTAL_ACTIVE_SECONDS, LAST_ACTIVE_START
    now = time.time()
    if is_halting:
        if LAST_ACTIVE_START is not None:
            TOTAL_ACTIVE_SECONDS += (now - LAST_ACTIVE_START)
            LAST_ACTIVE_START = None
    else:
        if LAST_ACTIVE_START is None:
            LAST_ACTIVE_START = now

def get_formatted_uptime() -> str:
    """Calculates active running uptime of the software (excluding paused/halted time)."""
    global TOTAL_ACTIVE_SECONDS, LAST_ACTIVE_START
    current_accumulated = TOTAL_ACTIVE_SECONDS
    if LAST_ACTIVE_START is not None:
        current_accumulated += (time.time() - LAST_ACTIVE_START)
    
    elapsed = int(current_accumulated)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

LAST_CAMERA_ATTEMPT = 0

def get_camera():
    """Initializes and returns the physical webcam capture device with retry throttling."""
    global camera, LAST_CAMERA_ATTEMPT
    now = time.time()
    if camera is not None and camera.isOpened():
        return camera

    # Only attempt to open camera every 3 seconds to avoid freezing the event loop
    if now - LAST_CAMERA_ATTEMPT > 3.0:
        LAST_CAMERA_ATTEMPT = now
        try:
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cam.isOpened():
                cam = cv2.VideoCapture(0)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera = cam
                logger.info("Physical webcam hardware initialized successfully.")
            else:
                logger.warning("Webcam not accessible right now. Using synthetic stream fallback.")
                camera = None
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            camera = None
    return camera

def release_camera():
    """Releases the camera hardware completely (turns off webcam LED)."""
    global camera
    if camera is not None:
        try:
            if camera.isOpened():
                camera.release()
                logger.info("Physical webcam hardware released and powered OFF.")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
        camera = None

def generate_frames():
    """
    MJPEG Video Stream Generator.
    Processes camera frames directly through vision_engine.
    """
    while True:
        # Check if system is halted
        if vision_engine.is_halted:
            release_camera()
            halt_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(halt_frame, (0, 0), (640, 480), (10, 10, 15), -1)
            cv2.putText(halt_frame, "SYSTEM HALTED / WEBCAM OFF", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 255), 2)
            ret, buffer = cv2.imencode('.jpg', halt_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        cap = get_camera()
        success = False
        frame = None

        if cap is not None and cap.isOpened():
            try:
                success, frame = cap.read()
            except Exception as e:
                logger.error(f"Camera read error: {e}")
                success = False

        if not success or frame is None:
            # Synthetic frame fallback
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SAFESITE AI - CAMERA ACTIVE", (130, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)

        # Process frame directly through vision engine with full helmet/vest analysis
        processed_frame, status_result = vision_engine.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SafeSite AI Application Running. index.html loading...</h1>"

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/halt_toggle")
def halt_toggle():
    """Toggles system halt state. Releases webcam hardware when halted."""
    is_halted = vision_engine.toggle_halt()
    update_active_uptime(is_halted)
    if is_halted:
        release_camera()
    return {
        "is_halted": is_halted,
        "message": "System Halted & Camera Off" if is_halted else "System Resumed"
    }

@app.post("/api/rescan_worker")
def rescan_worker():
    """Unlocks kiosk state to scan the current worker again."""
    vision_engine.rescan_worker()
    return {"success": True, "message": "Worker state unlocked for re-scan."}

@app.post("/api/next_worker")
def next_worker():
    """Resets kiosk state for the next worker in line."""
    vision_engine.next_worker()
    return {"success": True, "worker_id": vision_engine.worker_id_counter}

@app.post("/api/force_clear")
def force_clear():
    """Presenter override: Instantly clears the worker for shift (Hotkey 'C' or '2')."""
    vision_engine.force_clear()
    return {"success": True, "state": "CLEARED", "message": "Worker cleared via Presenter Key"}

@app.post("/api/force_missing")
def force_missing():
    """Presenter override: Instantly flags worker for missing gear (Hotkey 'M' or '1')."""
    vision_engine.force_missing()
    return {"success": True, "state": "ALL_MISSING", "message": "Worker flagged via Presenter Key"}

@app.get("/api/get_status")
def get_status():
    """Returns current state, analytics stats, software uptime, and audio playlist."""
    stats = vision_engine.daily_stats
    alert_key = vision_engine.current_alert_key
    is_locked = vision_engine.is_worker_locked
    is_halted = vision_engine.is_halted
    uptime = get_formatted_uptime()

    playlist = audio_engine.get_sequential_playlist(alert_key) if alert_key not in ["WAITING", "HALTED", "SCANNING"] else []

    return {
        "status": stats,
        "alert_key": alert_key,
        "state_text": vision_engine.current_state_text,
        "is_locked": is_locked,
        "is_halted": is_halted,
        "worker_id": vision_engine.worker_id_counter,
        "uptime": uptime,
        "playlist": playlist,
        "helmet_detected": vision_engine.helmet_detected,
        "vest_detected": vision_engine.vest_detected
    }

@app.get("/api/generate_report")
def generate_report():
    """Generates a complete contractor daily compliance report with uptime and audit logs."""
    stats = vision_engine.daily_stats
    total = stats["total_scans"]
    cleared = stats["cleared_count"]
    rejected = stats["violations_count"]
    spare_issued = stats["spare_ppe_issued"]
    safety_score = int((cleared / max(total, 1)) * 100) if total > 0 else 100
    uptime = get_formatted_uptime()

    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SafeSite AI — Contractor Daily Safety Report</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }}
            body {{ background: #f8fafc; color: #0f172a; padding: 40px; line-height: 1.5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 16px; border: 1px solid #cbd5e1; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
            .header {{ text-align: center; border-bottom: 3px solid #0284c7; padding-bottom: 24px; margin-bottom: 30px; }}
            .header h1 {{ font-size: 26px; color: #0f172a; margin-bottom: 6px; letter-spacing: 0.5px; }}
            .header p {{ font-size: 14px; color: #64748b; }}
            .meta-bar {{ display: flex; justify-content: space-between; background: #f1f5f9; padding: 14px 20px; border-radius: 10px; margin-bottom: 30px; font-size: 14px; font-weight: 600; color: #334155; }}
            .score-card {{ background: linear-gradient(135deg, #e0f2fe, #bae6fd); border: 2px solid #0284c7; border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 30px; }}
            .score-val {{ font-size: 52px; font-weight: 800; color: #0369a1; }}
            .score-lbl {{ font-size: 13px; font-weight: 700; text-transform: uppercase; color: #0284c7; letter-spacing: 1px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 35px; }}
            .stat-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center; }}
            .stat-num {{ font-size: 28px; font-weight: 800; color: #0f172a; margin-bottom: 4px; }}
            .stat-num.green {{ color: #16a34a; }}
            .stat-num.red {{ color: #dc2626; }}
            .stat-num.amber {{ color: #d97706; }}
            .stat-lbl {{ font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; }}
            .section-title {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 16px; border-left: 4px solid #0284c7; padding-left: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: white; font-size: 13px; }}
            th, td {{ padding: 12px 16px; border: 1px solid #e2e8f0; text-align: left; }}
            th {{ background: #0284c7; color: white; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
            tr:nth-child(even) {{ background: #f8fafc; }}
            .print-btn {{ display: block; width: 100%; padding: 14px; background: #0284c7; color: white; text-align: center; border-radius: 10px; text-decoration: none; font-weight: 700; font-size: 15px; border: none; cursor: pointer; }}
            @media print {{ .print-btn {{ display: none; }} body {{ padding: 0; background: white; }} .container {{ border: none; box-shadow: none; width: 100%; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏗️ SAFESITE AI — CONTRACTOR SAFETY AUDIT REPORT</h1>
                <p>Field-Deployable Construction PPE Compliance & Kiosk Summary</p>
            </div>

            <div class="meta-bar">
                <span>📅 Date: {time.strftime('%Y-%m-%d')}</span>
                <span>⏱️ Software Uptime: {uptime}</span>
                <span>📍 Location: Builder Site #4</span>
            </div>

            <div class="score-card">
                <div class="score-lbl">Site Safety Compliance Index</div>
                <div class="score-val">{safety_score}%</div>
            </div>

            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-num">{total}</div>
                    <div class="stat-lbl">Total Workers Scanned</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num green">{cleared}</div>
                    <div class="stat-lbl">Shift Cleared</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num red">{rejected}</div>
                    <div class="stat-lbl">Non-Compliant / Flagged</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num amber">{spare_issued}</div>
                    <div class="stat-lbl">Spare PPE Issued</div>
                </div>
            </div>

            <div class="section-title">Audit Log History</div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Worker ID</th>
                        <th>Status</th>
                        <th>Audit Details</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([f"<tr><td>{log['time']}</td><td>{log.get('worker','--')}</td><td><strong>{log['status']}</strong></td><td>{log['message']}</td></tr>" for log in stats['logs'][-20:]]) if stats['logs'] else "<tr><td colspan='4' style='text-align:center;'>No audit logs recorded yet today.</td></tr>"}
                </tbody>
            </table>

            <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF Report</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=report_html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
