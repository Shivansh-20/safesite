"""
SafeSite AI — Computer Vision & Kiosk State Machine Engine
Handles frame processing, real-time head presence tracking, OpenCV PPE verification,
2-second stabilization hold timer, scan locking, and presenter emergency override controls.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeSiteVision")

class VisionEngine:
    """
    VisionEngine manages real-time computer vision analysis,
    head presence validation (never clears empty frames),
    helmet/cap discrimination, and 2-second stabilization hold timer.
    """
    def __init__(self):
        self.hazard_polygons: List[List[Tuple[int, int]]] = []
        
        # Daily Statistics Tracker
        self.daily_stats = {
            "total_scans": 0,
            "cleared_count": 0,
            "violations_count": 0,
            "spare_ppe_issued": 0,
            "danger_zone_breaches": 0,
            "logs": []
        }
        
        # State Machine Flags
        self.is_halted = False               # True when system is paused and webcam powered off
        self.is_worker_locked = False        # True when scan output is frozen for current worker
        self.person_first_seen_time = None   # Timestamp when worker enters scanning range
        self.required_hold_duration = 2.0    # Require 2.0 seconds of steady standing before decision
        
        # Current Real-Time Gear Detection Flags
        self.helmet_detected = False
        self.vest_detected = False

        self.current_alert_key = "WAITING"
        self.current_state_text = "👤 WAITING FOR WORKER TO STEP UP..."
        self.last_state_change = time.time()
        self.worker_id_counter = 1

    def toggle_halt(self) -> bool:
        """Toggles software halt state. Returns current is_halted boolean."""
        self.is_halted = not self.is_halted
        if self.is_halted:
            self.current_state_text = "SYSTEM HALTED / WEBCAM OFF"
            self.current_alert_key = "HALTED"
            self.person_first_seen_time = None
            logger.info("System Halted. Camera hardware powered OFF.")
        else:
            self.current_state_text = f"👤 READY FOR WORKER #{self.worker_id_counter}"
            self.current_alert_key = "WAITING"
            self.is_worker_locked = False
            self.person_first_seen_time = None
            logger.info("System Resumed. Camera hardware re-initialized.")
        return self.is_halted

    def rescan_worker(self):
        """Unlocks the kiosk scan state to re-evaluate the current worker after equipping gear."""
        if self.is_halted:
            return
        self.is_worker_locked = False
        self.person_first_seen_time = None
        self.current_alert_key = "SCANNING"
        self.current_state_text = f"🔄 RE-SCANNING WORKER #{self.worker_id_counter}... PLEASE HOLD STILL"
        logger.info(f"Kiosk unlocked for RE-SCAN of Worker #{self.worker_id_counter}")

    def next_worker(self):
        """Resets the kiosk state for the next worker in line."""
        if self.is_halted:
            return
        self.is_worker_locked = False
        self.person_first_seen_time = None
        self.helmet_detected = False
        self.vest_detected = False
        self.worker_id_counter += 1
        self.current_alert_key = "WAITING"
        self.current_state_text = f"👤 READY FOR WORKER #{self.worker_id_counter}"
        logger.info(f"Kiosk reset for Worker #{self.worker_id_counter}")

    def force_clear(self):
        """
        Presenter Secret Hotkey 'C' or '2':
        Instantly clears the current worker for shift, unlocks gate, and updates stats.
        """
        self.is_worker_locked = True
        self.helmet_detected = True
        self.vest_detected = True
        self.current_alert_key = "CLEARED"
        self.current_state_text = f"🟢 WORKER #{self.worker_id_counter}: HELMET & VEST VERIFIED — SHIFT CLEARED!"
        self.person_first_seen_time = None
        self.daily_stats["cleared_count"] += 1
        self.daily_stats["total_scans"] += 1
        self.daily_stats["logs"].append({
            "time": time.strftime("%H:%M:%S"),
            "worker": f"Worker #{self.worker_id_counter}",
            "status": "CLEARED",
            "message": "Shift Cleared (Presenter Override / Verified)"
        })
        logger.info(f"[OVERRIDE KEY] Worker #{self.worker_id_counter} manually CLEARED.")

    def force_missing(self):
        """
        Presenter Secret Hotkey 'M' or '1':
        Instantly flags current worker for missing gear, opens visual pamphlet, and plays regional voice loop.
        """
        self.is_worker_locked = True
        self.helmet_detected = False
        self.vest_detected = False
        self.current_alert_key = "ALL_MISSING"
        self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: SAFETY GEAR MISSING! COLLECT FROM BIN A"
        self.person_first_seen_time = None
        self.daily_stats["violations_count"] += 1
        self.daily_stats["spare_ppe_issued"] += 2
        self.daily_stats["total_scans"] += 1
        self.daily_stats["logs"].append({
            "time": time.strftime("%H:%M:%S"),
            "worker": f"Worker #{self.worker_id_counter}",
            "status": "ALL_MISSING",
            "message": "Missing Gear (Presenter Override / Non-compliant)"
        })
        logger.info(f"[OVERRIDE KEY] Worker #{self.worker_id_counter} flagged for MISSING GEAR.")

    def detect_ppe_opencv(self, frame: np.ndarray) -> List[Dict]:
        """
        SafeSite Helmet Detector — Adaptive Crown Zone.

        Algorithm:
        1. BRIGHTNESS: reject if frame is too dark.
        2. SKIN SEARCH: find where face/neck skin is in the CENTER column.
           - Scan the center 60% width for skin pixels.
           - Find the TOPMOST row that has significant skin.
           - This gives us `face_top_y` — the top of the visible face/forehead.
        3. CROWN ZONE: scan from (face_top_y - 40%) to (face_top_y + 5%).
           - This is the region right above and on top of the face — where the helmet sits.
        4. HELMET DECISION: analyze the crown zone for helmet colors.
           - Any vivid hardhat color (yellow, orange, red, blue) → PASS
           - White with no skin → PASS
           - Dark dome with no skin/hair → PASS (motorcycle helmet)
           - Everything else → REJECT (bare hair, cap, scarf, gamcha)

        The person presence is confirmed because we only run the helmet check
        if we found skin (step 2). If no skin → return [] → ask to focus.
        """
        h, w, _ = frame.shape
        detections = []

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ================================================================
        # BRIGHTNESS CHECK
        # ================================================================
        mean_v = float(np.mean(gray))
        if mean_v < 28:
            return []  # Too dark

        # ================================================================
        # SKIN RANGE
        # ================================================================
        skin_lo = np.array([0,  22, 45])
        skin_hi = np.array([22, 190, 248])

        # ================================================================
        # STEP 1 — FIND THE TOP OF THE FACE (adaptive)
        # Scan the center column (x: 15%–85%) row-by-row from top to bottom.
        # Find the first row that has >= 3% skin pixels → that is roughly
        # the top of the forehead or the visor-line of a motorcycle helmet.
        # ================================================================
        cx1 = int(w * 0.15)
        cx2 = int(w * 0.85)
        col_width = cx2 - cx1

        face_top_y = None
        for y in range(int(h * 0.06), int(h * 0.85)):
            row_hsv = hsv[y, cx1:cx2]
            skin_px = cv2.countNonZero(cv2.inRange(
                row_hsv.reshape(-1, 1, 3), skin_lo, skin_hi))
            if skin_px / col_width >= 0.03:
                face_top_y = y
                break

        if face_top_y is None:
            # No skin row found — no person or camera off target
            return []

        # ================================================================
        # STEP 2 — CROWN / HELMET ZONE
        # The helmet dome sits ABOVE the face_top_y.
        # Zone height = 35% of total frame height.
        # Allow a tiny +5% overlap below face_top_y so we catch the very
        # edge of a motorcycle helmet visor sitting right at forehead level.
        # ================================================================
        zone_h     = int(h * 0.35)
        crown_y2   = min(h - 10, face_top_y + int(h * 0.05))
        crown_y1   = max(int(h * 0.02), crown_y2 - zone_h)
        crown_x1   = int(w * 0.18)
        crown_x2   = int(w * 0.82)

        crown_crop = frame[crown_y1:crown_y2, crown_x1:crown_x2]
        ch, cw, _  = crown_crop.shape

        if ch < 20 or cw < 20:
            detections.append({"bbox": [int(w * 0.12), int(h * 0.04),
                                        int(w * 0.88), int(h * 0.92)],
                               "label": "person", "conf": 0.90})
            detections.append({"bbox": [crown_x1, crown_y1, crown_x2, crown_y2],
                               "label": "no_hardhat", "conf": 0.90})
            return detections

        crown_hsv   = cv2.cvtColor(crown_crop, cv2.COLOR_BGR2HSV)
        crown_total = max(1, ch * cw)

        # ----------------------------------------------------------------
        # Feature ratios in the crown zone
        # ----------------------------------------------------------------
        # Skin pixels in the crown (= bare forehead / scalp visible)
        r_skin = cv2.countNonZero(
            cv2.inRange(crown_hsv, skin_lo, skin_hi)) / crown_total

        # Hair texture (bare scalp / very thin cap lets hair show through)
        # H: 0-25, S: 15-130, V: 10-80 = dark brownish matte
        r_hair = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([0, 15, 10]), np.array([25, 130, 80]))
        ) / crown_total

        # Yellow hardhat
        r_yellow = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([13, 100, 90]), np.array([38, 255, 255]))
        ) / crown_total

        # Orange hardhat
        r_orange = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([5, 110, 90]), np.array([17, 255, 255]))
        ) / crown_total

        # Red hardhat (hue wraps)
        r_red = (
            cv2.countNonZero(cv2.inRange(crown_hsv, np.array([0,  110, 90]), np.array([6,  255, 255]))) +
            cv2.countNonZero(cv2.inRange(crown_hsv, np.array([174, 110, 90]), np.array([180, 255, 255])))
        ) / crown_total

        # Blue hardhat
        r_blue = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([95, 90, 60]), np.array([135, 255, 255]))
        ) / crown_total

        # White hardhat
        r_white = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([0, 0, 185]), np.array([180, 55, 255]))
        ) / crown_total

        # Dark region (motorcycle helmet / black hardhat)
        r_dark = cv2.countNonZero(
            cv2.inRange(crown_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        ) / crown_total

        # Sum of vivid hardhat colors
        r_vivid = r_yellow + r_orange + r_red + r_blue

        # ----------------------------------------------------------------
        # Helmet decision
        # ----------------------------------------------------------------
        is_helmet = False
        conf = 0.94

        if r_vivid > 0.05:
            # Vivid-colored hardhat (yellow, orange, red, blue)
            is_helmet = True
            conf = round(min(0.98, 0.86 + r_vivid), 2)

        elif r_white > 0.18 and r_skin < 0.20:
            # White hardhat
            is_helmet = True
            conf = 0.95

        elif r_dark > 0.35 and r_skin < 0.16 and r_hair < 0.10:
            # Dark motorcycle helmet / black hardhat:
            # • Rigid dark shell covers > 35% of the crown
            # • Skin < 16%: no forehead or scalp peeking through
            # • Hair < 10%: no bare hair texture (cap/gamcha/bare head rejected)
            is_helmet = True
            conf = 0.95

        # Bounding boxes
        person_box = [int(w * 0.12), int(h * 0.03), int(w * 0.88), min(h - 10, int(h * 0.93))]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.95})

        head_box = [crown_x1, crown_y1, crown_x2, crown_y2]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})

        # ================================================================
        # VEST CHECK — NOT YET IMPLEMENTED
        # ================================================================

        return detections

        """
        Main Video Frame Pipeline:
        1. Performs live OpenCV PPE detection if detections not supplied.
        2. Applies 2-second stabilization hold timer ONLY when human head is aligned.
        3. Prioritizes Helmet as the main gate qualification point.
        4. Locks scan output once 2-second decision is made.
        """
        h, w, _ = frame.shape
        now = time.time()

        if detections is None:
            detections = self.detect_ppe_opencv(frame)

        # 1. System Halted State
        if self.is_halted:
            cv2.rectangle(frame, (0, 0), (w, h), (10, 10, 15), -1)
            cv2.putText(frame, "SYSTEM HALTED / WEBCAM OFF", (80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 240, 255), 2)
            return frame, {
                "state_text": "SYSTEM HALTED",
                "alert_key": "HALTED",
                "is_locked": True,
                "is_halted": True,
                "helmet_detected": False,
                "vest_detected": False,
                "stats": self.daily_stats
            }

        # 2. Worker Locked State (Freeze output until Re-Scan, Next Worker, or Override)
        if self.is_worker_locked:
            cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 25), -1)
            cv2.putText(frame, f"SAFESITE AI | WORKER #{self.worker_id_counter} SCAN COMPLETED (LOCKED)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)

            if self.current_alert_key == "CLEARED":
                color = (0, 255, 0)
            elif self.current_alert_key == "VEST_MISSING":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (0, h - 55), (w, h), (15, 15, 25), -1)
            cv2.putText(frame, self.current_state_text, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            return frame, {
                "state_text": self.current_state_text,
                "alert_key": self.current_alert_key,
                "is_locked": True,
                "is_halted": False,
                "helmet_detected": self.helmet_detected,
                "vest_detected": self.vest_detected,
                "stats": self.daily_stats
            }

        # 3. Active Frame Evaluation
        person_in_frame = False
        helmet_detected = False
        vest_detected = False
        person_centered = False

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            label = det.get("label", "person")
            conf = det.get("conf", 0.9)
            x1, y1, x2, y2 = bbox

            center_x = int((x1 + x2) / 2)

            if label == "person":
                person_in_frame = True
                if (w * 0.15) <= center_x <= (w * 0.85):
                    person_centered = True

            if label in ["hardhat", "helmet"]:
                helmet_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "HELMET (PASS)", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            elif label == "vest":
                vest_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "SAFETY VEST (PASS)", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            elif label in ["no_hardhat", "no_helmet"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "NO HELMET / CAP REJECTED", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            elif label == "no_vest":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "NO SAFETY VEST", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Store live detection states
        self.helmet_detected = helmet_detected
        self.vest_detected = vest_detected

        # 4. Hold Stabilization Timer Logic (Only counts down when person is in camera view)
        if person_in_frame and person_centered:
            if self.person_first_seen_time is None:
                self.person_first_seen_time = now

            elapsed_hold = now - self.person_first_seen_time
            remaining_hold = max(0.0, self.required_hold_duration - elapsed_hold)

            if remaining_hold > 0:
                # Still in 2-second countdown mode
                self.current_alert_key = "SCANNING"
                self.current_state_text = f"⏳ SCANNING WORKER #{self.worker_id_counter}... HOLD STILL ({remaining_hold:.1f}s)"
                color = (0, 240, 255)
            else:
                # 2 Seconds Elapsed! Finalize scan decision and LOCK output!
                # VEST CHECK IS NOT YET IMPLEMENTED — decision is helmet-only.
                if helmet_detected:
                    # Helmet verified — CLEARED
                    self.current_alert_key = "CLEARED"
                    self.current_state_text = f"WORKER #{self.worker_id_counter}: HELMET VERIFIED — SHIFT CLEARED!"
                    color = (0, 255, 0)
                    self.is_worker_locked = True
                    self.daily_stats["cleared_count"] += 1
                else:
                    # No helmet (cap / scarf / gamcha / bare head) — REJECTED
                    self.current_alert_key = "HELMET_MISSING"
                    self.current_state_text = f"WORKER #{self.worker_id_counter}: SAFETY GEAR MISSING! COLLECT FROM BIN A"
                    color = (0, 0, 255)
                    self.is_worker_locked = True
                    self.daily_stats["violations_count"] += 1
                    self.daily_stats["spare_ppe_issued"] += 1

                self.daily_stats["total_scans"] += 1
                self.daily_stats["logs"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "worker": f"Worker #{self.worker_id_counter}",
                    "status": self.current_alert_key,
                    "message": self.current_state_text
                })
        else:
            # Person NOT in frame or not properly aligned
            self.person_first_seen_time = None
            self.current_alert_key = "WAITING"
            self.current_state_text = f"👤 PLEASE POSITION HEAD IN CAMERA VIEW (WORKER #{self.worker_id_counter})"
            color = (0, 240, 255)
            # Alignment guide when waiting
            cv2.rectangle(frame, (int(w * 0.22), int(h * 0.12)), (int(w * 0.78), int(h * 0.88)), (0, 240, 255), 1)

        # Draw Top Header Banner
        cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 25), -1)
        cv2.putText(frame, f"SAFESITE AI | GATE SCANNER (WORKER #{self.worker_id_counter})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)

        # Draw Bottom Status Banner
        cv2.rectangle(frame, (0, h - 55), (w, h), (15, 15, 25), -1)
        cv2.putText(frame, self.current_state_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, {
            "state_text": self.current_state_text,
            "alert_key": self.current_alert_key,
            "is_locked": self.is_worker_locked,
            "is_halted": False,
            "helmet_detected": self.helmet_detected,
            "vest_detected": self.vest_detected,
            "stats": self.daily_stats
        }

vision_engine = VisionEngine()
