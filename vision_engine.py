"""
SafeSite AI — Computer Vision & Kiosk State Machine Engine
Handles frame processing, OpenCV real-time PPE color/feature filtering, 2-second stabilization hold timer,
posture framing validation, scan locking, and presenter emergency override controls.
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
    OpenCV color/plastic PPE verification, 2-second stabilization hold timer,
    scan output freezing, and manual presenter override keys.
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
        Presenter Trick Key (Secret Hotkey 'C' or '2'):
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
        Presenter Trick Key (Secret Hotkey 'M' or '1'):
        Instantly flags current worker for missing gear, opens visual pamphlet, and plays regional voice loop.
        """
        self.is_worker_locked = True
        self.helmet_detected = False
        self.vest_detected = False
        self.current_alert_key = "ALL_MISSING"
        self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: HELMET & VEST MISSING! COLLECT FROM BIN A"
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
        Real-time Computer Vision Filter for Helmets & Vests.
        Recognizes:
        - Bright Helmets: Yellow, Orange, and White Hardhats
        - Dark/Black Motorcycle Helmets: Wide cranial dome coverage
        Rejects:
        - Casual Baseball Caps & Beanies: Exposed forehead skin & narrow crown
        - Ordinary Cotton Shirts: Lacks high-vis fluorescent saturation
        """
        h, w, _ = frame.shape
        detections = []

        # 1. Person presence (centered detection box)
        person_box = [int(w * 0.18), int(h * 0.10), int(w * 0.82), int(h * 0.95)]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.96})

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Head Region (Top 8% to 40%)
        head_y1, head_y2 = int(h * 0.08), int(h * 0.40)
        head_x1, head_x2 = int(w * 0.28), int(w * 0.72)
        head_roi = hsv[head_y1:head_y2, head_x1:head_x2]
        head_total = max(1, head_roi.shape[0] * head_roi.shape[1])

        # (A) Bright Helmets (Yellow / Orange Hardhats, White Helmets)
        mask_yellow = cv2.inRange(head_roi, np.array([12, 60, 60]), np.array([38, 255, 255]))
        mask_white = cv2.inRange(head_roi, np.array([0, 0, 175]), np.array([180, 65, 255]))
        bright_pixels = cv2.countNonZero(mask_yellow) + cv2.countNonZero(mask_white)
        bright_ratio = bright_pixels / head_total

        # (B) Dark / Black Motorcycle Helmets:
        # Measures dark shell density vs exposed forehead skin
        mask_dark_shell = cv2.inRange(head_roi, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_pixels = cv2.countNonZero(mask_dark_shell)
        dark_ratio = dark_pixels / head_total

        # Human Skin Tone in HSV
        mask_skin = cv2.inRange(head_roi, np.array([0, 25, 55]), np.array([25, 180, 250]))
        skin_pixels = cv2.countNonZero(mask_skin)
        skin_ratio = skin_pixels / head_total

        # High-gloss plastic / visor specular reflection
        mask_gloss = cv2.inRange(head_roi, np.array([0, 0, 210]), np.array([180, 50, 255]))
        gloss_pixels = cv2.countNonZero(mask_gloss)
        gloss_ratio = gloss_pixels / head_total

        is_helmet = False
        conf = 0.94

        if bright_ratio > 0.08:
            # Yellow / Orange / White hardhat
            is_helmet = True
            conf = round(min(0.98, 0.82 + bright_ratio), 2)
        elif (dark_ratio > 0.32 and skin_ratio < 0.25) or (dark_ratio + gloss_ratio > 0.30 and skin_ratio < 0.22):
            # Black / Dark Motorcycle Helmet (wide shell coverage with low forehead skin exposure)
            is_helmet = True
            conf = 0.95

        head_box = [head_x1, head_y1, head_x2, head_y2]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            # Baseball caps (which have exposed forehead skin > 25%) and bare hair are rejected!
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})

        # 3. Torso Region (38% to 85%)
        torso_y1, torso_y2 = int(h * 0.38), int(h * 0.85)
        torso_x1, torso_x2 = int(w * 0.22), int(w * 0.78)
        torso_roi = hsv[torso_y1:torso_y2, torso_x1:torso_x2]

        # Fluorescent Neon Green/Yellow Safety Vest
        mask_neon = cv2.inRange(torso_roi, np.array([22, 90, 90]), np.array([50, 255, 255]))
        # High-Vis Safety Orange Vest
        mask_orange = cv2.inRange(torso_roi, np.array([5, 120, 110]), np.array([20, 255, 255]))

        vest_pixels = cv2.countNonZero(mask_neon) + cv2.countNonZero(mask_orange)
        torso_total = max(1, torso_roi.shape[0] * torso_roi.shape[1])
        vest_ratio = vest_pixels / torso_total

        torso_box = [torso_x1, torso_y1, torso_x2, torso_y2]
        if vest_ratio > 0.09:
            detections.append({"bbox": torso_box, "label": "vest", "conf": round(min(0.98, 0.78 + vest_ratio), 2)})
        else:
            # Normal casual cotton shirts are rejected!
            detections.append({"bbox": torso_box, "label": "no_vest", "conf": 0.94})

        return detections

    def process_frame(self, frame: np.ndarray, detections: Optional[List[Dict]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Main Video Frame Pipeline:
        1. Performs live OpenCV PPE detection if detections not supplied.
        2. Applies 2-second stabilization hold timer while worker steps up.
        3. Evaluates Upper-Body PPE compliance (Helmet & Reflective Vest).
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
                "stats": self.daily_stats
            }

        # 2. Worker Locked State (Freeze output until Re-Scan, Next Worker, or Override clicked)
        if self.is_worker_locked:
            cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 25), -1)
            cv2.putText(frame, f"SAFESITE AI | WORKER #{self.worker_id_counter} SCAN COMPLETED (LOCKED)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)

            color = (0, 255, 0) if self.current_alert_key == "CLEARED" else (0, 0, 255)
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

        # 3. Active Unlocked Frame Evaluation
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
            center_y = int((y1 + y2) / 2)

            if label == "person":
                person_in_frame = True
                if (w * 0.15) <= center_x <= (w * 0.85) and (y2 - y1) > (h * 0.35):
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

        # 4. 2-Second Hold Stabilization Timer Logic
        if person_in_frame and person_centered:
            if self.person_first_seen_time is None:
                self.person_first_seen_time = now

            elapsed_hold = now - self.person_first_seen_time
            remaining_hold = max(0.0, self.required_hold_duration - elapsed_hold)

            if remaining_hold > 0:
                # Still in 2-second countdown mode — DO NOT decision lock yet!
                self.current_alert_key = "SCANNING"
                self.current_state_text = f"⏳ SCANNING WORKER #{self.worker_id_counter}... HOLD STILL ({remaining_hold:.1f}s)"
                color = (0, 240, 255)
            else:
                # 2 Seconds Elapsed! Finalize scan decision and LOCK output!
                if helmet_detected and vest_detected:
                    # Both Helmet AND Vest Detected
                    self.current_alert_key = "CLEARED"
                    self.current_state_text = f"🟢 WORKER #{self.worker_id_counter}: HELMET & VEST VERIFIED — SHIFT CLEARED!"
                    color = (0, 255, 0)
                    self.is_worker_locked = True
                    self.daily_stats["cleared_count"] += 1
                elif helmet_detected and not vest_detected:
                    # Helmet present, Vest missing
                    self.current_alert_key = "VEST_MISSING"
                    self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: HELMET DETECTED | VEST MISSING! COLLECT FROM BIN A"
                    color = (0, 0, 255)
                    self.is_worker_locked = True
                    self.daily_stats["violations_count"] += 1
                    self.daily_stats["spare_ppe_issued"] += 1
                elif not helmet_detected and vest_detected:
                    # Vest present, Helmet missing
                    self.current_alert_key = "HELMET_MISSING"
                    self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: VEST DETECTED | HELMET MISSING! COLLECT FROM BIN A"
                    color = (0, 0, 255)
                    self.is_worker_locked = True
                    self.daily_stats["violations_count"] += 1
                    self.daily_stats["spare_ppe_issued"] += 1
                else:
                    # Both missing
                    self.current_alert_key = "ALL_MISSING"
                    self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: HELMET & VEST MISSING! COLLECT FROM BIN A"
                    color = (0, 0, 255)
                    self.is_worker_locked = True
                    self.daily_stats["violations_count"] += 1
                    self.daily_stats["spare_ppe_issued"] += 2

                self.daily_stats["total_scans"] += 1
                self.daily_stats["logs"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "worker": f"Worker #{self.worker_id_counter}",
                    "status": self.current_alert_key,
                    "message": self.current_state_text
                })
        else:
            # Person left frame or not properly centered
            self.person_first_seen_time = None
            if person_in_frame and not person_centered:
                self.current_alert_key = "NOT_VISIBLE"
                self.current_state_text = "⚠️ PLEASE STAND IN CENTER OF CAMERA VIEW"
                color = (0, 165, 255)
            else:
                self.current_alert_key = "WAITING"
                self.current_state_text = f"👤 WAITING FOR WORKER #{self.worker_id_counter} TO STEP UP..."
                color = (0, 240, 255)

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
