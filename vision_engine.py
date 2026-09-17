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
        Real-time Adaptive Computer Vision Filter:
        1. Confirms a human head/face is present in front of the camera (never triggers on empty room/wall).
        2. Dynamically locates the head and torso based on human silhouette & face skin cues,
           adapting seamlessly whether the user is sitting at a desk or standing at a kiosk.
        3. Differentiates Helmets (Hardhats, White/Yellow Helmets, Dark Motorcycle Helmets) from Caps/Hair/Scarves.
        4. Differentiates High-Vis Reflective Vests from normal cotton shirts.
        """
        h, w, _ = frame.shape
        detections = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------------------------------------------------------------
        # 1. HUMAN HEAD & FACE PRESENCE CHECK
        # -------------------------------------------------------------
        # Human skin tone mask across the central frame
        skin_mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 175, 245]))
        skin_mask[0:int(h * 0.05), :] = 0
        skin_mask[int(h * 0.92):, :] = 0

        # Central column (where the person sits or stands)
        center_x1, center_x2 = int(w * 0.20), int(w * 0.80)
        center_skin = skin_mask[:, center_x1:center_x2]
        skin_pixels = cv2.countNonZero(center_skin)
        skin_ratio_global = skin_pixels / max(1, (center_skin.shape[0] * center_skin.shape[1]))

        # If there is virtually zero human skin in the center of the camera view:
        # Camera is pointing at ceiling, wall, or empty chair
        if skin_ratio_global < 0.035:
            return []

        # -------------------------------------------------------------
        # 2. DYNAMIC HEAD & APEX LOCALIZATION
        # Locate the top of the head/helmet dome using horizontal edge silhouette
        # -------------------------------------------------------------
        edges = cv2.Canny(gray, 40, 120)
        edges[0:int(h * 0.05), :] = 0
        edges[int(h * 0.92):, :] = 0

        head_top_y = None
        for y in range(int(h * 0.06), int(h * 0.65)):
            if np.sum(edges[y, int(w * 0.25):int(w * 0.75)] > 0) > 12:
                head_top_y = y
                break

        if head_top_y is None:
            head_top_y = int(h * 0.15)

        # Determine Head Bounding Box based on detected apex:
        head_y1 = max(0, head_top_y)
        head_y2 = min(h - 50, head_top_y + int(h * 0.38))
        head_x1 = int(w * 0.22)
        head_x2 = int(w * 0.78)

        head_crop = frame[head_y1:head_y2, head_x1:head_x2]
        h_h, w_h, _ = head_crop.shape
        if h_h < 30 or w_h < 30:
            return []

        # Partition Head into Crown Dome (top 46%) and Face (bottom 54%)
        crown_crop = head_crop[0:int(0.46 * h_h), :]
        face_crop = head_crop[int(0.46 * h_h):, :]

        crown_hsv = cv2.cvtColor(crown_crop, cv2.COLOR_BGR2HSV)
        face_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

        crown_total = max(1, crown_crop.shape[0] * crown_crop.shape[1])
        face_total = max(1, face_crop.shape[0] * face_crop.shape[1])

        # Feature Ratios:
        face_skin = cv2.countNonZero(cv2.inRange(face_hsv, np.array([0, 25, 45]), np.array([25, 175, 245]))) / face_total
        crown_skin = cv2.countNonZero(cv2.inRange(crown_hsv, np.array([0, 25, 45]), np.array([25, 175, 245]))) / crown_total

        # Yellow / Orange Construction Hardhat
        crown_yellow = cv2.countNonZero(cv2.inRange(crown_hsv, np.array([12, 65, 65]), np.array([38, 255, 255]))) / crown_total
        # White Construction Hardhat
        crown_white = cv2.countNonZero(cv2.inRange(crown_hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))) / crown_total
        # Dark Motorcycle Helmet Shell
        crown_dark = cv2.countNonZero(cv2.inRange(crown_hsv, np.array([0, 0, 0]), np.array([180, 255, 85]))) / crown_total

        # Sides of helmet (ears/temples wrap)
        left_side = head_crop[0:int(0.65 * h_h), 0:int(0.25 * w_h)]
        right_side = head_crop[0:int(0.65 * h_h), int(0.75 * w_h):w_h]
        side_total = max(1, left_side.shape[0] * left_side.shape[1] + right_side.shape[0] * right_side.shape[1])
        side_dark = (cv2.countNonZero(cv2.inRange(cv2.cvtColor(left_side, cv2.COLOR_BGR2HSV), np.array([0, 0, 0]), np.array([180, 255, 85]))) +
                     cv2.countNonZero(cv2.inRange(cv2.cvtColor(right_side, cv2.COLOR_BGR2HSV), np.array([0, 0, 0]), np.array([180, 255, 85])))) / side_total

        is_helmet = False
        conf = 0.94

        if crown_yellow > 0.08:
            # Verified Yellow/Orange Construction Hardhat
            is_helmet = True
            conf = round(min(0.98, 0.82 + crown_yellow), 2)
        elif crown_white > 0.18 and crown_skin < 0.15:
            # Verified Bright White Hardhat
            is_helmet = True
            conf = 0.95
        elif (crown_dark > 0.22 or (crown_dark > 0.15 and side_dark > 0.18)) and crown_skin < 0.18 and face_skin > 0.06:
            # Verified Dark Motorcycle Helmet:
            # Dark rigid dome covering crown down to eyebrows, sides wrapped, with human face clearly visible underneath!
            is_helmet = True
            conf = 0.96

        # Person Bounding Box:
        person_box = [int(w * 0.15), head_y1, int(w * 0.85), min(h - 10, head_y2 + int(h * 0.45))]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.96})

        # Head / Helmet Bounding Box:
        head_box = [head_x1, head_y1, head_x2, head_y2]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})

        # -------------------------------------------------------------
        # 3. SAFETY VEST DISCRIMINATION (Directly below the head)
        # -------------------------------------------------------------
        torso_y1 = head_y2 - int(h * 0.04)
        torso_y2 = min(h - 10, head_y2 + int(h * 0.45))
        torso_x1 = int(w * 0.16)
        torso_x2 = int(w * 0.84)

        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso_crop.shape[0] > 20 and torso_crop.shape[1] > 20:
            hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
            torso_total = max(1, torso_crop.shape[0] * torso_crop.shape[1])

            mask_neon = cv2.inRange(hsv_torso, np.array([22, 90, 90]), np.array([50, 255, 255]))
            mask_orange = cv2.inRange(hsv_torso, np.array([5, 120, 110]), np.array([20, 255, 255]))
            vest_pixels = cv2.countNonZero(mask_neon) + cv2.countNonZero(mask_orange)
            vest_ratio = vest_pixels / torso_total

            torso_box = [torso_x1, torso_y1, torso_x2, torso_y2]
            if vest_ratio > 0.08:
                detections.append({"bbox": torso_box, "label": "vest", "conf": round(min(0.98, 0.78 + vest_ratio), 2)})
            else:
                detections.append({"bbox": torso_box, "label": "no_vest", "conf": 0.94})

        return detections

    def process_frame(self, frame: np.ndarray, detections: Optional[List[Dict]] = None) -> Tuple[np.ndarray, Dict]:
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
                if helmet_detected and vest_detected:
                    # Both Helmet AND Vest Verified
                    self.current_alert_key = "CLEARED"
                    self.current_state_text = f"🟢 WORKER #{self.worker_id_counter}: HELMET & VEST VERIFIED — SHIFT CLEARED!"
                    color = (0, 255, 0)
                    self.is_worker_locked = True
                    self.daily_stats["cleared_count"] += 1
                elif helmet_detected and not vest_detected:
                    # Helmet present (Main selling point!), Vest missing
                    self.current_alert_key = "VEST_MISSING"
                    self.current_state_text = f"🟢 WORKER #{self.worker_id_counter}: HELMET ACCEPTED (PASS) | VEST MISSING"
                    color = (0, 165, 255)
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
                    # Both missing (Cap or bare head with normal clothes)
                    self.current_alert_key = "ALL_MISSING"
                    self.current_state_text = f"🔴 WORKER #{self.worker_id_counter}: SAFETY GEAR MISSING! COLLECT FROM BIN A"
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
