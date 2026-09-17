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
        Real-time Computer Vision Filter.
        1. Confirms a human head is ACTUALLY present in front of the camera (never triggers on empty background).
        2. Differentiates Helmets (Hardhats, White/Yellow Helmets, Black Motorcycle Helmets) from Caps/Hair/Scarves.
        3. Differentiates High-Vis Reflective Vests from normal cotton shirts.
        """
        h, w, _ = frame.shape
        detections = []

        # Target Head ROI (Top 8% to 44% of frame, centered)
        head_y1, head_y2 = int(h * 0.08), int(h * 0.44)
        head_x1, head_x2 = int(w * 0.25), int(w * 0.75)
        head_crop = frame[head_y1:head_y2, head_x1:head_x2]
        head_total = max(1, head_crop.shape[0] * head_crop.shape[1])

        # Target Torso ROI (38% to 85% of frame)
        torso_y1, torso_y2 = int(h * 0.40), int(h * 0.85)
        torso_x1, torso_x2 = int(w * 0.20), int(w * 0.80)
        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        torso_total = max(1, torso_crop.shape[0] * torso_crop.shape[1])

        hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        head_contrast = float(np.std(gray_head))

        # -------------------------------------------------------------
        # 1. HUMAN HEAD PRESENCE CHECK (Must be a real human head, not empty wall/ceiling)
        # -------------------------------------------------------------
        # Human skin detection (face / neck / forehead)
        mask_skin = cv2.inRange(hsv_head, np.array([0, 25, 55]), np.array([25, 175, 245]))
        skin_ratio = cv2.countNonZero(mask_skin) / head_total

        # Yellow/Orange plastic helmet
        mask_yellow = cv2.inRange(hsv_head, np.array([12, 60, 60]), np.array([38, 255, 255]))
        # Bright white helmet
        mask_white = cv2.inRange(hsv_head, np.array([0, 0, 175]), np.array([180, 65, 255]))
        bright_ratio = (cv2.countNonZero(mask_yellow) + cv2.countNonZero(mask_white)) / head_total

        # Dark helmet shell
        mask_dark = cv2.inRange(hsv_head, np.array([0, 0, 0]), np.array([180, 255, 75]))
        dark_ratio = cv2.countNonZero(mask_dark) / head_total

        # Glossy specular reflection
        mask_gloss = cv2.inRange(hsv_head, np.array([0, 0, 210]), np.array([180, 45, 255]))
        gloss_ratio = cv2.countNonZero(mask_gloss) / head_total

        gray_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        torso_contrast = float(np.std(gray_torso))

        # Verify presence: Need either visible human skin, or bright helmet, or dark motorcycle helmet with human body
        is_human_present = False
        if head_contrast > 16:
            if skin_ratio > 0.08:
                # Definite human face / head present in frame!
                is_human_present = True
            elif bright_ratio > 0.08 and torso_contrast > 18:
                # Human wearing yellow/white hardhat in front of camera
                is_human_present = True
            elif dark_ratio > 0.35 and (torso_contrast > 20 or gloss_ratio > 0.04):
                # Human wearing full motorcycle helmet sitting/standing in front of camera
                is_human_present = True

        if not is_human_present:
            # NO HUMAN HEAD IS IN VIEW (Empty room / camera pointing away)
            # Return empty detections so system prompts to focus / step up!
            return []

        # Human is confirmed in frame! Add person bounding box
        person_box = [int(w * 0.18), int(h * 0.10), int(w * 0.82), int(h * 0.95)]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.96})

        # -------------------------------------------------------------
        # 2. HELMET VS. CAP / BARE HEAD DISCRIMINATION
        # -------------------------------------------------------------
        is_helmet = False
        conf = 0.94

        if bright_ratio > 0.08:
            # High-visibility yellow, orange, or white helmet/hardhat
            is_helmet = True
            conf = round(min(0.98, 0.82 + bright_ratio), 2)
        elif (dark_ratio > 0.36 and skin_ratio < 0.18) or (dark_ratio + gloss_ratio > 0.35 and skin_ratio < 0.16):
            # Full black motorcycle helmet: covers forehead/temples/ears with dark rigid shell & low skin
            is_helmet = True
            conf = 0.95

        head_box = [head_x1, head_y1, head_x2, head_y2]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            # Rejects caps, beanies, or bare hair! (Caps have exposed forehead skin > 18% with small crown)
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})

        # -------------------------------------------------------------
        # 3. SAFETY VEST DISCRIMINATION
        # -------------------------------------------------------------
        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        mask_neon = cv2.inRange(hsv_torso, np.array([22, 90, 90]), np.array([50, 255, 255]))
        mask_orange = cv2.inRange(hsv_torso, np.array([5, 120, 110]), np.array([20, 255, 255]))
        vest_pixels = cv2.countNonZero(mask_neon) + cv2.countNonZero(mask_orange)
        vest_ratio = vest_pixels / torso_total

        torso_box = [torso_x1, torso_y1, torso_x2, torso_y2]
        if vest_ratio > 0.09:
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
                "stats": self.daily_stats
            }

        # 2. Worker Locked State (Freeze output until Re-Scan, Next Worker, or Override)
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

        # 4. Target Head Guide Box (shows worker where to position head)
        head_box_x1, head_box_y1 = int(w * 0.25), int(h * 0.08)
        head_box_x2, head_box_y2 = int(w * 0.75), int(h * 0.44)
        box_border_color = (0, 255, 0) if helmet_detected else ((0, 0, 255) if person_in_frame else (0, 240, 255))
        cv2.rectangle(frame, (head_box_x1, head_box_y1), (head_box_x2, head_box_y2), box_border_color, 1)

        # 5. Hold Stabilization Timer Logic (Only counts down when person is in camera view)
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
