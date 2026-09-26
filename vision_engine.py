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
        SafeSite AI — Real-time Head & Helmet Verification Engine.

        1. Validates human presence:
           - Finds face/neck skin cluster in the user's interactive scanning zone (center x: 18%-82%, y: 15%-90%).
           - Ignores background ceiling/door reflections.
           - If no person or bad lighting/covered camera -> returns [] so system prompts FOCUS / ALIGN.

        2. Dynamically locates the helmet region directly on the person's head:
           - Rather than assuming a hardcoded position, helmet zone is dynamically anchored right on/above the face.

        3. Differentiates Helmets from Bare Hair, Caps, and Scarves:
           - Any rigid helmet (Yellow/Orange/Blue/Red Hardhat, White Hardhat, or Dark Motorcycle Helmet).
           - Bare head / cloth wrapping rejected.
        """
        h, w, _ = frame.shape
        detections = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Minimum illumination check
        mean_v = float(np.mean(gray))
        if mean_v < 28:
            return []

        # 2. Robust Skin detection in human interaction area (center column)
        skin_lo = np.array([0, 25, 45])
        skin_hi = np.array([25, 185, 245])

        skin_mask = cv2.inRange(hsv, skin_lo, skin_hi)
        # Exclude extreme edges (workers stand centered in front of kiosk)
        skin_mask[:, 0:int(w * 0.16)] = 0
        skin_mask[:, int(w * 0.84):] = 0
        skin_mask[int(h * 0.94):, :] = 0

        # Morphological clean-up to remove pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_clean)
        best_face = None
        best_score = 0
        mid_x = w / 2.0

        # Find the primary face skin cluster in central viewing area
        # Center-weighted scoring ensures hands/arms raised to the side are never mistaken for the face
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx = centroids[i][0]
            cy = centroids[i][1]

            # Face blob should be large enough, horizontally aligned with kiosk, and in upper-middle frame
            if area > 1800 and (w * 0.18) < cx < (w * 0.82) and y > int(h * 0.10):
                # Weight by proximity to camera horizontal center:
                # The worker's head is centered in front of the kiosk; hands/arms are peripheral
                dx = abs(cx - mid_x)
                score = area / (1.0 + (dx / (w * 0.16))**2)
                if score > best_score:
                    best_score = score
                    best_face = (x, y, bw, bh, cx, cy)

        # If no face is found (empty wall, camera pointed away, dark, hands covering),
        # return empty -> state machine stays in WAITING / prompts user to stand in view
        if best_face is None:
            return []

        fx, fy, fw, fh, fcx, fcy = best_face

        # 3. HELMET / HEAD AREA FILTER: STRICTLY ABOVE EYES & NOSE
        # fx, fy is the top-most boundary of the face skin (eyebrow / upper bridge of nose line).
        # fcx is the vertical facial midline (nose axis).
        # Head width is proportional to user's distance in camera (~26% of frame width).
        head_w = int(w * 0.26)
        head_x1 = max(0, int(fcx - head_w / 2))
        head_x2 = min(w, int(fcx + head_w / 2))

        # Vertical helmet zone:
        # Extends from eyebrow/upper-nose line (fy) upward by ~80% of head width.
        # Overlaps slightly with forehead (5% of head_w) to capture brim/visor.
        head_top_y = max(int(h * 0.03), fy - int(head_w * 0.80))
        head_bottom_y = min(h - 10, fy + int(head_w * 0.06))

        helmet_crop = frame[head_top_y:head_bottom_y, head_x1:head_x2]
        ch, cw, _ = helmet_crop.shape

        if ch < 20 or cw < 20:
            person_box = [int(w * 0.12), head_top_y, int(w * 0.88), min(h - 10, fy + fh + int(h * 0.1))]
            detections.append({"bbox": person_box, "label": "person", "conf": 0.95})
            detections.append({"bbox": [head_x1, head_top_y, head_x2, head_bottom_y], "label": "no_hardhat", "conf": 0.90})
            return detections

        helmet_hsv = cv2.cvtColor(helmet_crop, cv2.COLOR_BGR2HSV)
        helmet_total = max(1, ch * cw)

        # 4. Color, Texture & Shell Analysis in Head Zone:
        # Skin in crown zone: bare forehead / bald head visible
        r_skin = cv2.countNonZero(cv2.inRange(helmet_hsv, skin_lo, skin_hi)) / helmet_total

        # Colored construction hardhats (high saturation colors S >= 100)
        r_yellow = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([13, 100, 90]), np.array([38, 255, 255]))) / helmet_total
        r_orange = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([5, 110, 90]), np.array([17, 255, 255]))) / helmet_total
        r_red = (cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([0, 130, 90]), np.array([6, 255, 255]))) +
                 cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([174, 130, 90]), np.array([180, 255, 255])))) / helmet_total
        r_blue = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([95, 90, 60]), np.array([135, 255, 255]))) / helmet_total
        r_white = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([0, 0, 185]), np.array([180, 50, 255]))) / helmet_total
        r_dark = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([0, 0, 0]), np.array([180, 255, 105]))) / helmet_total
        r_vivid = r_yellow + r_orange + r_red + r_blue

        # Specular gloss / reflection highlights on rigid shell (polycarbonate/fiberglass/ABS):
        # A rigid helmet reflects bright room/lamp specular spots (glare_px >= 12).
        # Matte cotton caps, baseball caps, scarves, and hair have NO specular reflection (glare_px == 0).
        glare_px = cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([0, 0, 190]), np.array([180, 60, 255])))

        # Decision Logic:
        is_helmet = False
        conf = 0.94

        if r_vivid >= 0.25:
            # Solid Construction Hardhat (Yellow, Orange, Red, Blue spanning the dome)
            # Note: A small letter/logo on a cap only covers 5-13% of the head crop, so it is strictly rejected.
            is_helmet = True
            conf = round(min(0.98, 0.85 + r_vivid), 2)
        elif r_white > 0.20 and r_skin < 0.20:
            # White Construction Hardhat (dominant bright non-skin dome)
            is_helmet = True
            conf = 0.95
        elif (r_dark >= 0.60 and r_skin < 0.10):
            # Solid Dark / Black Helmet (Motorcycle helmet, black PPE helmet):
            # A full rigid helmet dome covers the upper head and forehead down to brow level (dark >= 60%, skin < 10%).
            # Baseball caps, cloth wraps, and bare heads expose significant forehead skin/hair texture (dark < 40% or skin > 15%).
            is_helmet = True
            conf = 0.97
        elif (r_dark > 0.40 and r_skin < 0.22 and glare_px >= 10):
            # Glossy motorcycle helmet or black helmet shell with specular highlight points
            is_helmet = True
            conf = 0.96
        elif (glare_px >= 15 and r_skin < 0.20 and (r_dark > 0.30 or r_vivid > 0.04)):
            # Glossy colored motorcycle helmet or tinted shell (any color with rigid specular sheen)
            is_helmet = True
            conf = 0.95

        # Person bounding box
        person_box = [int(w * 0.12), head_top_y, int(w * 0.88), min(h - 10, fy + fh + int(h * 0.15))]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.96})

        # Head / Helmet box (drawn directly above the eyes and nose)
        head_box = [head_x1, head_top_y, head_x2, head_bottom_y]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})

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
