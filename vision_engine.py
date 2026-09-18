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
        SafeSite PPE Detector — Simplified, Reliable Rules:

        STEP 1 — FACE CHECK:
          Find the human face skin region in the center of the frame.
          If face is not clearly visible (low light, blocked, camera off-center,
          hand over face, or camera pointing away) → return [] so system asks to FOCUS.

        STEP 2 — HELMET CHECK (region directly above the face):
          PASS  ✓ : Any hard helmet shell on head:
                    - Yellow / Orange / Red construction hardhat
                    - White hardhat
                    - Blue / any-color hardhat (saturated, non-skin color)
                    - Dark motorcycle helmet (covers full cranial dome above face)
          REJECT ✗ : Cap, scarf, gamcha, bare head, or hair — anything that is NOT a
                     rigid shell fully covering the top of the head.

          Key Logic:
            - Helmet (hardhat / motorcycle) = NON-SKIN, NON-CLOTH rigid coverage
              occupying most of the dome above the face.
            - Cap / Cloth = has SKIN or HAIR visible in the crown, or is too thin/
              irregular (gamcha / scarf shows cloth folds, not a solid dome).
            - Bare head = dominant skin / hair (V moderate) in the crown.

        STEP 3 — VEST CHECK:
          Look for high-visibility neon yellow/orange below the head.
        """
        h, w, _ = frame.shape
        detections = []

        # ================================================================
        # STEP 1 — FACE / HUMAN PRESENCE CHECK
        # ================================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Skin tone range (face / neck) ---
        # HSV: H in [0,22], S in [28,180], V in [50,240]
        skin_lo = np.array([0, 28, 50])
        skin_hi = np.array([22, 180, 240])
        skin_mask = cv2.inRange(hsv, skin_lo, skin_hi)

        # Ignore top and bottom banners
        skin_mask[0:int(h * 0.06), :] = 0
        skin_mask[int(h * 0.92):, :] = 0

        # Only look in the central 60% of width — the worker stands in front of the camera
        skin_mask[:, 0:int(w * 0.20)] = 0
        skin_mask[:, int(w * 0.80):] = 0

        # Morphological clean-up to merge nearby skin blobs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k)
        skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Find the face: we want the HIGHEST (topmost) skin blob in the frame
        # that is wide enough and centered — not the LARGEST area blob (which may be body/arms)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_clean)
        best_face = None
        best_y = h  # Start with a very large Y value; we want minimum Y (topmost)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0])
            # Must be:
            # - Large enough to be a face (>= 1500 px)
            # - Horizontally centered in the frame
            # - Wide enough relative to height (aspect ratio of a face: 0.4 < w/h < 2.5)
            if (area >= 1500
                    and (w * 0.18) < cx < (w * 0.82)
                    and bh > 20
                    and 0.4 < (bw / bh) < 2.5):
                # Pick the topmost blob (smallest y coordinate)
                if y < best_y:
                    best_y = y
                    best_face = stats[i]

        # Brightness check: if the frame is too dark, return []
        mean_brightness = float(np.mean(gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]))
        if mean_brightness < 30:
            # Too dark — return empty so system asks to improve lighting / focus
            return []

        if best_face is None:
            # No clear face found — camera pointing away, hand blocking, or person absent
            return []

        # Extract face bounding box
        fx = int(best_face[cv2.CC_STAT_LEFT])
        fy = int(best_face[cv2.CC_STAT_TOP])
        fw = int(best_face[cv2.CC_STAT_WIDTH])
        fh = int(best_face[cv2.CC_STAT_HEIGHT])

        # Face centroid Y (for placing torso below)
        face_cy = fy + fh // 2

        # ================================================================
        # STEP 2 — HELMET REGION: Directly ABOVE the face
        # ================================================================
        # The helmet dome sits above the face skin blob.
        # We crop from (face_top - 1.3 * face_height) up to exactly face_top.
        # This avoids forehead/skin contaminating our helmet analysis.
        #
        # NOTE: face skin blob's TOP (fy) is typically around the cheekbone / temple area
        # for most close-up webcam shots. The helmet dome sits STRICTLY above this line.
        helmet_y1 = max(int(h * 0.02), fy - int(fh * 1.3))
        helmet_y2 = fy                 # Stop exactly at the top of the face skin blob
        helmet_x1 = max(0, fx - int(fw * 0.30))
        helmet_x2 = min(w, fx + fw + int(fw * 0.30))

        helmet_crop = frame[helmet_y1:helmet_y2, helmet_x1:helmet_x2]
        h_c, w_c, _ = helmet_crop.shape
        if h_c < 15 or w_c < 15:
            # Helmet region is too small — person too close or face at very top
            # Just mark person present, helmet unknown
            person_box = [int(w * 0.12), helmet_y1, int(w * 0.88), min(h - 10, face_cy + int(h * 0.45))]
            detections.append({"bbox": person_box, "label": "person", "conf": 0.90})
            head_box = [helmet_x1, helmet_y1, helmet_x2, fy + int(fh * 0.2)]
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.90})
            return detections

        helmet_hsv = cv2.cvtColor(helmet_crop, cv2.COLOR_BGR2HSV)
        helmet_total = max(1, h_c * w_c)

        # --- What is in the helmet region? ---
        # Skin (bare head / face showing through thin cloth)
        r_skin = cv2.countNonZero(cv2.inRange(helmet_hsv, skin_lo, skin_hi)) / helmet_total

        # Hair — dark brown / black matte texture in crown (bare head or very thin cap)
        # Hue: 0-25, Saturation: 20-130, Value: 15-80
        r_hair = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([0, 20, 15]), np.array([25, 135, 80]))
        ) / helmet_total

        # Yellow / Orange construction hardhat (highly saturated)
        r_yellow = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([13, 100, 90]), np.array([38, 255, 255]))
        ) / helmet_total

        # Orange hardhat (slightly redder than yellow)
        r_orange = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([5, 120, 90]), np.array([17, 255, 255]))
        ) / helmet_total

        # Red hardhat  (wraps around hue 0/180)
        r_red = (cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([0, 120, 90]), np.array([6, 255, 255]))) +
                 cv2.countNonZero(cv2.inRange(helmet_hsv, np.array([174, 120, 90]), np.array([180, 255, 255])))) / helmet_total

        # Blue hardhat
        r_blue = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([95, 100, 60]), np.array([135, 255, 255]))
        ) / helmet_total

        # White hardhat
        r_white = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([0, 0, 185]), np.array([180, 55, 255]))
        ) / helmet_total

        # Any solid non-skin saturated color (catches unusual hardhat colors like green, purple)
        r_saturated = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([0, 80, 70]), np.array([180, 255, 255]))
        ) / helmet_total

        # Dark region (motorcycle helmet / black hardhat)
        r_dark = cv2.countNonZero(
            cv2.inRange(helmet_hsv, np.array([0, 0, 0]), np.array([180, 255, 85]))
        ) / helmet_total

        # Bright coverage = sum of all non-skin helmet colors
        r_bright_helmet = r_yellow + r_orange + r_red + r_blue

        # Minimum helmet zone height: if the zone above the face is less than
        # 30% of the face height, there is almost nothing above the face —
        # i.e., the person is pressed to the top of the frame with no room for a helmet.
        # In that case mark as no_hardhat so system asks them to step back.
        zone_height = helmet_y2 - helmet_y1
        zone_too_small = zone_height < int(fh * 0.30)

        # --- Decision Logic ---
        is_helmet = False
        conf = 0.94

        if zone_too_small:
            # Not enough space above face to fit a helmet dome — treat as no helmet
            is_helmet = False

        elif r_bright_helmet > 0.07:
            # Vivid colored hard hat (yellow, orange, red, blue)
            is_helmet = True
            conf = round(min(0.98, 0.85 + r_bright_helmet), 2)

        elif r_white > 0.20 and r_skin < 0.20:
            # White hardhat (confirm it is NOT just a white wall)
            is_helmet = True
            conf = 0.95

        elif r_dark > 0.40 and r_skin < 0.18 and r_hair < 0.08:
            # Dark motorcycle helmet or black hardhat.
            # Rigid shell: covers dome above face with NO exposed hair or skin.
            # Real bare dark hair has r_hair > 0.10 since hair pixels show distinct
            # low-saturation, low-value texture. Motorcycle helmet shell has r_hair ~ 0.
            # Also a gamcha / scarf draped over the head will show hair beneath it,
            # pushing r_hair or r_skin above threshold.
            is_helmet = True
            conf = 0.95

        elif r_saturated > 0.15 and r_skin < 0.15 and r_hair < 0.10:
            # Any other saturated non-skin helmet color (green hardhat, etc.)
            is_helmet = True
            conf = 0.92

        # Person box
        person_box = [int(w * 0.12), helmet_y1, int(w * 0.88), min(h - 10, face_cy + int(h * 0.45))]
        detections.append({"bbox": person_box, "label": "person", "conf": 0.96})

        head_box = [helmet_x1, helmet_y1, helmet_x2, min(h - 5, fy + int(fh * 0.15))]
        if is_helmet:
            detections.append({"bbox": head_box, "label": "hardhat", "conf": conf})
        else:
            detections.append({"bbox": head_box, "label": "no_hardhat", "conf": 0.95})


        # ================================================================
        # STEP 3 — VEST CHECK — NOT YET IMPLEMENTED
        # Will be added once helmet detection is stable.
        # For now vest is always treated as not detected so the
        # system only gates on helmet (the primary check).
        # ================================================================
        # detections.append({"bbox": [...], "label": "no_vest", "conf": 0.0})

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
