import cv2
import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def send_hardware_trigger(grade):
    """Mock the serial command that actuates a physical sorting flap."""
    logger.info(f"Hardware trigger: route {grade} produce to sorting flap.")


class VisionEngine:
    # --- DEFECT THRESHOLD PROFILES ---
    # LENIENT is tuned for real, imperfect, field-photographed onions.
    # STRICT is closer to the old behaviour (tuned for clean/synthetic images).
    # `defect_sensitivity` (0.0 = lenient, 1.0 = strict) interpolates between them,
    # so a manager can dial this in per-mandi instead of it being hardcoded.
    _LENIENT = {
        "sprout_total": 0.06,      # % of crop that must be green
        "sprout_blob": 0.03,       # largest connected green blob, as % of crop
        "rot_v_max": 40,           # how dark a pixel must be to count as "black rot"
        "rot_total": 0.05,
        "rot_blob": 0.035,
        "fungus_total": 0.16,
        "fungus_blob": 0.06,
        "crack_edge_density": 0.22,
    }
    _STRICT = {
        "sprout_total": 0.015,
        "sprout_blob": 0.008,
        "rot_v_max": 60,
        "rot_total": 0.02,
        "rot_blob": 0.01,
        "fungus_total": 0.02,
        "fungus_blob": 0.01,
        "crack_edge_density": 0.08,
    }

    def __init__(self, model_path="best.pt", pixels_per_mm=2.5, conf_thresh=0.55, defect_sensitivity=0.35):
        self.model = YOLO(model_path)
        self.pixels_per_mm = pixels_per_mm
        self.conf_thresh = conf_thresh

        # 0.0 = lenient (fewer false rejects on real onions), 1.0 = strict (old behaviour)
        self.defect_sensitivity = float(np.clip(defect_sensitivity, 0.0, 1.0))

        # Hardware Acceleration: Automatically uses CUDA GPU if available
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except (ImportError, RuntimeError):
            self.device = "cpu"

        print("=" * 50)
        print("FINAL HYBRID ENGINE (YOLO + Multi-Stage Surface Assaying)")
        print(f"Device: {self.device} | Defect Sensitivity: {self.defect_sensitivity}")
        print("=" * 50)

    def _threshold(self, key):
        """Linearly interpolate a threshold between the lenient and strict profiles."""
        lo = self._LENIENT[key]
        hi = self._STRICT[key]
        return lo + (hi - lo) * self.defect_sensitivity

    @staticmethod
    def _iou(box_a, box_b):
        """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / float(area_a + area_b - inter_area)

    @staticmethod
    def _largest_blob_ratio(mask, total_area):
        """
        Returns the area of the single largest connected component in `mask`,
        as a fraction of `total_area`. This is what lets us tell the difference
        between a real, solid defect patch and scattered noise from natural skin
        texture, dirt, or lighting variance spread thinly across the surface.
        """
        if total_area <= 0:
            return 0.0
        # Light morphological close so a defect isn't fragmented into many tiny
        # blobs by natural surface texture (which would otherwise all get skipped).
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(cv2.contourArea(c) for c in contours)
        return largest / total_area

    def analyze_surface_defects(self, crop):
        """
        Analyzes the surface using generalized texture, color variance, and necrotic/fungal spot metrics.
        Returns: ('Reject' | 'Sprouted' | 'Clean', reason_string)

        NOTE: These are HSV/texture heuristics, not a learned defect classifier, so they
        can never perfectly separate "real onion, slightly rustic" from "genuinely diseased."
        The thresholds below are deliberately relaxed and blob-aware (a defect must be one
        connected patch, not just scattered pixels) to avoid rejecting real, healthy onions.
        For best results, retrain/fine-tune on photos of real onions (not just clean/
        generated training images) rather than relying on this heuristic alone long-term.
        """
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return "Clean", "OK"

        hsv_full = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = h * w

        # 1. VEGETATIVE SPROUT DETECTION
        lower_green = np.array([30, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv_full, lower_green, upper_green)
        green_total_ratio = cv2.countNonZero(green_mask) / total_pixels
        green_blob_ratio = self._largest_blob_ratio(green_mask, total_pixels)
        if green_total_ratio > self._threshold("sprout_total") and green_blob_ratio > self._threshold("sprout_blob"):
            return "Sprouted", "Sprout Detected"

        # --- EXPANDED CROP FOR DEFECTS ---
        ch, cw = int(h * 0.8), int(w * 0.8)
        y1, x1 = int(h * 0.1), int(w * 0.1)
        center_crop = crop[y1:y1 + ch, x1:x1 + cw]

        if center_crop.size == 0:
            return "Clean", "OK"

        center_area = ch * cw
        hsv_center = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

        # 2. BLACK ROT DETECTION
        # Genuinely rotten patches are near-black AND form one solid mass.
        # Natural shadow, dirt specks, or a dark root scar are scattered/small - not a big solid blob.
        rot_v_max = int(round(self._threshold("rot_v_max")))
        dark_mask = cv2.inRange(
            hsv_center,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([180, 255, rot_v_max], dtype=np.uint8),
        )
        dark_total_ratio = cv2.countNonZero(dark_mask) / center_area
        dark_blob_ratio = self._largest_blob_ratio(dark_mask, center_area)
        if dark_total_ratio > self._threshold("rot_total") and dark_blob_ratio > self._threshold("rot_blob"):
            return "Reject", "Black Rot Detected"

        # 3. POWDERY MILDEW & GREY FUNGUS
        # Narrowed vs. the old version: dry papery onion skin is itself low-saturation and
        # mid-brightness, so a loose mask here flags healthy onions constantly. We now also
        # require the flagged area to form one solid patch, not diffuse texture across the skin.
        lower_fungus = np.array([0, 0, 150], dtype=np.uint8)   # true mold reads brighter/whiter
        upper_fungus = np.array([180, 45, 245], dtype=np.uint8)
        fungus_mask = cv2.inRange(hsv_center, lower_fungus, upper_fungus)

        lower_green_mold = np.array([35, 15, 60], dtype=np.uint8)
        upper_green_mold = np.array([85, 255, 190], dtype=np.uint8)
        green_mold_mask = cv2.inRange(hsv_center, lower_green_mold, upper_green_mold)

        combined_fungus = cv2.bitwise_or(fungus_mask, green_mold_mask)
        fungus_total_ratio = cv2.countNonZero(combined_fungus) / center_area
        fungus_blob_ratio = self._largest_blob_ratio(combined_fungus, center_area)
        if fungus_total_ratio > self._threshold("fungus_total") and fungus_blob_ratio > self._threshold("fungus_blob"):
            return "Reject", "Powdery Mildew"

        # 4. TEXTURE & DEEP CRACKS
        # Slightly larger blur to smooth over normal papery-skin fold lines and root fibers,
        # and a higher edge-density bar so only genuinely heavy cracking triggers a reject.
        gray_center = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_center, (7, 7), 0)
        edges = cv2.Canny(blurred, 70, 190)
        edge_density = cv2.countNonZero(edges) / center_area

        if edge_density > self._threshold("crack_edge_density"):
            return "Reject", "Severe Cracks"

        return "Clean", "Healthy"

    def process_frame(self, frame):
        h_frame, w_frame = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=0.75,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        summary = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Reject": 0}
        annotated_frame = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:

            # Sort all detected boxes by confidence score (Highest to Lowest). We process every
            # onion in frame, but keep only the highest-confidence box for each *physical* onion -
            # accepted_boxes below is used to drop duplicate/phantom boxes that overlap a box we
            # already accepted, rather than dropping every box except the single best one in the frame.
            sorted_boxes = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)
            accepted_boxes = []
            duplicate_iou_thresh = 0.35

            for box in sorted_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id].lower()

                w = x2 - x1
                h = y2 - y1
                diameter_px = max(w, h)
                diameter_mm = diameter_px / self.pixels_per_mm

                # Physical limit filters (Skip this box if it's too big/small)
                if diameter_mm < 10 or diameter_mm > 500:
                    continue
                if h == 0 or w == 0:
                    continue
                if min(w, h) / max(w, h) < 0.20:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                # --- PHANTOM SHADOW & REFLECTION REJECTION ---
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                avg_saturation = np.mean(hsv_crop[:, :, 1])
                if avg_saturation < 25:
                    continue  # Silently drop white/grey table reflections and check the next box

                # --- DUPLICATE / PHANTOM BOX REJECTION ---
                # If this box heavily overlaps one we've already accepted, it's almost certainly
                # a second detection of the SAME onion (not a second onion), so skip it.
                if any(self._iou((x1, y1, x2, y2), accepted) > duplicate_iou_thresh for accepted in accepted_boxes):
                    continue
                accepted_boxes.append((x1, y1, x2, y2))

                defect_status, defect_msg = self.analyze_surface_defects(crop)

                # --- GRADING HIERARCHY ---
                if any(k in cls_name for k in ["rot", "defect", "reject", "mold", "bad"]) or defect_status == "Reject":
                    grade = "Reject"
                    color = (0, 0, 255)  # Red (BGR)
                    display_label = f"Reject: {defect_msg}"
                elif defect_status == "Sprouted" or "sprout" in cls_name:
                    grade = "Grade C"
                    color = (0, 255, 255)  # Yellow (BGR)
                    display_label = "Grade C: Sprouted"
                else:
                    if diameter_mm > 55:
                        grade = "Grade A"
                        color = (0, 255, 0)  # Green (BGR)
                    elif 45 <= diameter_mm <= 55:
                        grade = "Grade B"
                        color = (0, 165, 255)  # Orange (BGR)
                    else:
                        grade = "Grade C"
                        color = (0, 255, 255)  # Yellow (BGR)
                    display_label = f"{grade} ({int(diameter_mm)}mm)"

                summary[grade] += 1
                send_hardware_trigger(grade)

                # Draw bounding box and high-contrast label tag
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                tag_text = f"{display_label} | {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 24)), (x1 + tw + 6, max(24, y1)), color, -1)
                cv2.putText(
                    annotated_frame,
                    tag_text,
                    (x1 + 3, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # No break here - continue on to grade every other onion detected in this frame.

        return annotated_frame, summary