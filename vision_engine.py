import cv2
import numpy as np
from ultralytics import YOLO

class VisionEngine:
    def __init__(self, model_path="best.pt", pixels_per_mm=2.5, conf_thresh=0.10):
        self.model = YOLO(model_path)
        self.pixels_per_mm = pixels_per_mm
        self.conf_thresh = conf_thresh

    def enhance_image(self, frame):
        """Applies CLAHE to handle poor lighting and reveal background details."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame):
        processed_frame = self.enhance_image(frame)
        
        # THE FIX: 
        # max_det=2000 overrides the default 300 limit so no onion is left behind.
        # imgsz=1920 gives the model enough pixels to see tiny background objects.
        # agnostic_nms=True stops overlapping background onions from hiding each other.
        results = self.model(
            processed_frame, 
            conf=self.conf_thresh, 
            iou=0.45, 
            imgsz=1920, 
            max_det=2000, 
            agnostic_nms=True,
            verbose=False
        )[0]
        
        summary = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Reject": 0}
        annotated_frame = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id].lower()

                valid_keywords = ["onion", "grade", "reject", "rot", "bad", "bawang", "bulb"]
                if not any(keyword in class_name for keyword in valid_keywords) and len(self.model.names) > 10:
                    continue 

                # Calculate diameter
                width_pixels = x2 - x1
                diameter_mm = width_pixels / self.pixels_per_mm

                # Map classes or apply fallback logic
                if "reject" in class_name or "rot" in class_name or "bad" in class_name:
                    grade, color = "Reject", (0, 0, 255) 
                elif "grade-a" in class_name or "large" in class_name:
                    grade, color = "Grade A", (0, 255, 0) 
                elif "grade-b" in class_name or "medium" in class_name:
                    grade, color = "Grade B", (255, 165, 0) 
                elif "grade-c" in class_name or "small" in class_name:
                    grade, color = "Grade C", (0, 255, 255) 
                else:
                    if diameter_mm > 55:
                        grade, color = "Grade A", (0, 255, 0)
                    elif 45 <= diameter_mm <= 55:
                        grade, color = "Grade B", (255, 165, 0)
                    elif 35 < diameter_mm < 45:
                        grade, color = "Grade C", (0, 255, 255)
                    else:
                        grade, color = "Reject", (0, 0, 255)

                summary[grade] += 1

                # Draw thinner boxes so massive batches don't become a giant block of color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                
                # Only draw text labels for foreground (larger) objects to prevent visual clutter
                if width_pixels > 25:
                    cv2.putText(annotated_frame, grade, (x1, max(15, y1 - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return annotated_frame, summary