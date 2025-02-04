import torch
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load YOLOv5n model
            self.model = YOLO('yolov5n.pt')
            logger.info("Successfully loaded YOLOv5n model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        # COCO class names
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Detect objects in a frame
        Returns: frame with detections drawn, list of detections
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Get confidence
                    conf = box.conf[0]
                    # Get class
                    cls = int(box.cls[0])
                    class_name = self.class_names[cls]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': class_name
                    })
                    
                    # Draw on frame
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Add label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, 
                              (int(x1), int(y1 - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
            
            return frame, detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return frame, []

    def __call__(self, frame):
        return self.detect(frame) 