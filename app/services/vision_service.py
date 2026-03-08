import io
import ssl
import certifi
import numpy as np
import easyocr
from PIL import Image, ImageDraw
from pdf2image import convert_from_bytes
from ultralytics import YOLO
from app.models.schemas import DocumentElement, BoundingBox

class VisionService:
    def __init__(self):
        # 1. Load the Object Detection Model
        self.model = YOLO('yolov8n.pt')
        # 2. Use lazy initialization for OCR to avoid startup delays
        self._ocr_reader = None
    
    @property
    def ocr_reader(self):
        """Lazy initialization of EasyOCR reader with SSL fix"""
        if self._ocr_reader is None:
            # Fix SSL certificate verification for macOS
            ssl._create_default_https_context = ssl._create_unverified_context
            self._ocr_reader = easyocr.Reader(['en'])
        return self._ocr_reader

    def process_document(self, file_bytes: bytes, content_type: str) -> list[Image.Image]:
        images = []
        if content_type == "application/pdf":
            images = convert_from_bytes(file_bytes)
        else:
            images = [Image.open(io.BytesIO(file_bytes))]
        return images

    def detect_layouts(self, image: Image.Image) -> list[DocumentElement]:
        detected_elements = []
        
        # First, run full-page OCR to capture ALL text on the page
        full_page_array = np.array(image)
        full_page_ocr = self.ocr_reader.readtext(full_page_array, detail=1)  # detail=1 gives coordinates
        
        # Add all detected text regions as text_paragraph elements
        for detection in full_page_ocr:
            bbox, text, confidence = detection
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] - convert to (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            
            element = DocumentElement(
                element_type="text_paragraph",
                box=BoundingBox(coordinates=(x1, y1, x2, y2)),
                confidence_score=float(confidence),
                extracted_text=text
            )
            detected_elements.append(element)
        
        # Then, run YOLO to find objects/figures
        results = self.model(image)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf)
                class_name = self.model.names[int(box.cls)]
                
                element_type = "figure" 
                if class_name == "table":
                    element_type = "table"

                # For figures and tables, try to extract any text within them
                cropped_img = image.crop((x1, y1, x2, y2))
                cropped_array = np.array(cropped_img)
                ocr_results = self.ocr_reader.readtext(cropped_array, detail=0)
                # Always include class name in extracted_text so images are queryable
                if ocr_results:
                    extracted_text = f"Detected {class_name}: " + " ".join(ocr_results)
                else:
                    extracted_text = f"Detected {class_name} in image."

                element = DocumentElement(
                    element_type=element_type,
                    box=BoundingBox(coordinates=(x1, y1, x2, y2)),
                    confidence_score=confidence,
                    extracted_text=extracted_text
                )
                detected_elements.append(element)
                
        return detected_elements