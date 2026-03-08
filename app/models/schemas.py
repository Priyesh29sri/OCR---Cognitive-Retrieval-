from pydantic import BaseModel, Field
from typing import Literal, Tuple, Optional

class BoundingBox(BaseModel):
    # A box is strictly defined by 4 coordinates: (x_min, y_min, x_max, y_max)
    coordinates: Tuple[float, float, float, float] = Field(
        description="Bounding box coordinates in pixels"
    )

class DocumentElement(BaseModel):
    # The 'Literal' type strictly forces the CV model to ONLY output these 3 exact strings [9]
    element_type: Literal["text_paragraph", "table", "figure"]
    box: BoundingBox
    # We constrain the confidence score to mathematically valid percentages (0.0 to 1.0) [10]
    confidence_score: float = Field(ge=0.0, le=1.0, description="CV model confidence from 0 to 1")
    # Optional text extracted by our future OCR engine
    extracted_text: Optional[str] = None