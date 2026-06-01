from pydantic import BaseModel, Field
from typing import Optional, List

class PatientContextInput(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    age: Optional[int] = Field(default=None, description="Patient age in years")
    genetic_markers: Optional[List[str]] = Field(default=None, description="e.g., ['BRCA1']")

class ChatRequest(BaseModel):
    session_id: str = Field(description="Unique ID for this diagnosis session")
    message: str = Field(default="Please analyze this scan.", description="The prompt or command for the agent")
    mri_gcs_uri: Optional[str] = Field(default=None, description="The gs:// URI of the uploaded DICOM")
    patient_context: Optional[PatientContextInput] = Field(default=None, description="Patient priors")

class Artifacts(BaseModel):
    cropped_image_uri: Optional[str] = Field(default=None, description="URI of the cropped lesion in GCS")
    bounding_box: Optional[List[int]] = Field(default=None, description="Coordinates [x_min, y_min, x_max, y_max]")
    confidence: Optional[float] = Field(default=None, description="ML Model confidence score")

class ChatResponse(BaseModel):
    status: str = Field(default="success")
    session_id: str
    current_stage: str
    ai_response: str
    artifacts: Optional[Artifacts] = None
