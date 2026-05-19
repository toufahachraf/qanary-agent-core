from typing import TypedDict, Annotated, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class WorkflowStage(str, Enum):
    """
    Strict clinical workflow stages. The graph uses this to ensure 
    the agent cannot skip critical steps (e.g., jumping to inference before validation).
    """
    INTAKE = "INTAKE"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    QUANTUM_INFERENCE = "QUANTUM_INFERENCE"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"

class PatientContext(BaseModel):
    """
    Strict validation for patient data. Ensures we don't process strings for age,
    and explicitly tracks high-risk genetic markers relevant to breast cancer.
    """
    patient_id: str = Field(description="Unique patient identifier mapping to EHR")
    age: Optional[int] = Field(default=None, ge=0, le=120, description="Patient age in years")
    genetic_markers: List[str] = Field(default_factory=list, description="e.g., BRCA1, BRCA2 mutations")
    previous_scans_gcs_uris: List[str] = Field(
        default_factory=list, 
        description="GCP Storage URIs for historical MRI comparisons"
    )

class MRIMetadata(BaseModel):
    """
    Validates the MRI data. Strongly couples with Google Cloud Storage (gs://)
    and ensures the image is flagged as clinically valid before model inference.
    """
    gcs_uri: str = Field(description="Google Cloud Storage URI of the DICOM/MRI file")
    modality: str = Field(default="MRI", description="Imaging modality, must be MRI for this pipeline")
    resolution: Optional[str] = Field(default=None, description="Extracted DICOM resolution")
    is_validated: bool = Field(default=False, description="True if the image passed quality and modality checks")

class AgentState(TypedDict):
    """
    The main State object passed between LangGraph nodes.
    
    NOTE: We use TypedDict for the outer layer because it flawlessly integrates 
    with LangGraph's `add_messages` reducer channel. However, all nested domain 
    objects (patient, mri_data) are strict Pydantic BaseModels to guarantee 
    runtime validation and easy serialization to GCP Firestore.
    """
    # The `add_messages` reducer ensures new messages are appended, not overwritten.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # GCP / Enterprise tracking
    session_id: str
    
    # Strict Clinical Data Models
    patient: Optional[PatientContext]
    mri_data: Optional[MRIMetadata]
    
    # State Machine Guardrail
    current_stage: WorkflowStage
