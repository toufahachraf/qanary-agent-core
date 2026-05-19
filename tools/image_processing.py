import io
import json
import cv2
import torch
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import torchvision.transforms.functional as F

# GCP Imports
from google.cloud import storage

# Local ML Imports
from ml_models.utils.gcp_dicom import load_dcm_from_gcs, get_pixel_array
from ml_models.model_loader import load_cropper_model

class CropLesionInput(BaseModel):
    gcs_uri: str = Field(description="The Google Cloud Storage URI of the raw DICOM image (e.g., gs://bucket/file.dcm)")
    confidence_threshold: float = Field(default=0.60, description="Minimum confidence score to accept a bounding box")

@tool("lesion_cropper_tool", args_schema=CropLesionInput)
def lesion_cropper_tool(gcs_uri: str, confidence_threshold: float = 0.60) -> str:
    """
    Use this tool to find and crop a breast lesion from a full high-resolution DICOM Mammogram/MRI.
    It runs a Quantum-Ready Faster R-CNN detection model to find the Region of Interest.
    Returns a JSON string containing the GCS URI of the newly cropped lesion image.
    """
    try:
        # 1. Parse the GCS URI (gs://bucket_name/blob_name)
        if not gcs_uri.startswith("gs://"):
            return json.dumps({"status": "Error", "message": "URI must start with gs://"})
            
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        if len(parts) != 2:
            return json.dumps({"status": "Error", "message": "Invalid GCS URI format."})
            
        bucket_name, blob_name = parts[0], parts[1]
        
        # Connect to GCP
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # 2. Load DICOM securely from GCS to memory
        dcm = load_dcm_from_gcs(bucket, blob_name)
        img_full_raw = get_pixel_array(dcm)
        raw_h, raw_w = img_full_raw.shape
        
        # 3. Resize for Inference (max_dim=1024) to save VRAM
        max_dim = 1024
        scale = 1.0
        img_inf = img_full_raw.copy()
        if max(raw_h, raw_w) > max_dim:
            scale = max_dim / max(raw_h, raw_w)
            new_w, new_h = int(raw_w * scale), int(raw_h * scale)
            img_inf = cv2.resize(img_inf, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Normalize and Tensorize
        img_norm = (img_inf - img_inf.min()) / (img_inf.max() - img_inf.min() + 1e-8)
        img_rgb = np.stack([img_norm]*3, axis=-1).astype(np.float32)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)
        
        # 4. Load Singleton Model and Infer
        model = load_cropper_model(device=device)
        
        with torch.no_grad():
            outputs = model(image_tensor)[0]
            
        if len(outputs['boxes']) == 0:
            return json.dumps({"status": "Failed", "message": "No lesion detected by the model."})
            
        confidence = outputs['scores'][0].cpu().item()
        if confidence < confidence_threshold:
            return json.dumps({
                "status": "Failed", 
                "message": f"Lesion detected but below confidence threshold ({confidence:.2f} < {confidence_threshold})"
            })
            
        # 5. Scale bounding box back to high-res coordinates
        top_box = outputs['boxes'][0].cpu().numpy()
        inv_scale = 1.0 / scale
        x_min = max(0, int(top_box[0] * inv_scale))
        y_min = max(0, int(top_box[1] * inv_scale))
        x_max = min(raw_w, int(top_box[2] * inv_scale))
        y_max = min(raw_h, int(top_box[3] * inv_scale))
        
        # Perform the actual crop
        high_res_crop = img_full_raw[y_min:y_max, x_min:x_max]
        
        # 6. Stateless GCS Upload (Save as PNG)
        # Normalize the crop to 0-255 for standard PNG saving
        crop_norm = (high_res_crop - high_res_crop.min()) / (high_res_crop.max() - high_res_crop.min() + 1e-8)
        crop_uint8 = (crop_norm * 255).astype(np.uint8)
        
        # Encode to PNG in memory
        _, buffer = cv2.imencode(".png", crop_uint8)
        byte_stream = io.BytesIO(buffer)
        
        # Generate unique filename with timestamp to prevent overriding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_blob_name = f"agent_artifacts/crops/crop_{timestamp}.png"
        crop_blob = bucket.blob(crop_blob_name)
        
        # Upload
        crop_blob.upload_from_file(byte_stream, content_type="image/png")
        new_gcs_uri = f"gs://{bucket_name}/{crop_blob_name}"
        
        return json.dumps({
            "status": "Success",
            "confidence": round(confidence, 4),
            "bounding_box": [x_min, y_min, x_max, y_max],
            "cropped_image_uri": new_gcs_uri
        })
        
    except Exception as e:
        return json.dumps({"status": "Error", "message": str(e)})
