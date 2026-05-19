import torch
import os
from ml_models.architectures.lesion_cropper import get_detection_model

# Singleton instance to prevent reloading weights on every tool call
_CROPPER_MODEL = None

def load_cropper_model(weights_path: str = "ml_models/weights/breast_lesion_detector_v1.pth", device: str = None) -> torch.nn.Module:
    """
    Loads the Faster R-CNN cropper model into memory exactly once (Singleton pattern).
    This ensures the AI Agent is lightning fast during inference.
    """
    global _CROPPER_MODEL
    
    if _CROPPER_MODEL is not None:
        return _CROPPER_MODEL
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading Lesion Cropper Model into memory on {device}...")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Please place the .pth file there.")
        
    model = get_detection_model(num_classes=2)
    
    # Load weights mapping to the correct device (CPU or GPU)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Set to evaluation mode! Crucial for inference.
    
    _CROPPER_MODEL = model
    return _CROPPER_MODEL
