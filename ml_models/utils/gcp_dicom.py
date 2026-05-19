import io
import numpy as np
import pydicom
from google.cloud import storage

def load_dcm_from_gcs(bucket: storage.Bucket, blob_name: str) -> pydicom.dataset.FileDataset:
    """
    Downloads a .dcm file from a Google Cloud Storage bucket into memory 
    and reads it securely using pydicom, avoiding local disk writes.
    """
    blob = bucket.blob(blob_name)
    dcm_bytes = blob.download_as_bytes()
    return pydicom.dcmread(io.BytesIO(dcm_bytes))

def get_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Safely extracts the pixel array from a DICOM dataset.
    Automatically handles Photometric Interpretation inversion (MONOCHROME1).
    """
    img = dcm.pixel_array
    if 'PhotometricInterpretation' in dcm and dcm.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img
    return img
