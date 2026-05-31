import os
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

def test_model_access():
    print("=========================================================")
    print(" Vertex AI Model Access Inspector")
    print("=========================================================")
    
    location = "us-central1"
    print(f"Initializing Vertex AI in region: {location}...\n")
    aiplatform.init(location=location)
    
    models_to_test = [
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro-preview-0409",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-1.0-pro-002",
        "gemini-pro"
    ]
    
    print("Testing Access to Gemini Models...\n")
    
    success_count = 0
    for model_name in models_to_test:
        try:
            model = GenerativeModel(model_name)
            # Send a tiny request to verify execution rights
            response = model.generate_content("Respond with 'OK' if you can read this.")
            print(f"✅ {model_name} : SUCCESS (Access Granted)")
            success_count += 1
        except Exception as e:
            # Extract just the first line of the error to keep output clean
            error_msg = str(e).split('\n')[0]
            print(f"❌ {model_name} : FAILED -> {error_msg}")
            
    print("\n=========================================================")
    if success_count == 0:
        print("CRITICAL FAILURE: No models are accessible.")
        print("REASON: The Vertex AI API is likely NOT ENABLED in your GCP project ('novate-ai'),")
        print("or your VM's service account does not have the 'Vertex AI User' role.")
        print("ACTION REQUIRED: Go to GCP Console -> APIs & Services -> Enable 'Vertex AI API'.")
    else:
        print("Please update `core/agent.py` to use one of the SUCCESS models above!")

if __name__ == "__main__":
    test_model_access()
