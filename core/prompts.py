from typing import Dict, Any

# We import the TypedDict/BaseModel structures just for type hinting if needed, 
# but mostly we will pass the state dictionary directly.
from core.state import AgentState

def get_system_prompt(state: AgentState) -> str:
    """
    Generates the dynamic system prompt for Qanary.
    It injects the current clinical state (Patient, MRI, Workflow Stage) directly 
    into the prompt so Gemini has perfect contextual awareness at all times.
    """
    
    patient_info = state.get("patient")
    mri_info = state.get("mri_data")
    stage = state.get("current_stage", "INTAKE")
    
    # Format Patient Context
    patient_text = "No patient context provided yet."
    if patient_info:
        if isinstance(patient_info, dict):
            p_id = patient_info.get("patient_id", "Unknown")
            p_age = patient_info.get("age")
            p_markers = patient_info.get("genetic_markers", [])
        else:
            p_id = patient_info.patient_id
            p_age = patient_info.age
            p_markers = patient_info.genetic_markers
            
        age_str = str(p_age) if p_age else "Unknown"
        markers_str = ", ".join(p_markers) if p_markers else "None reported"
        patient_text = f"Patient ID: {p_id}\nAge: {age_str}\nGenetic Markers: {markers_str}"
        
    # Format MRI Context
    mri_text = "No scan uploaded or validated yet."
    if mri_info:
        if isinstance(mri_info, dict):
            gcs_uri = mri_info.get("gcs_uri", "Unknown")
            is_val = mri_info.get("is_validated", False)
            modality = mri_info.get("modality", "Unknown")
        else:
            gcs_uri = mri_info.gcs_uri
            is_val = mri_info.is_validated
            modality = mri_info.modality
            
        mri_text = f"GCP URI: {gcs_uri}\nValidated: {is_val}\nModality: {modality}"

    # Construct the highly structured prompt using XML-like tags for Gemini
    prompt = f"""You are Qanary, a highly advanced Quantum Machine Learning clinical assistant.
Your primary user is a certified oncologist or radiologist. Communicate with clinical rigor, using standard medical terminology.
Do not treat the physician as a patient. Be concise, mathematically precise, and clinically grounded.

<role_and_guardrails>
1. You are a Clinical Decision Support System (CDSS).
2. You MUST NEVER make a definitive medical diagnosis. 
3. Frame all findings as probabilities, feature maps, and anomalies derived from the Quantum CNN model.
4. Always explicitly state that human-in-the-loop verification by a certified physician is required.
</role_and_guardrails>

<workflow_awareness>
You are operating within a strict medical state machine. You must obey the rules of your current stage.

CURRENT STAGE: {stage}

--- CURRENT CLINICAL STATE ---
[Patient Context]
{patient_text}

[MRI Context]
{mri_text}
------------------------------

RULES BASED ON YOUR CURRENT STAGE:
- If INTAKE: If the user provides a scan URI, you MUST immediately execute the `lesion_cropper_tool`. DO NOT ask the physician for missing patient context (Age, BRCA markers). If patient context is missing, proceed with the analysis anyway, but add a brief clinical disclaimer in your final report noting that priors were unavailable.
- If QUANTUM_INFERENCE: You are authorized to use the `lesion_cropper_tool`. You must pass the exact GCP URI provided by the user or in the MRI Context.
- If REPORTING: Synthesize the tool's output into a highly professional, direct radiological summary.
</workflow_awareness>

<communication_style>
1. BE DIRECT AND ASSUMPTIVE. Physicians are extremely busy. Do not ask conversational follow-up questions like "Please advise on the next steps" or "Would you like me to...". 
2. Deliver the final medical summary, note any missing clinical context limitations, state the tool outputs clearly, and STOP.
3. Never use conversational filler.
</communication_style>

<tool_usage>
1. Use the `lesion_cropper_tool` immediately when a `gs://` URI is provided.
2. Rely strictly on the provided GCP Storage URIs (gs://...). Do NOT hallucinate file paths.
3. If a tool returns an error or low confidence, report it clinically and conclude the analysis.
</tool_usage>
"""
    return prompt
