import json
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from core.agent import qanary_agent
from core.state import WorkflowStage
from api.schemas import ChatRequest, ChatResponse, Artifacts

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """
    The main endpoint for the Frontend UI to communicate with the Qanary Agent.
    It takes the payload, injects it into LangGraph, and extracts the final medical report
    along with any generated artifacts (like cropped images).
    """
    try:
        # LangGraph memory thread config
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Build the initial state payload for the agent
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "current_stage": WorkflowStage.INTAKE,
            "session_id": request.session_id
        }
        
        # Inject optional context if provided by the UI
        if request.patient_context:
            input_state["patient"] = request.patient_context.model_dump()
            
        if request.mri_gcs_uri:
            input_state["mri_data"] = {
                "gcs_uri": request.mri_gcs_uri,
                "modality": "MRI",
                "is_validated": False
            }

        # Run the LangGraph agent synchronously
        # For a truly massive workload, this could be refactored to an async background task,
        # but for REST API flow, invoke() blocks until the pipeline finishes.
        final_state = qanary_agent.invoke(input_state, config)
        
        # Extract the final AI response text
        messages = final_state.get("messages", [])
        ai_response = messages[-1].content if messages else "No response generated."
        
        # Extract Artifacts (Parse the tool outputs to find the cropped image URI)
        artifacts = Artifacts()
        for msg in reversed(messages):
            if msg.type == "tool":
                try:
                    tool_output = json.loads(msg.content)
                    if tool_output.get("status") == "Success":
                        artifacts.cropped_image_uri = tool_output.get("cropped_image_uri")
                        artifacts.bounding_box = tool_output.get("bounding_box")
                        artifacts.confidence = tool_output.get("confidence")
                        break # Found the latest successful crop
                except Exception:
                    pass

        return ChatResponse(
            status="success",
            session_id=request.session_id,
            current_stage=final_state.get("current_stage", "COMPLETED"),
            ai_response=ai_response,
            artifacts=artifacts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
