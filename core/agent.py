import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage

# Import our custom enterprise modules
from core.state import AgentState, WorkflowStage
from core.prompts import get_system_prompt
from tools.image_processing import lesion_cropper_tool

# 1. Initialize Tools and LLM
tools = [lesion_cropper_tool]

# We use Gemini 1.5 Pro via Vertex AI for enterprise data privacy.
# Temperature is set to 0.1 for high determinism (essential in clinical settings).
llm = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

# 2. Define the Reasoning Node
def reasoning_node(state: AgentState):
    """
    The core cognitive function of Qanary. 
    It reads the clinical state, generates the dynamic guardrail prompt,
    and calls Gemini to make a decision (Chat vs Tool Call).
    """
    # Inject current state (Patient Info, Workflow Stage) into the system prompt
    system_prompt_text = get_system_prompt(state)
    system_message = SystemMessage(content=system_prompt_text)
    
    # Combine the system instructions with the doctor's chat history
    messages = [system_message] + state["messages"]
    
    # Invoke the model
    response = llm_with_tools.invoke(messages)
    
    # LangGraph's add_messages reducer will automatically append this response
    return {"messages": [response]}

# 3. Build the State Graph (The Clinical State Machine)
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges (The Flow)
workflow.add_edge(START, "reasoning")

# Conditional Routing: 
# `tools_condition` automatically checks if Gemini's response contains a tool_call.
# If True -> routes to "tools" node. If False -> routes to END.
workflow.add_conditional_edges(
    "reasoning",
    tools_condition,
    {True: "tools", False: END}
)

# After the PyTorch tool finishes cropping, it routes back to Gemini 
# so Gemini can read the JSON result and summarize it for the doctor.
workflow.add_edge("tools", "reasoning")

# 4. Compile with Memory
# MemorySaver allows Qanary to remember the conversation across multiple interactions
memory = MemorySaver()
qanary_agent = workflow.compile(checkpointer=memory)

print("Qanary AI Agent successfully compiled and ready for inference.")
