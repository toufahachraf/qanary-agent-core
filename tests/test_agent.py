import uuid
import warnings
from langchain_core.messages import HumanMessage

# Filter out GCP/Langchain deprecation warnings for a cleaner terminal
warnings.filterwarnings("ignore")

from core.agent import qanary_agent
from core.state import WorkflowStage

def run_chat_loop():
    print("=========================================================")
    print(" Dr. Qanary - Quantum ML Clinical Assistant - INITIALIZED")
    print("=========================================================")
    print("Type your messages below. Type 'quit' or 'exit' to stop.\n")
    
    # We create a unique session ID for this terminal session 
    # so MemorySaver remembers the context!
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        try:
            user_input = input("\nDoctor: ")
            if user_input.lower() in ['quit', 'exit']:
                print("\nShutting down Qanary interface. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("\n[Qanary is processing...]\n")
            
            # Prepare the state update.
            # We assume INTAKE stage for simplicity in the test terminal.
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_stage": WorkflowStage.INTAKE,
                "session_id": config["configurable"]["thread_id"]
            }
            
            # Stream the events from the graph so we can watch it think
            for event in qanary_agent.stream(input_state, config):
                for node_name, node_data in event.items():
                    
                    if node_name == "tools":
                        # The tool finished executing
                        print(f"   --> [TOOL EXECUTED]: Successfully ran image processing.")
                        
                    elif node_name == "reasoning":
                        # The LLM responded
                        response_msg = node_data['messages'][-1]
                        
                        if response_msg.content:
                            print(f"\nQanary:\n{response_msg.content}")
                            
                        elif response_msg.tool_calls:
                            for tc in response_msg.tool_calls:
                                print(f"   --> [AGENT DECISION]: Calling tool '{tc['name']}' with args: {tc['args']}")

        except Exception as e:
            print(f"\n[SYSTEM ERROR]: {e}")

if __name__ == "__main__":
    run_chat_loop()
