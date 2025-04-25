from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from chatbot.utils.tools import google_search_and_extract
from chatbot.utils.state import CombinedState
from chatbot.utils.nodes import call_model, process_tool_results
# Load environment variables
load_dotenv()

tool_node = ToolNode(tools=[google_search_and_extract])

def should_continue(state: CombinedState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(CombinedState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("process_results", process_tool_results)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "process_results")
workflow.add_edge("process_results", "agent")

my_agent = workflow.compile()