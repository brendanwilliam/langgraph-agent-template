
# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Local imports
from agent.utils.states import CombinedState, CommandEnabledState
from agent.utils.tools import AVAILABLE_TOOLS, tool_node, should_use_tool, process_commands
from agent.utils.nodes import (
    call_model, 
    process_search_results, 
    track_tool_usage, 
    update_tool_execution_time,
    handle_weather_updates
)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# =====================================================================
# BASIC GRAPH DEFINITION
# =====================================================================

def create_basic_graph():
    """Create a basic LangGraph with tool support."""
    # Create a new graph with the CombinedState type
    workflow = StateGraph(CombinedState)

    # Add nodes to the graph
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_results", process_search_results)

    # Add edges to the graph
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        should_use_tool, 
        {"tools": "tools", "END": END}
    )
    workflow.add_edge("tools", "process_results")
    workflow.add_edge("process_results", "agent")

    # Compile the graph
    return workflow.compile()

# =====================================================================
# ADVANCED GRAPH WITH ANALYTICS
# =====================================================================

def create_analytics_graph():
    """Create a LangGraph with tool support and analytics tracking."""
    # Create a new graph with the CombinedState type
    workflow = StateGraph(CombinedState)

    # Add nodes to the graph
    workflow.add_node("agent", call_model)
    workflow.add_node("track_tools", track_tool_usage)
    workflow.add_node("tools", tool_node)
    workflow.add_node("update_metrics", update_tool_execution_time)
    workflow.add_node("process_results", process_search_results)

    # Add edges to the graph
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tool,
        {"tools": "track_tools", "END": END}
    )
    workflow.add_edge("track_tools", "tools")
    workflow.add_edge("tools", "update_metrics")
    workflow.add_edge("update_metrics", "process_results")
    workflow.add_edge("process_results", "agent")

    # Compile the graph
    return workflow.compile()

# =====================================================================
# COMMAND-ENABLED GRAPH
# =====================================================================

def create_command_graph():
    """Create a LangGraph with command processing capabilities."""
    # Create a new graph with the CommandEnabledState type
    workflow = StateGraph(CommandEnabledState)

    # Add nodes to the graph
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_results", process_search_results)
    workflow.add_node("process_commands", process_commands)

    # Add edges to the graph
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        should_use_tool, 
        {"tools": "tools", "END": END}
    )
    workflow.add_edge("tools", "process_results")
    workflow.add_edge("process_results", "process_commands")
    workflow.add_edge("process_commands", "agent")

    # Compile the graph
    return workflow.compile()

# =====================================================================
# DEFAULT GRAPH INSTANCE
# =====================================================================

# Create the default graph instance
my_agent = create_basic_graph()

# Uncomment one of these lines to use a different graph configuration
my_analytics = create_analytics_graph()
my_command = create_command_graph()