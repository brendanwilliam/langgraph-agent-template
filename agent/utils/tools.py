import os
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import tool, BaseTool, ToolException
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import requests
import json

# For Tavily integration
from tavily import TavilyClient
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re

# Load environment variables
load_dotenv()

# =====================================================================
# TOOL DEFINITIONS
# =====================================================================

@tool
def tavily_search_and_extract(query: str, max_results: int = 3) -> str:
    """
    Search for information and extract content from relevant web pages.

    Args:
        query: The search query string.
        max_results: The maximum number of results to return.

    Returns:
        A summary of search results and extracted content.
    """
    try:
        # Get search results from Tavily
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search = tavily_client.search(
            query=query,
            max_results=max_results,
            include_answer="basic"
        )

        return search

    except Exception as e:
        # Handle errors gracefully
        return f"Error during search: {str(e)}\nFailed to retrieve information about '{query}'."

# =====================================================================
# CUSTOM TOOL CLASSES
# =====================================================================

# Example of a custom tool using Pydantic for schema definition
class WeatherInput(BaseModel):
    """Input for the weather tool."""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="fahrenheit", description="The temperature unit to use. Celsius or Fahrenheit")

class WeatherTool(BaseTool):
    """Tool that gets the current weather in a given location"""
    name: str = "get_weather"
    description: str = "Get the current weather in a location"
    args_schema: type = WeatherInput

    def _run(self, location: str, unit: str = "fahrenheit", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        # This is a placeholder implementation
        # In a real implementation, you would call a weather API here
        return f"The current weather in {location} is 72°{unit[0].upper()}"

# =====================================================================
# TOOL REGISTRATION AND CONFIGURATION
# =====================================================================

# Define all available tools
AVAILABLE_TOOLS = [
    tavily_search_and_extract,
    WeatherTool(),
    # Add more tools here
]

# Create a ToolNode for LangGraph integration
tool_node = ToolNode(tools=AVAILABLE_TOOLS)

# =====================================================================
# STATE MANAGEMENT HELPERS
# =====================================================================

def process_tool_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the results of tool execution and update the state.

    This function is used as a node in the LangGraph to process tool outputs
    and format them as messages that can be added to the conversation history.

    Args:
        state: The current state of the conversation

    Returns:
        Updated state with tool results added as messages
    """
    # Get the last tool call and its result
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    # If there's no last message or it's not a tool result, return unchanged state
    if not last_message or not hasattr(last_message, "tool_call_id"):
        return state

    # Create a new ToolMessage from the result
    tool_message = ToolMessage(
        content=last_message.content,
        tool_call_id=last_message.tool_call_id,
        name=last_message.name
    )

    # Return updated state with the new message
    return {"messages": [tool_message]}

# =====================================================================
# TOOL ROUTING FUNCTIONS
# =====================================================================

def should_use_tool(state: Dict[str, Any]) -> str:
    """
    Determine if we should route to a tool based on the current state.

    This function examines the last AI message to check if it contains
    tool calls. If it does, route to the tools node, otherwise end the graph.

    Args:
        state: The current state of the conversation

    Returns:
        Next node to route to ("tools" or END)
    """
    messages = state.get("messages", [])
    if not messages:
        return "END"

    last_message = messages[-1]

    # Check if the last message is from the AI and has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # No tool calls, end the graph
    return "END"

# =====================================================================
# ADVANCED TOOL INTEGRATION PATTERNS
# =====================================================================

# Example of a Command object for updating state from tools
class Command(BaseModel):
    """Command object for updating state from tools."""
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)

# Example of a custom tool that returns a Command object
class UpdateStateWeatherTool(BaseTool):
    """Tool that gets weather and updates state with a command object."""
    name: str = "update_state_weather"
    description: str = "Get weather and update state with the information"
    args_schema: type = WeatherInput

    def _run(
        self,
        location: str,
        unit: str = "fahrenheit",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Use the tool and return a Command to update state."""
        # Get weather (placeholder implementation)
        weather_info = f"The current weather in {location} is 72°{unit[0].upper()}"

        # Return a Command object that will be processed to update state
        return {
            "content": weather_info,
            "command": Command(
                name="update_weather_state",
                args={
                    "location": location,
                    "weather": weather_info,
                    "timestamp": "2023-04-28T12:00:00Z"  # Example timestamp
                }
            )
        }

# Function to process Command objects from tools
def process_commands(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process Command objects from tool results and update state accordingly.

    Args:
        state: The current state

    Returns:
        Updated state
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]

    # Check if the last message has a command
    if hasattr(last_message, "additional_kwargs") and "command" in last_message.additional_kwargs:
        command = last_message.additional_kwargs["command"]

        # Process different command types
        if command.name == "update_weather_state":
            # Update weather information in state
            weather_state = state.get("weather_data", {})
            weather_state[command.args["location"]] = {
                "info": command.args["weather"],
                "timestamp": command.args["timestamp"]
            }

            # Return updated state
            return {**state, "weather_data": weather_state}

    # If no command or unrecognized command, return state unchanged
    return state