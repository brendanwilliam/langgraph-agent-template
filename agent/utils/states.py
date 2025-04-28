from typing import TypedDict, Annotated, Dict, List, Any, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

# =====================================================================
# STATE DEFINITIONS FOR LANGGRAPH
# =====================================================================

# Define the base state for the LangGraph agent
class State(TypedDict):
    """
    Base state for the LangGraph agent.

    This state contains the essential components needed for a basic agent:
    - messages: The conversation history with add_messages reducer
    """
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], add_messages]

# Extended state that includes weather data storage
class WeatherState(TypedDict):
    """
    State that includes weather information storage.

    Extends the base State with:
    - weather_data: Dictionary mapping locations to their weather information
    """
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], add_messages]
    weather_data: Dict[str, Dict[str, Any]]  # Maps location to weather info

# Weather information model for structured data
class WeatherInfo(BaseModel):
    """
    Structured weather information.

    Used for storing weather data in a structured format within the state.
    """
    location: str
    temperature: float
    unit: str
    conditions: str
    timestamp: str

# Combined state for agents with multiple capabilities
class CombinedState(TypedDict):
    """
    Combined state for agents with multiple tool capabilities.

    This state includes:
    - messages: The conversation history
    - weather_data: Weather information by location
    - search_results: Search results from Tavily
    - tool_history: History of tool usage for analytics
    """
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], add_messages]
    weather_data: Dict[str, WeatherInfo]  # Structured weather data
    search_results: Dict[str, Any]  # Search results from Tavily
    tool_history: List[Dict[str, Any]]  # History of tool usage

# Tool usage record for analytics
class ToolUsage(BaseModel):
    """
    Record of tool usage for analytics.

    Tracks when and how tools are used in the agent.
    """
    tool_name: str
    timestamp: str
    parameters: Dict[str, Any]
    success: bool
    execution_time: float  # in seconds

# State with command processing capability
class CommandEnabledState(TypedDict):
    """
    State that supports command processing from tools.

    This state includes:
    - messages: The conversation history
    - commands: Queue of commands to be processed
    - command_results: Results of processed commands
    """
    messages: Annotated[List[Union[AIMessage, HumanMessage, ToolMessage]], add_messages]
    commands: List[Dict[str, Any]]  # Queue of commands to process
    command_results: Dict[str, Any]  # Results of processed commands