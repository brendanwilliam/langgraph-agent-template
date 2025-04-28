import os
import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.prebuilt import ToolNode

# Local imports
from agent.utils.states import State, CombinedState, CommandEnabledState, WeatherState
from agent.utils.tools import (
    AVAILABLE_TOOLS, 
    tavily_search_and_extract, 
    process_tool_results,
    process_commands,
    WeatherTool,
    UpdateStateWeatherTool
)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# =====================================================================
# MODEL CONFIGURATION
# =====================================================================

def get_llm(model_name: str = "gpt-3.5-turbo") -> BaseChatModel:
    """Initialize and return an LLM."""
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# =====================================================================
# PROMPT TEMPLATES
# =====================================================================

# System prompt for the agent
SYSTEM_PROMPT = """
You are a helpful AI assistant with access to various tools.

When you receive a request that requires using tools, you should use them.
Tools available to you:
{tools}

Follow these guidelines:
1. Understand what the user is asking for
2. Use appropriate tools when needed
3. Provide clear and concise responses
4. If you use a tool, wait for its response before continuing
"""

# Create a prompt template for the agent
def create_agent_prompt():
    """Create a prompt template for the agent."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

# =====================================================================
# GRAPH NODES
# =====================================================================

def call_model(state: State) -> Dict[str, Any]:
    """Call the LLM to generate a response or tool calls.

    Args:
        state: The current state of the conversation

    Returns:
        Updated state with the model's response
    """
    # Initialize the LLM
    llm = get_llm(model_name="o4-mini")
    llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

    # Create a prompt with tool descriptions
    prompt = create_agent_prompt()

    # Format the prompt with tool descriptions
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" 
                               for tool in AVAILABLE_TOOLS 
                               if hasattr(tool, 'name') and hasattr(tool, 'description')])

    formatted_prompt = prompt.partial(tools=tool_descriptions)

    # Get the messages from state
    messages = state.get("messages", [])

    # Apply the formatted prompt to create a chain
    chain = formatted_prompt | llm_with_tools

    # Invoke the chain with the messages
    response = chain.invoke({"messages": messages})

    # Return the updated state with the model's response
    return {"messages": [response]}


def process_search_results(state: CombinedState) -> Dict[str, Any]:
    """Process search results from Tavily and update the state.

    Args:
        state: The current state with search results

    Returns:
        Updated state with processed search results
    """
    # Get the messages from the state
    messages = state.get("messages", [])
    if not messages:
        return state

    # Find the last tool message that contains search results
    search_message = None
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "tavily_search_and_extract":
            search_message = message
            break

    if not search_message:
        return state

    # Extract the search results
    search_content = search_message.content

    # Update the search_results in the state
    search_results = state.get("search_results", {})
    search_results["latest"] = search_content

    # Instead of creating a new tool message, just update the state
    # and let the existing messages flow through
    return {
        "search_results": search_results
    }


def track_tool_usage(state: CombinedState) -> Dict[str, Any]:
    """Track tool usage for analytics.

    Args:
        state: The current state

    Returns:
        Updated state with tool usage tracking
    """
    # Get the messages from the state
    messages = state.get("messages", [])
    if not messages:
        return state

    # Get the last message
    last_message = messages[-1]

    # If it's an AI message with tool calls, track the tool usage
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Get the tool history
        tool_history = state.get("tool_history", [])

        # Add a record for each tool call
        for tool_call in last_message.tool_calls:
            # Safely get parameters - handle different tool call structures
            parameters = {}
            if isinstance(tool_call, dict):
                # Handle dictionary format
                tool_name = tool_call.get("name", "unknown_tool")
                parameters = tool_call.get("args", {}) or tool_call.get("input", {}) or {}
            else:
                # Handle object format
                tool_name = getattr(tool_call, "name", "unknown_tool")
                parameters = getattr(tool_call, "args", {}) or getattr(tool_call, "input", {}) or {}

            tool_record = {
                "tool_name": tool_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": parameters,
                "success": True,  # Assume success initially
                "execution_time": 0.0  # Will be updated after execution
            }
            tool_history.append(tool_record)

        # Return the updated state
        return {**state, "tool_history": tool_history}

    return state


def update_tool_execution_time(state: CombinedState) -> Dict[str, Any]:
    """Update the execution time for tools after they've been executed.

    Args:
        state: The current state

    Returns:
        Updated state with execution times
    """
    # Get the tool history
    tool_history = state.get("tool_history", [])
    if not tool_history:
        return state

    # Get the last tool record
    last_tool = tool_history[-1]

    # Update the execution time (in a real implementation, this would be measured)
    last_tool["execution_time"] = 0.5  # Example execution time in seconds

    # Return the updated state
    return {**state, "tool_history": tool_history}


def handle_weather_updates(state: WeatherState) -> Dict[str, Any]:
    """Handle weather updates from the weather tool.

    Args:
        state: The current state

    Returns:
        Updated state with weather information
    """
    # Get the messages from the state
    messages = state.get("messages", [])
    if not messages:
        return state

    # Find the last tool message from the weather tool
    weather_message = None
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "get_weather":
            weather_message = message
            break

    if not weather_message:
        return state

    # Extract the location from the tool call
    # In a real implementation, you would parse this from the message
    location = "Example Location"  # Placeholder

    # Update the weather_data in the state
    weather_data = state.get("weather_data", {})
    weather_data[location] = {
        "info": weather_message.content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Return the updated state
    return {**state, "weather_data": weather_data}
