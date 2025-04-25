from langsmith import Client
from chatbot.utils.state import State
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompt name
PROMPT_NAME = "rag-prompt-with-transcript"

def call_model(state: State) -> State:
    """
    Call the LLM model with the current state, including messages, transcript, and search results.

    Args:
        state: The current state containing messages, transcript, and serialized search results

    Returns:
        Updated state with the model's response
    """
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    llm = client.pull_prompt(PROMPT_NAME, include_model=True)

    # Pass all state components to the model
    res = llm.invoke({
        "messages": state["messages"],
    })

    return {"messages": [res]}

