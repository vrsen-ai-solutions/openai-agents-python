import asyncio
import logging
import os
import sys
import uuid

from agents import function_tool

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Set agency_swarm logger level specifically to INFO to see agent flow
logging.getLogger("agency_swarm").setLevel(logging.INFO)

# Define logger for this module
logger = logging.getLogger(__name__)

# --- Assume agency_swarm package is installed or in PYTHONPATH ---
# If running directly from the repo root, adjust path:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agency_swarm.agency import Agency
from agency_swarm.agent import Agent

# --- Define Tools for Worker --- #

# In-memory state for the worker agent (simple example)
worker_state = {}


@function_tool
def store_value(key: str, value: str) -> str:
    """Stores a value in the worker's memory under a specific key."""
    worker_state[key] = value
    logger.info(f"Tool 'store_value': Stored '{value}' under key '{key}'")
    return f"Successfully stored '{value}' for key '{key}'."


@function_tool
def retrieve_value(key: str) -> str:
    """Retrieves a value from the worker's memory using its key."""
    value = worker_state.get(key)
    if value:
        logger.info(f"Tool 'retrieve_value': Found '{value}' for key '{key}'")
        return f"The value for key '{key}' is '{value}'."
    else:
        logger.warning(f"Tool 'retrieve_value': Key '{key}' not found.")
        return f"Error: No value found for key '{key}'."


# --- Define Agents --- #

# Agent 1: User Interface Agent
ui_agent = Agent(
    name="UI_Agent",
    instructions="You are the user-facing agent. Receive user requests and delegate tasks (like storing or retrieving data) to the Worker Agent. Report the final result back to the user.",
)

# Agent 2: Worker Agent
worker_agent = Agent(
    name="Worker_Agent",
    description="A worker agent responsible for storing and retrieving simple key-value data.",
    instructions="You are the worker agent. Use your tools to store or retrieve data based on instructions from the UI Agent.",
    tools=[
        store_value,  # Add the new tools
        retrieve_value,
    ],
)

# --- Define Agency Chart --- #
agency_chart = [
    ui_agent,
    [ui_agent, worker_agent],
]

# --- Create Agency Instance --- #
agency = Agency(
    agency_chart=agency_chart,
    shared_instructions="All agents must be precise and follow instructions exactly.",
)

# --- Run Interaction --- #


async def run_conversation():
    print("\n--- Running Stateful Two-Agent Conversation Example (Testing Memory) ---")

    chat_id = f"chat_{uuid.uuid4()}"
    print(f"\nInitiating conversation with Chat ID: {chat_id}")

    # --- Turn 1: Ask worker to STORE a value --- #
    key_to_store = "user_data_1"
    value_to_store = "DataPointAlpha"
    user_message_1 = f"Please ask the worker agent to store the value '{value_to_store}' with the key '{key_to_store}'."
    print(f"\nUser Message 1 to {ui_agent.name}: '{user_message_1}' (Chat ID: {chat_id})")

    try:
        response1 = await agency.get_response(
            message=user_message_1,
            recipient_agent=ui_agent,
            chat_id=chat_id,
        )

        print(f"\n--- Turn 1 Finished (Store Value) ---")
        if response1 and response1.final_output:
            print(f"Turn 1 Final Output from {ui_agent.name}: '{response1.final_output}'")
        else:
            print("No valid response or final output received for Turn 1.")

        # --- Turn 2: Ask worker to RETRIEVE the value --- #
        print("\n--- Turn 2: Asking worker to retrieve the stored value (Testing Worker Memory) ---")
        user_message_2 = f"What value did the worker store for the key '{key_to_store}'?"
        print(f"\nUser Message 2 to {ui_agent.name}: '{user_message_2}' (Chat ID: {chat_id})")

        response2 = await agency.get_response(
            message=user_message_2,
            recipient_agent=ui_agent,
            chat_id=chat_id,
        )

        print(f"\n--- Turn 2 Finished (Retrieve Value) ---")
        if response2 and response2.final_output:
            final_output_2 = response2.final_output
            print(f"Turn 2 Final Output from {ui_agent.name}: '{final_output_2}'")
            # Verify the expected result is mentioned
            if isinstance(final_output_2, str) and value_to_store in final_output_2:
                print(f"  SUCCESS: Agent correctly reported the stored value ('{value_to_store}').")
            else:
                print(
                    f"  FAILURE: Agent did not report the stored value ('{value_to_store}'). Output was: {final_output_2}"
                )
        else:
            print("No valid response or final output received for Turn 2.")

        # --- Inspect the final conversation history --- #
        if agency.thread_manager:
            thread = agency.thread_manager.get_thread(chat_id)
            if thread:
                print(f"\n--- Final Conversation History (Chat ID: {chat_id}) ---")
                history_items = thread.get_history()
                print(f"Total items in history: {len(history_items)}")
                for i, item in enumerate(history_items):
                    print(f"Item {i+1}: {item}")
                print("----------------------------------------------------")
            else:
                print(f"\nWarning: Chat thread {chat_id} not found.")
        else:
            print("\nWarning: Agency does not have a ThreadManager instance.")

    except Exception as e:
        logging.error(f"An error occurred during the conversation: {e}", exc_info=True)


# --- Main Execution --- #
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(run_conversation())
