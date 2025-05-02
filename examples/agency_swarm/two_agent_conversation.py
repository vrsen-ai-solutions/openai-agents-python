import asyncio
import logging
import os

from agents import function_tool

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Assume agency_swarm package is installed or in PYTHONPATH ---
# If running directly from the repo root, adjust path:
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agency_swarm.agency import Agency
from agency_swarm.agent import Agent


@function_tool
def perform_task(task: str) -> str:
    return "The task is complete, the result is: 891."


# --- Define Agents ---

# Agent 1: User Interface Agent
ui_agent = Agent(
    name="UI_Agent",
    instructions="You are the user-facing agent. Receive user requests and delegate tasks to the Worker Agent.",
    # No specific tools needed for this simple example, will rely on send_message injected by Agency
)

# Agent 2: Worker Agent
worker_agent = Agent(
    name="Worker_Agent",
    instructions="You are the worker agent. Receive tasks from the UI Agent and perform them.",
    # Add a simple placeholder tool
    tools=[
        perform_task,
    ],
)


# --- Define Agency Chart ---
# Defines the structure and communication flow
# Here: UI_Agent can talk to Worker_Agent
agency_chart = [
    ui_agent,  # Entry point
    [ui_agent, worker_agent],  # Communication flow: UI -> Worker
]

# --- Create Agency Instance ---
agency = Agency(
    agency_chart=agency_chart,
    shared_instructions="All agents must be polite and efficient.",
)

# --- Run Interaction ---


async def run_conversation():
    print("\n--- Running Two-Agent Conversation Example ---")

    user_message = "Please tell the worker to describe its function."

    print(f"\nUser Message to {ui_agent.name}: '{user_message}'")

    try:
        # Use agency.get_response to interact with an entry point agent
        response = await agency.get_response(
            message=user_message,
            recipient_agent=ui_agent.name,  # Start with the UI Agent
        )

        if response:
            # The agent providing the final response is the one we initiated with (ui_agent)
            print(f"\nFinal Response from {ui_agent.name}:")
            # Access final_output for the result, check if it's a string
            final_output = response.final_output
            print(
                f"  Output: {final_output if isinstance(final_output, str) else type(final_output)}"
            )
            print(f"  Items Generated: {len(response.new_items)}")
            # Optionally print item details
            for item in response.new_items:
                print(f"    - {type(item).__name__}: {getattr(item, 'raw_item', 'N/A')}")

        # Ask about task result
        print("\n--- Asking about task result ---")
        user_message = "What was the task result?"
        print(f"\nUser Message to {ui_agent.name}: '{user_message}'")

        response = await agency.get_response(
            message=user_message,
            recipient_agent=ui_agent.name,
        )

        if response:
            print(f"\nFinal Response from {ui_agent.name}:")
            final_output = response.final_output
            print(
                f"  Output: {final_output if isinstance(final_output, str) else type(final_output)}"
            )

        # --- Inspect the conversation history ---
        # You would typically use persistence callbacks to access threads,
        # but for demonstration, we might access it via the internal manager
        # IF persistence wasn't set up. (Not recommended for production)
        if agency.thread_manager._threads:
            # Get the thread ID (assuming only one chat was created)
            chat_id = list(agency.thread_manager._threads.keys())[0]
            thread = agency.thread_manager.get_thread(chat_id)
            print("\n--- Conversation History ---")
            for item in thread.get_history():
                print(f"- {item}")
            print("--------------------------")

    except Exception as e:
        logging.error(f"An error occurred during the conversation: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(run_conversation())
