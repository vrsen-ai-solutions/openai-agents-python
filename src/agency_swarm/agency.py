from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from agents import Agent as BaseAgent, FunctionTool, RunResult
from agents.items import MessageOutputItem, UserInputItem

from .agent import Agent
from .thread import ConversationThread

if TYPE_CHECKING:
    from agents import TResponseInputItem

# Type alias for agency chart structure
AgencyChartEntry = Union[Agent, List[Agent]]
AgencyChart = List[AgencyChartEntry]

# Placeholder for SendMessage Tool Name
SEND_MESSAGE_TOOL_NAME = "SendMessage"

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Basic config for now


# --- State Enum ---
class InteractionState(Enum):
    INITIALIZING = 1
    AGENT_TURN = 2
    ROUTING = 3
    FINISHED = 4
    ERROR = 5
    MAX_STEPS_REACHED = 6
    # WAITING_FOR_TOOL_OUTPUT = 7 # Add if SDK interaction requires it


class Agency:
    """
    Orchestrates interactions between multiple Agents using a defined communication chart.
    Manages conversation threads and facilitates message passing.
    """

    agents: Dict[str, Agent]  # Agent name -> Agent instance
    chart: AgencyChart
    parsed_chart: Dict[str, List[str]]  # Parsed chart: sender_name -> list[receiver_names]
    threads: Dict[str, ConversationThread]  # thread_id -> Thread instance
    interaction_context: Dict[str, Any]  # Added: Temp context during interaction
    shared_instructions: Optional[str] = None
    # TODO: Add other original Agency Swarm parameters if needed (e.g., shared_files, async_mode, default_settings)

    def __init__(
        self,
        agency_chart: AgencyChart,
        shared_instructions_path: Optional[str] = None,
        # TODO: Add other relevant params from original Agency Swarm (e.g., settings_path)
    ):
        """
        Initializes the Agency.

        Args:
            agency_chart: Defines the structure and communication paths between agents.
                          Example: [agent1, [agent1, agent2]] means agent1 can talk to user
                          and agent1 can send messages to agent2.
            shared_instructions_path: Optional path to a file containing instructions
                                       to be prepended to all agents' system prompts.
        """
        logger.info("Initializing Agency...")
        self.chart = agency_chart
        self.threads = {}
        self.interaction_context = {}  # Initialize interaction context

        # Load shared instructions if provided
        if shared_instructions_path:
            try:
                # Basic file loading, consider error handling and encoding
                with open(shared_instructions_path) as f:
                    self.shared_instructions = f.read()
                logger.info(f"Loaded shared instructions from {shared_instructions_path}")
            except FileNotFoundError:
                logger.warning(f"Shared instructions file not found at {shared_instructions_path}")
                self.shared_instructions = None
            except Exception as e:
                logger.error(
                    f"Error loading shared instructions from {shared_instructions_path}: {e}"
                )
                self.shared_instructions = None
        else:
            self.shared_instructions = None

        # --- Agent Registration and Chart Parsing ---
        self.agents, self.parsed_chart = self._parse_chart_and_register_agents(agency_chart)
        if not self.agents:
            raise ValueError("Agency chart must contain at least one agent.")
        logger.info(f"Agency initialized with agents: {list(self.agents.keys())}")
        logger.info(f"Parsed communication chart: {self.parsed_chart}")

        # --- Setup Communication Tools ---
        self._setup_communication_tools()

        # TODO: Implement loading/applying shared files if that parameter is added
        logger.info("Agency initialization complete.")

    def _parse_chart_and_register_agents(
        self, chart: AgencyChart
    ) -> Tuple[Dict[str, Agent], Dict[str, List[str]]]:
        """Parses the agency chart, registers agents, and builds communication map."""
        agents: Dict[str, Agent] = {}
        parsed: Dict[str, List[str]] = {}

        def register_agent(agent_instance):
            if not isinstance(agent_instance, BaseAgent):  # Check against base SDK agent type
                raise TypeError(
                    f"Invalid object in agency_chart: {agent_instance}. Must be an Agent instance."
                )
            if agent_instance.name in agents and id(agents[agent_instance.name]) != id(
                agent_instance
            ):
                # Allowing same agent instance multiple times is fine, but different instances with same name is error
                raise ValueError(
                    f"Duplicate agent name '{agent_instance.name}' found in agency_chart with different instances."
                )
            agents[agent_instance.name] = agent_instance

        for entry in chart:
            if isinstance(entry, BaseAgent):
                register_agent(entry)
            elif isinstance(entry, list) and len(entry) == 2:
                sender, receiver = entry
                register_agent(sender)
                register_agent(receiver)

                # Add communication path
                if sender.name not in parsed:
                    parsed[sender.name] = []
                if receiver.name not in parsed.get(sender.name, []):
                    parsed[sender.name].append(receiver.name)
            else:
                raise ValueError(f"Invalid agency_chart entry type: {type(entry)}. Entry: {entry}")

        return agents, parsed

    def _get_send_message_tool(self) -> FunctionTool:
        """Creates the SendMessage tool definition.
        The on_invoke implementation is a no-op as the orchestrator handles it.
        """
        schema = {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Name of the agent to send the message to.",
                    # TODO: Potentially add enum based on allowed recipients for the specific sender?
                },
                "message": {"type": "string", "description": "The message content to send."},
            },
            "required": ["recipient", "message"],
        }

        # This function does nothing, serves only to satisfy FunctionTool requirements.
        # The orchestrator intercepts calls to this tool name.
        async def dummy_invoke(ctx, args_json):
            logger.info(f"[SendMessage Tool] Invoked (handled by orchestrator). Args: {args_json}")
            return "Message dispatch handled by agency orchestrator."

        return FunctionTool(
            name=SEND_MESSAGE_TOOL_NAME,
            description="Sends a message to another specific agent within the agency framework.",
            params_json_schema=schema,
            on_invoke_tool=dummy_invoke,
        )

    def _setup_communication_tools(self) -> None:
        """Adds the SendMessage tool to agents that are allowed to send messages."""
        send_message_tool = self._get_send_message_tool()
        for sender_name in self.parsed_chart:
            if sender_name in self.agents:
                sender_agent = self.agents[sender_name]
                # Add the tool to the sender agent if they don't already have it
                # Need to check by name as instances might differ if called multiple times
                if not any(
                    tool.name == SEND_MESSAGE_TOOL_NAME
                    for tool in sender_agent.tools
                    if isinstance(tool, FunctionTool)
                ):
                    sender_agent.add_tool(send_message_tool)
                    logger.info(f"Added SendMessage tool to agent: {sender_name}")
            else:
                logger.warning(f"Sender agent '{sender_name}' defined in chart but not registered.")

    # --- Thread Management ---

    def get_or_create_thread(
        self, thread_id: Optional[str] = None
    ) -> Tuple[str, ConversationThread]:
        """Gets an existing thread or creates a new one."""
        if thread_id and thread_id in self.threads:
            logger.info(f"Retrieving existing thread: {thread_id}")
            return thread_id, self.threads[thread_id]
        else:
            # Create new thread
            new_thread = ConversationThread()
            new_id = new_thread.thread_id
            self.threads[new_id] = new_thread
            logger.info(f"Created new thread: {new_id}")
            return new_id, new_thread

    def delete_thread(self, thread_id: str) -> bool:
        """Deletes a conversation thread."""
        if thread_id in self.threads:
            del self.threads[thread_id]
            logger.info(f"Deleted thread: {thread_id}")
            return True
        logger.warning(f"Thread not found for deletion: {thread_id}")
        return False

    # --- Communication Check ---

    def is_communication_allowed(self, sender_name: str, recipient_name: str) -> bool:
        """Checks if communication is allowed based on the parsed chart."""
        return recipient_name in self.parsed_chart.get(sender_name, [])

    # --- Orchestration Methods (Implementing 14.1 & 14.2) ---

    def _parse_send_message_call(self, run_result: RunResult) -> Optional[Tuple[str, str]]:
        """Parses the first SendMessage tool call from a RunResult."""
        if not run_result.tool_calls:
            return None

        for tool_call in run_result.tool_calls:
            if tool_call.name == SEND_MESSAGE_TOOL_NAME:
                try:
                    # Assuming args_json is a dict after SDK parsing
                    args = tool_call.args_json
                    recipient = args.get("recipient")
                    message = args.get("message")
                    if recipient and message:
                        logger.debug(
                            f"Parsed SendMessage call: to {recipient}, message: '{message[:50]}...'"
                        )
                        return recipient, message
                    else:
                        logger.warning(
                            f"SendMessage tool call missing recipient or message: {args}"
                        )
                        return None  # Malformed call
                except Exception as e:
                    logger.error(
                        f"Error parsing SendMessage tool call arguments: {tool_call.args_raw}. Error: {e}",
                        exc_info=True,
                    )
                    return None  # Error during parsing

        return None  # No SendMessage tool call found

    async def _execute_agent_turn(
        self, agent: Agent, thread: ConversationThread, current_interaction_context: Dict[str, Any]
    ) -> RunResult:
        """Internal method to execute one turn of an agent using the agents SDK runner."""
        history = thread.get_history()  # Assuming get_history returns format suitable for run_sync
        logger.info(f"Executing turn for agent: {agent.name} in thread {thread.thread_id}")
        # TODO: Add shared instructions to agent if applicable before run?
        # Note: The base agents.Agent handles prepending instructions, ensure it's done correctly.

        try:
            # Assuming agency_swarm.Agent inherits from agents.Agent and has run_sync
            # The base SDK's run_sync should handle tool calls internally
            result: RunResult = await agent.run_sync(history)  # Use await for async run_sync
            logger.info(f"Agent {agent.name} completed turn. Result type: {type(result)}")

            # Log tool calls if any
            if result.tool_calls:
                logger.info(
                    f"Agent {agent.name} made tool calls: {[tc.name for tc in result.tool_calls]}"
                )
            else:
                logger.debug(f"Agent {agent.name} made no tool calls.")

            thread.add_run_result(result)  # Add the full RunResult to the thread
            return result
        except Exception as e:
            logger.error(f"Error during agent {agent.name} execution: {e}", exc_info=True)
            # TODO: Consider how errors should be propagated or handled (Subtask 14.5)
            raise  # Re-raise for now

    async def run_interaction(
        self,
        initial_agent: Union[str, Agent],
        message: TResponseInputItem,
        thread_id: Optional[str] = None,
        max_steps: int = 25,
    ) -> RunResult:
        """
        Runs a multi-agent interaction sequence based on the agency chart,
        tracking the interaction state.

        Args:
            initial_agent: The agent (instance or name) to start the interaction.
            message: The initial message or input to the agent.
            thread_id: Optional ID of an existing thread to continue.
            max_steps: Maximum number of agent turns allowed in the interaction.

        Returns:
            The RunResult from the *last* agent that executed in the sequence.
        """
        logger.info(
            f"Starting interaction... Initial agent: {initial_agent}, Thread ID: {thread_id}, Max steps: {max_steps}"
        )
        current_state = InteractionState.INITIALIZING
        logger.debug(f"Thread {thread_id or '(new)'} state -> {current_state}")

        # Resolve agent instance
        if isinstance(initial_agent, str):
            agent_name = initial_agent
            if agent_name not in self.agents:
                error_msg = f"Initial agent '{agent_name}' not found in this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
            current_agent = self.agents[agent_name]
        elif isinstance(initial_agent, Agent):
            current_agent = initial_agent
            if current_agent.name not in self.agents or id(self.agents[current_agent.name]) != id(
                current_agent
            ):
                error_msg = f"Provided agent instance '{current_agent.name}' is not registered with this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            raise TypeError(f"Invalid type for initial_agent: {type(initial_agent)}")

        # Get thread and add initial message
        thread_id, thread = self.get_or_create_thread(thread_id)
        thread.add_user_message(message)

        # Initialize interaction context with state
        # Use setdefault to avoid overwriting if context partially exists (e.g., from retry)
        interaction_data = self.interaction_context.setdefault(thread_id, {})
        interaction_data["state"] = InteractionState.ROUTING  # State before first agent turn
        current_state = interaction_data["state"]
        logger.debug(f"Thread {thread_id} state -> {current_state}")

        logger.info(
            f"Starting interaction loop for agent {current_agent.name} in thread {thread_id}"
        )

        # --- Interaction Loop ---
        current_step = 0
        last_result: Optional[RunResult] = None

        try:
            while current_step < max_steps:
                current_step += 1

                # Set state before agent execution
                interaction_data["state"] = InteractionState.AGENT_TURN
                current_state = interaction_data["state"]
                logger.info(
                    f"--- Interaction Step {current_step}/{max_steps}, Agent: {current_agent.name}, State: {current_state} ---"
                )

                result = await self._execute_agent_turn(
                    current_agent,
                    thread,
                    interaction_data,  # Pass the whole context dict
                )
                last_result = result

                # Check for SendMessage tool call to determine next agent
                send_call_data = self._parse_send_message_call(result)

                if send_call_data:
                    recipient_name, message_content = send_call_data
                    sender_name = current_agent.name

                    logger.info(
                        f"Agent {sender_name} requested to send message to {recipient_name}."
                    )

                    # Validate communication path
                    if not self.is_communication_allowed(sender_name, recipient_name):
                        error_msg = f"Communication DENIED: Agent '{sender_name}' is not allowed to send messages to '{recipient_name}' according to the agency chart."
                        logger.error(error_msg)
                        interaction_data["state"] = InteractionState.ERROR  # Update state on error
                        current_state = interaction_data["state"]
                        logger.debug(f"Thread {thread_id} state -> {current_state}")
                        # TODO: Decide how to handle disallowed communication
                        logger.warning(
                            "Ending interaction due to disallowed communication attempt."
                        )
                        break

                    # Find recipient agent
                    next_agent = self.agents.get(recipient_name)
                    if not next_agent:
                        logger.error(
                            f"Recipient agent '{recipient_name}' not found in agency. Ending interaction."
                        )
                        interaction_data["state"] = InteractionState.ERROR  # Update state on error
                        current_state = interaction_data["state"]
                        logger.debug(f"Thread {thread_id} state -> {current_state}")
                        # TODO: Handle missing recipient agent
                        break

                    # Prepare for next agent's turn
                    interaction_data["state"] = (
                        InteractionState.ROUTING
                    )  # State before next agent turn
                    current_state = interaction_data["state"]
                    logger.info(
                        f"Routing message from {sender_name} to {recipient_name}. State: {current_state}"
                    )
                    thread.add_user_message(message_content)
                    current_agent = next_agent
                    # Continue loop

                else:
                    # Agent turn completed without SendMessage, sequence ends
                    interaction_data["state"] = InteractionState.FINISHED
                    current_state = interaction_data["state"]
                    logger.info(
                        f"Agent {current_agent.name} did not send a message. Interaction sequence complete. State: {current_state}"
                    )
                    break

            if current_step >= max_steps and interaction_data["state"] not in [
                InteractionState.FINISHED,
                InteractionState.ERROR,
            ]:
                interaction_data["state"] = InteractionState.MAX_STEPS_REACHED
                current_state = interaction_data["state"]
                logger.warning(
                    f"Interaction reached max steps ({max_steps}). State: {current_state}"
                )

        except Exception as e:
            # Ensure state is updated even if error occurs outside the main checks
            if thread_id in self.interaction_context:
                self.interaction_context[thread_id]["state"] = InteractionState.ERROR
                current_state = self.interaction_context[thread_id]["state"]
            else:  # Should not happen if context was set
                current_state = InteractionState.ERROR
            logger.error(f"Interaction failed. State: {current_state}. Error: {e}", exc_info=True)
            raise
        finally:
            # Log final state before cleanup
            final_state = self.interaction_context.get(thread_id, {}).get(
                "state", InteractionState.ERROR
            )  # Get state safely
            logger.debug(
                f"Cleaning up interaction context for thread {thread_id}. Final state: {final_state}"
            )
            if thread_id in self.interaction_context:
                del self.interaction_context[thread_id]

        # --- Return Final Result ---
        logger.info(
            f"Interaction finished in thread {thread_id}. Returning last result. Final State: {final_state}"
        )
        if last_result is None:
            logger.error("Interaction ended unexpectedly with no result.")
            raise RuntimeError("Interaction ended without producing any result.")

        return last_result

    # --- Backward Compatibility & Demo Methods (Placeholders) ---

    async def run_interaction_streamed(
        self,
        initial_agent: Union[str, Agent],
        message: TResponseInputItem,
        thread_id: Optional[str] = None,
        max_steps: int = 25,
    ):
        """(Placeholder for Task 14/16) Runs the interaction and yields streaming results."""
        logger.warning("run_interaction_streamed is not fully implemented yet.")
        # TODO (Task 14/16): Implement streaming orchestration
        yield "Streaming not implemented."  # Placeholder yield

    async def get_completion(
        self,
        message: str,
        message_files: Optional[List[str]] = None,
        recipient_agent: Optional[Union[str, Agent]] = None,
        thread_id: Optional[str] = None,
        yield_messages: bool = False,  # Deprecated? SDK likely handles streaming differently
        # ... other original params?
    ) -> Union[str, List[Dict[str, Any]]]:
        """(Placeholder for Task 17) Backward compatibility for get_completion."""
        logger.warning("get_completion is a backward compatibility method and may be removed.")
        # TODO (Task 17): Map to run_interaction, handle message_files if supported
        if yield_messages:
            logger.warning("yield_messages is deprecated.")
            return []  # Or raise error

        # Simplified mapping for now
        if not recipient_agent:
            # Need a default recipient logic or raise error
            raise ValueError("recipient_agent must be specified for get_completion.")

        # Convert simple string message to TResponseInputItem if needed
        input_item = UserInputItem(content=message)  # Basic conversion

        result = await self.run_interaction(recipient_agent, input_item, thread_id)
        # Extract final message content (assuming it's in the last item)
        if result and result.items:
            last_item = result.items[-1]
            if isinstance(last_item, MessageOutputItem):
                return last_item.content
        return ""  # Default empty response

    async def get_completion_streamed(
        self,
        message: str,
        message_files: Optional[List[str]] = None,
        recipient_agent: Optional[Union[str, Agent]] = None,
        thread_id: Optional[str] = None,
        # ... other original params?
    ):
        """(Placeholder for Task 17) Backward compatibility for get_completion_streamed."""
        logger.warning(
            "get_completion_streamed is a backward compatibility method and may be removed."
        )
        # TODO (Task 17): Map to run_interaction_streamed
        yield "Streaming compatibility not implemented."

    def run_demo(self, height=600):
        """(Placeholder for Task 18) Runs a Gradio demo."""
        logger.warning("run_demo functionality is not implemented yet.")
        # TODO (Task 18): Re-implement demo using Gradio or similar
        print("Demo functionality not available in this version.")
