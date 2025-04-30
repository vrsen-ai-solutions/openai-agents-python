from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from agents import Agent as BaseAgent, FunctionTool, RunResult, TResponseInputItem
from agents.items import MessageOutputItem

from .agent import Agent
from .thread import ConversationThread, ThreadManager

if TYPE_CHECKING:
    pass

# Type alias for agency chart structure
AgencyChartEntry = Union[Agent, List[Agent]]
AgencyChart = List[AgencyChartEntry]

# Placeholder for SendMessage Tool Name
SEND_MESSAGE_TOOL_NAME = "SendMessage"

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Basic config for now


class Agency:
    """
    Manages a collection of Agents, sets up communication paths,
    and provides delegation methods to initiate agent interactions.
    Does NOT handle runtime orchestration directly.
    """

    agents: Dict[str, Agent]  # Agent name -> Agent instance
    chart: AgencyChart
    parsed_chart: Dict[str, List[str]]  # Parsed chart: sender_name -> list[receiver_names]
    thread_manager: ThreadManager
    shared_instructions: Optional[str] = None

    def __init__(
        self,
        agency_chart: AgencyChart,
        shared_instructions_path: Optional[str] = None,
    ):
        """
        Initializes the Agency.

        Args:
            agency_chart: Defines the structure and communication paths between agents.
                          Example: [agent1, [agent1, agent2]] means agent1 can talk to user
                          and agent1 can send messages to agent2.
            shared_instructions_path: Optional path to a file containing instructions
                                       to be prepended to all agents\' system prompts.
        """
        logger.info("Initializing Agency...")
        self.chart = agency_chart
        self.thread_manager = ThreadManager()  # Instantiate ThreadManager

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

        # --- Configure Agents ---
        self._configure_agents()

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
            # Ensure agent has a name before trying to use it as a key
            agent_name = getattr(agent_instance, "name", None)
            if not agent_name:
                raise ValueError(f"Agent instance {agent_instance} must have a 'name' attribute.")

            if agent_name in agents and id(agents[agent_name]) != id(agent_instance):
                # Allowing same agent instance multiple times is fine, but different instances with same name is error
                raise ValueError(
                    f"Duplicate agent name '{agent_name}' found in agency_chart with different instances."
                )
            agents[agent_name] = agent_instance

        for entry in chart:
            if isinstance(entry, BaseAgent):
                register_agent(entry)
            elif isinstance(entry, list) and len(entry) == 2:
                sender, receiver = entry
                # Ensure agents have names before proceeding
                sender_name = getattr(sender, "name", None)
                receiver_name = getattr(receiver, "name", None)
                if not sender_name or not receiver_name:
                    raise ValueError(
                        "Agents in communication pair list must have a 'name' attribute."
                    )

                register_agent(sender)
                register_agent(receiver)

                # Add communication path
                if sender_name not in parsed:
                    parsed[sender_name] = []
                if receiver_name not in parsed.get(sender_name, []):
                    parsed[sender_name].append(receiver_name)
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
                    getattr(tool, "name", None) == SEND_MESSAGE_TOOL_NAME
                    for tool in getattr(sender_agent, "tools", [])  # Use getattr for safety
                ):
                    # Assume sender_agent has an add_tool method
                    if hasattr(sender_agent, "add_tool") and callable(sender_agent.add_tool):
                        sender_agent.add_tool(send_message_tool)
                        logger.info(f"Added SendMessage tool to agent: {sender_name}")
                    else:
                        logger.warning(f"Agent {sender_name} does not have an add_tool method.")
            else:
                logger.warning(f"Sender agent '{sender_name}' defined in chart but not registered.")

    def _configure_agents(self) -> None:
        """Configures individual agents with necessary references from the agency."""
        logger.info("Configuring agents with agency references...")
        for agent_name, agent_instance in self.agents.items():
            # Set reference to the agency instance itself
            if hasattr(agent_instance, "_set_agency_instance"):
                agent_instance._set_agency_instance(self)
            else:
                logger.warning(f"Agent {agent_name} does not have _set_agency_instance method.")

            # Inject ThreadManager reference
            if hasattr(agent_instance, "_set_thread_manager") and hasattr(self, "thread_manager"):
                agent_instance._set_thread_manager(self.thread_manager)
            elif not hasattr(self, "thread_manager"):
                # This warning shouldn't trigger now
                logger.warning(f"Agency does not have a ThreadManager instance to inject.")
            else:
                logger.warning(f"Agent {agent_name} does not have _set_thread_manager method.")

            # Determine allowed peers and set them
            allowed_peers = self.parsed_chart.get(agent_name, [])
            if hasattr(agent_instance, "_set_agency_peers"):
                agent_instance._set_agency_peers(allowed_peers)
                logger.debug(f"Set allowed peers for {agent_name}: {allowed_peers}")
            else:
                logger.warning(f"Agent {agent_name} does not have _set_agency_peers method.")

            # TODO: Apply shared instructions?
            # if self.shared_instructions:
            #     if agent_instance.instructions:
            #          agent_instance.instructions = self.shared_instructions + "\\n\\n---\\n\\n" + agent_instance.instructions
            #     else:
            #          agent_instance.instructions = self.shared_instructions
            #     logger.debug(f"Applied shared instructions to agent: {agent_name}")

    # --- Thread Management ---

    def get_or_create_thread(
        self, thread_id: Optional[str] = None
    ) -> Tuple[str, ConversationThread]:
        """Gets an existing thread or creates a new one using ThreadManager."""
        # Delegate to ThreadManager
        thread = self.thread_manager.get_thread(thread_id)
        logger.info(f"Retrieved thread {thread.thread_id} via ThreadManager.")
        return thread.thread_id, thread

    def delete_thread(self, thread_id: str) -> bool:
        """Deletes a conversation thread using ThreadManager."""
        # Delegate deletion to ThreadManager
        deleted = self.thread_manager.delete_thread(thread_id)
        if deleted:
            logger.info(f"Deleted thread: {thread_id}")
        else:
            logger.warning(f"Thread not found for deletion: {thread_id}")
        return deleted

    # --- Communication Check ---

    def is_communication_allowed(self, sender_name: str, recipient_name: str) -> bool:
        """Checks if communication is allowed based on the parsed chart."""
        return recipient_name in self.parsed_chart.get(sender_name, [])

    # --- Orchestration Methods ---

    async def get_response(
        self,
        message: str,
        recipient_agent: Union[str, Agent],
        thread_id: Optional[str] = None,
        text_only: bool = False,
        **kwargs: Any,
    ) -> Union[str, RunResult]:
        """
        Initiates an interaction by sending a message to the specified agent.
        Delegates the actual execution and orchestration to the agent's get_response method.

        Args:
            message: The initial message content.
            recipient_agent: The agent (instance or name) to receive the message.
            thread_id: Optional ID of an existing thread to use.
            text_only: If True, request only the final text output from the agent.
            **kwargs: Additional keyword arguments to pass to the agent's get_response method.

        Returns:
            The final text output (if text_only=True) or the full RunResult object
            as returned by the agent.
        """
        logger.info(
            f"Agency delegating get_response to agent: {getattr(recipient_agent, 'name', recipient_agent)} in thread: {thread_id or '(new)'}"
        )

        # Resolve agent instance
        if isinstance(recipient_agent, str):
            agent_name = recipient_agent
            if agent_name not in self.agents:
                error_msg = f"Recipient agent '{agent_name}' not found in this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
            agent_instance = self.agents[agent_name]
        elif isinstance(recipient_agent, Agent):
            agent_instance = recipient_agent
            # Ensure the instance provided is actually managed by this agency
            if agent_instance.name not in self.agents or id(self.agents[agent_instance.name]) != id(
                agent_instance
            ):
                error_msg = f"Provided agent instance '{agent_instance.name}' is not registered with this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            raise TypeError(
                f"Invalid type for recipient_agent: {type(recipient_agent)}. Expected str or agency_swarm.Agent."
            )

        # --- Get Thread ID (create if None) ---
        # The agent's get_response method will handle thread creation/retrieval
        # via the ThreadManager injected during config, so we just need the ID.
        if not thread_id:
            # If no thread_id provided, generate one for the agent to use/create
            thread_id = f"as_thread_{uuid.uuid4()}"
            logger.info(f"No thread_id provided, generated new one: {thread_id}")

        # --- Direct Delegation ---
        try:
            result = await agent_instance.get_response(
                message=message,
                chat_id=thread_id,  # Pass the resolved/generated thread ID
                sender_name=None,  # Initial call is from user/external
                text_only=text_only,
                **kwargs,  # Pass through any additional arguments
            )
            # Agent's get_response is responsible for the full orchestration
            # and returning the appropriate type based on text_only flag.
            return result
        except Exception as e:
            logger.error(
                f"Error during delegated call to agent {agent_instance.name}.get_response: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception

    async def get_response_stream(
        self,
        message: str,
        recipient_agent: Union[str, Agent],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """
        Initiates a streaming interaction with the specified agent.
        Delegates the actual streaming execution and orchestration to the agent's
        get_response_stream method.

        Args:
            message: The initial message content.
            recipient_agent: The agent (instance or name) to receive the message.
            thread_id: Optional ID of an existing thread to use.
            **kwargs: Additional keyword arguments to pass to the agent's get_response_stream method.

        Yields:
            Chunks from the agent's streaming response. The exact type depends
            on the implementation of Agent.get_response_stream.
        """
        logger.info(
            f"Agency delegating get_response_stream to agent: {getattr(recipient_agent, 'name', recipient_agent)} in thread: {thread_id or '(new)'}"
        )

        # Resolve agent instance (same logic as get_response)
        if isinstance(recipient_agent, str):
            agent_name = recipient_agent
            if agent_name not in self.agents:
                error_msg = f"Recipient agent '{agent_name}' not found in this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
            agent_instance = self.agents[agent_name]
        elif isinstance(recipient_agent, Agent):
            agent_instance = recipient_agent
            if agent_instance.name not in self.agents or id(self.agents[agent_instance.name]) != id(
                agent_instance
            ):
                error_msg = f"Provided agent instance '{agent_instance.name}' is not registered with this agency."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            raise TypeError(
                f"Invalid type for recipient_agent: {type(recipient_agent)}. Expected str or agency_swarm.Agent."
            )

        # --- Get Thread ID (create if None) ---
        if not thread_id:
            thread_id = f"as_thread_{uuid.uuid4()}"
            logger.info(f"No thread_id provided, generated new one for stream: {thread_id}")

        # --- Direct Delegation of Stream ---
        try:
            async for chunk in agent_instance.get_response_stream(
                message=message,
                chat_id=thread_id,  # Pass the resolved/generated thread ID
                sender_name=None,  # Initial call is from user/external
                **kwargs,  # Pass through any additional arguments
            ):
                yield chunk  # Yield directly from the agent's stream generator
        except Exception as e:
            logger.error(
                f"Error during delegated call to agent {agent_instance.name}.get_response_stream: {e}",
                exc_info=True,
            )
            # How to signal error in async generator? Re-raise? Yield specific error object?
            # Re-raising seems most direct for now.
            raise

    # --- Backward Compatibility & Demo Methods ---

    async def get_completion(
        self,
        message: str,
        message_files: Optional[List[str]] = None,
        recipient_agent: Optional[Union[str, Agent]] = None,
        thread_id: Optional[str] = None,
        yield_messages: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """(Backward Compatibility) Initiates interaction, delegates via get_response.

        Args:
            message: User message content.
            message_files: Optional list of file paths (Note: Automatic upload not fully implemented).
            recipient_agent: The target agent (name or instance).
            thread_id: Optional thread ID.
            yield_messages: If True, returns the full thread history (list of dicts) instead of just the final text.

        Returns:
            The final agent text response (str) or the full thread history (List[Dict]).
        """
        logger.warning(
            "get_completion is a backward compatibility method. Use get_response for RunResult or text_only=True."
        )
        if not recipient_agent:
            raise ValueError("recipient_agent must be specified for get_completion.")
        if message_files:
            logger.warning(
                "message_files parameter in get_completion is not fully supported for automatic upload yet."
            )
            # TODO: Consider calling agent.upload_file here first if required?

        try:
            # Call the primary agency method, requesting RunResult (text_only=False)
            # so we can potentially return the full history if yield_messages=True.
            result = await self.get_response(
                message=message,
                recipient_agent=recipient_agent,
                thread_id=thread_id,
                text_only=False,  # Get RunResult to access history if needed
            )

            # Handle yield_messages
            if yield_messages:
                logger.warning(
                    "yield_messages=True behavior changed: Returning full thread log as list of dicts."
                )
                # Need thread_id from result to re-fetch thread state
                final_thread_id = getattr(
                    result, "thread_id", thread_id
                )  # Get ID from result if available
                if not final_thread_id:
                    # This shouldn't happen if get_response worked, but handle defensively
                    logger.error(
                        "Could not determine thread ID to fetch history for yield_messages."
                    )
                    return []
                _, final_thread = self.get_or_create_thread(final_thread_id)
                # Assuming get_full_log returns List[RunItem]
                # Convert RunItems to dicts for backward compatibility
                return [item.model_dump() for item in final_thread.get_full_log()]
            else:
                # Default: return final text output
                final_text = getattr(result, "final_output_text", None)  # Safely get text
                return final_text if final_text is not None else ""

        except Exception as e:
            logger.error(f"Error during get_completion: {e}", exc_info=True)
            return ""  # Original behavior on error?

    async def get_completion_stream(
        self,
        message: str,
        message_files: Optional[List[str]] = None,
        recipient_agent: Optional[Union[str, Agent]] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """(Backward Compatibility) Initiates streaming interaction, delegates via get_response_stream.

        Yields:
            Text chunks (str) from the agent's response stream.
        """
        logger.warning(
            "get_completion_stream is a backward compatibility method. Use get_response_stream."
        )
        if not recipient_agent:
            raise ValueError("recipient_agent must be specified for get_completion_stream.")
        if message_files:
            logger.warning(
                "message_files parameter in get_completion_stream is not fully supported yet."
            )
            # TODO: Handle file uploads before starting stream?

        try:
            # Call the primary agency streaming method
            async for chunk in self.get_response_stream(
                message=message, recipient_agent=recipient_agent, thread_id=thread_id
            ):
                # Adapt chunk format to yield simple text strings
                # This requires knowing Agent.get_response_stream's yield type.
                # Simple str conversion for now, might need refinement.
                yield str(chunk)

        except Exception as e:
            logger.error(f"Error during get_completion_stream: {e}", exc_info=True)
            yield f"Error: {e}"  # Yield error string

    def run_demo(self, height=600):
        """(Placeholder for Task 18) Runs a Gradio demo."""
        logger.warning("run_demo functionality is not implemented yet.")
        print("Demo functionality not available in this version.")

    def demo_gradio(self, height=600):
        """(Placeholder for Task 18) Launches a Gradio web interface for the agency."""
        logger.warning("demo_gradio functionality is not implemented yet.")
        print("Gradio demo functionality not available in this version.")
