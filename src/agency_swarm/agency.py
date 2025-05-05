# --- agency.py ---
import logging
import uuid
import warnings
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from agents import (
    RunHooks,
    RunResult,
)

from .agent import Agent
from .hooks import PersistenceHooks
from .thread import ThreadLoadCallback, ThreadManager, ThreadSaveCallback

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Type Aliases ---
AgencyChartEntry = Agent | list[Agent]
AgencyChart = list[AgencyChartEntry]


# --- Agency Class --- (Tasks 12, 13, 14, 15)
class Agency:
    """
    Orchestrates a collection of Agents defined by an agency chart.

    Handles agent registration, communication setup, context management,
    and provides entry points for initiating interactions.
    """

    agents: Dict[str, Agent]
    chart: AgencyChart
    entry_points: List[Agent]
    thread_manager: ThreadManager
    persistence_hooks: Optional[PersistenceHooks]
    shared_instructions: Optional[str]
    user_context: Dict[str, Any]  # Shared user context for MasterContext

    def __init__(
        self,
        agency_chart: AgencyChart,
        shared_instructions: Optional[str] = None,
        # shared_files_path: Optional[str] = None, # Keep internal name
        load_callback: Optional[ThreadLoadCallback] = None,
        save_callback: Optional[ThreadSaveCallback] = None,
        user_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # Add kwargs catcher
    ):
        """
        Initializes the Agency object, setting up agents, threads, and core functionalities.

        Handles backward compatibility with deprecated parameters.
        """
        logger.info("Initializing Agency...")

        # --- Handle Deprecated Args ---
        deprecated_args_used = {}
        # Handle thread callbacks mapping
        final_load_callback = load_callback
        final_save_callback = save_callback
        if "threads_callbacks" in kwargs:
            warnings.warn(
                "'threads_callbacks' is deprecated. Pass 'load_callback' and 'save_callback' directly.",
                DeprecationWarning,
                stacklevel=2,
            )
            threads_callbacks = kwargs.pop("threads_callbacks")
            if isinstance(threads_callbacks, dict):
                # Only override if new callbacks weren't provided explicitly
                if final_load_callback is None and "load" in threads_callbacks:
                    final_load_callback = threads_callbacks["load"]
                if final_save_callback is None and "save" in threads_callbacks:
                    final_save_callback = threads_callbacks["save"]
            deprecated_args_used["threads_callbacks"] = threads_callbacks

        # Handle other deprecated args
        if "shared_files" in kwargs:
            warnings.warn(
                "'shared_files' parameter is deprecated and shared file handling is not currently implemented.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["shared_files"] = kwargs.pop("shared_files")
            # self.shared_files_path = deprecated_args_used["shared_files"] # Store if needed for future impl.
        if "async_mode" in kwargs:
            warnings.warn(
                "'async_mode' is deprecated. Asynchronous execution is handled by the underlying SDK.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["async_mode"] = kwargs.pop("async_mode")
        if "send_message_tool_class" in kwargs:
            warnings.warn(
                "'send_message_tool_class' is deprecated. The send_message tool is configured automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["send_message_tool_class"] = kwargs.pop("send_message_tool_class")
        if "settings_path" in kwargs or "settings_callbacks" in kwargs:
            warnings.warn(
                "'settings_path' and 'settings_callbacks' are deprecated. Agency settings are no longer persisted this way.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["settings_path"] = kwargs.pop("settings_path", None)
            deprecated_args_used["settings_callbacks"] = kwargs.pop("settings_callbacks", None)
        agent_level_params = [
            "temperature",
            "top_p",
            "max_prompt_tokens",
            "max_completion_tokens",
            "truncation_strategy",
        ]
        for param in agent_level_params:
            if param in kwargs:
                warnings.warn(
                    f"Global '{param}' on Agency is deprecated. Set '{param}' on individual Agent instances instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                deprecated_args_used[param] = kwargs.pop(param)

        # Log if any deprecated args were used
        if deprecated_args_used:
            logger.warning(f"Deprecated Agency parameters used: {list(deprecated_args_used.keys())}")
        # Warn about any remaining unknown kwargs
        for key in kwargs:
            logger.warning(f"Unknown parameter '{key}' passed to Agency constructor.")

        # --- Assign Core Attributes (Use potentially mapped callbacks) ---
        self.chart = agency_chart
        self.shared_instructions = shared_instructions  # Direct string, no file loading here
        self.user_context = user_context or {}

        # --- Initialize Core Components (Use potentially mapped callbacks) ---
        self.thread_manager = ThreadManager(load_callback=final_load_callback, save_callback=final_save_callback)
        self.persistence_hooks = None
        if final_load_callback and final_save_callback:
            self.persistence_hooks = PersistenceHooks(final_load_callback, final_save_callback)
            logger.info("Persistence hooks enabled.")

        # --- Register Agents and Parse Chart ---
        self.agents = {}
        self.entry_points = []
        self._parse_chart_and_register_agents(agency_chart)
        if not self.agents:
            raise ValueError("Agency chart must contain at least one agent.")
        logger.info(f"Registered agents: {list(self.agents.keys())}")

        # --- Configure Agents & Communication ---
        self._configure_agents()

        # TODO: Re-evaluate shared file handling based on deprecated 'shared_files'? Maybe use files_folder?
        # if self.shared_files_path: # Check internal var if set from deprecated param
        #      logger.warning("Shared file handling is not yet implemented.")

        logger.info("Agency initialization complete.")

    def _parse_chart_and_register_agents(self, chart: AgencyChart) -> None:
        """Iterates through the chart, registers unique agents, and identifies entry points."""
        registered_agent_ids = set()

        for entry in chart:
            if isinstance(entry, list) and len(entry) >= 1:
                # Communication path or group
                sender_agent = entry[0]
                if not isinstance(sender_agent, Agent):
                    raise TypeError(f"Invalid sender type in chart entry: {type(sender_agent)}")
                if id(sender_agent) not in registered_agent_ids:
                    self._register_agent(sender_agent)
                    registered_agent_ids.add(id(sender_agent))

                # Register receivers if it's a pair
                if len(entry) == 2:
                    receiver_agent = entry[1]
                    if not isinstance(receiver_agent, Agent):
                        raise TypeError(f"Invalid receiver type in chart entry: {type(receiver_agent)}")
                    if id(receiver_agent) not in registered_agent_ids:
                        self._register_agent(receiver_agent)
                        registered_agent_ids.add(id(receiver_agent))
            elif isinstance(entry, Agent):
                # Standalone agent (potential entry point)
                if id(entry) not in registered_agent_ids:
                    self._register_agent(entry)
                    registered_agent_ids.add(id(entry))
                # Standalone agents are considered entry points
                if entry not in self.entry_points:
                    self.entry_points.append(entry)
                    logger.info(f"Identified agent '{entry.name}' as an entry point.")
            else:
                raise ValueError(f"Invalid agency_chart entry: {entry}")

        if not self.entry_points:
            logger.warning(
                "No explicit entry points identified in the agency chart. Any agent might need to be callable."
            )
            # If no clear entry points, maybe consider all agents as potential entry points?
            self.entry_points = list(self.agents.values())

    def _register_agent(self, agent: Agent):
        """Adds a unique agent instance to the agency's agent map."""
        agent_name = agent.name
        if agent_name in self.agents:
            # Should not happen if id check works in _parse_chart...
            # but double-check name uniqueness.
            if id(self.agents[agent_name]) != id(agent):
                raise ValueError(f"Duplicate agent name '{agent_name}' with different instances.")
            return  # Already registered this instance

        logger.debug(f"Registering agent: {agent_name}")
        self.agents[agent_name] = agent

    def _configure_agents(self) -> None:
        """Injects agency refs, thread manager, shared instructions, and configures agent communication
        by calling register_subagent for defined flows.
        """
        logger.info("Configuring agents...")
        # Build the communication map from the chart
        communication_map: Dict[str, List[str]] = {agent_name: [] for agent_name in self.agents}
        for entry in self.chart:
            if isinstance(entry, list) and len(entry) == 2:
                sender, receiver = entry
                # Ensure both are actual Agent instances before accessing name
                if isinstance(sender, Agent) and isinstance(receiver, Agent):
                    sender_name = sender.name
                    receiver_name = receiver.name
                    if sender_name in communication_map and receiver_name not in communication_map[sender_name]:
                        communication_map[sender_name].append(receiver_name)
                else:
                    logger.warning(f"Invalid agent types found in communication chart entry: {entry}. Skipping.")

        # Configure each agent
        for agent_name, agent_instance in self.agents.items():
            # Inject Agency reference (for access to full agents map) and ThreadManager
            agent_instance._set_agency_instance(self)
            agent_instance._set_thread_manager(self.thread_manager)

            # Apply shared instructions (prepend)
            if self.shared_instructions:
                if agent_instance.instructions:
                    agent_instance.instructions = self.shared_instructions + "\n\n---\n\n" + agent_instance.instructions
                else:
                    agent_instance.instructions = self.shared_instructions
                logger.debug(f"Applied shared instructions to agent: {agent_name}")

            # --- Register subagents based on the explicit communication map ---
            # This will internally create the specific send_message_to_... tools
            allowed_recipients = communication_map.get(agent_name, [])
            if allowed_recipients:
                logger.debug(f"Agent '{agent_name}' can send messages to: {allowed_recipients}")
                for recipient_name in allowed_recipients:
                    if recipient_name in self.agents:
                        recipient_agent = self.agents[recipient_name]
                        try:
                            # Call the modified register_subagent
                            agent_instance.register_subagent(recipient_agent)
                        except Exception as e:
                            logger.error(
                                f"Error registering subagent '{recipient_name}' for sender '{agent_name}': {e}",
                                exc_info=True,
                            )
                    else:
                        # This should not happen if chart parsing is correct
                        logger.warning(
                            f"Recipient agent '{recipient_name}' defined in chart for sender '{agent_name}' but not found in registered agents."
                        )
            else:
                logger.debug(
                    f"Agent '{agent_name}' has no explicitly defined outgoing communication paths in the chart."
                )

        logger.info("Agent configuration complete.")

    # --- Agency Interaction Methods ---
    async def get_response(
        self,
        message: Union[str, List[Dict[str, Any]]],
        recipient_agent: Union[str, Agent],
        chat_id: Optional[str] = None,  # If None, a new chat ID is generated
        context_override: Optional[Dict[str, Any]] = None,
        hooks_override: Optional[RunHooks] = None,
        **kwargs: Any,
    ) -> RunResult:
        """Initiates an interaction with a specified entry point agent."""
        target_agent = self._resolve_agent(recipient_agent)
        if target_agent not in self.entry_points:
            logger.warning(f"Recipient agent '{target_agent.name}' is not a designated entry point.")
            # Allow calling non-entry points? Or raise error?
            # Let's allow it for now, but log a warning.
            # raise ValueError(f"Recipient agent '{target_agent.name}' is not a designated entry point.")

        effective_hooks = hooks_override or self.persistence_hooks  # Use agency persistence hooks by default

        # Create or use provided chat_id
        if not chat_id:
            chat_id = f"chat_{uuid.uuid4()}"
            logger.info(f"Initiating new chat with agent '{target_agent.name}', chat_id: {chat_id}")

        # Delegate to the target agent's get_response method
        return await target_agent.get_response(
            message=message,
            sender_name=None,  # Indicate message is from user/external
            chat_id=chat_id,
            context_override=context_override,
            hooks_override=effective_hooks,
            **kwargs,
        )

    async def get_response_stream(
        self,
        message: Union[str, List[Dict[str, Any]]],
        recipient_agent: Union[str, Agent],
        chat_id: Optional[str] = None,
        context_override: Optional[Dict[str, Any]] = None,
        hooks_override: Optional[RunHooks] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Initiates a streaming interaction with a specified entry point agent."""
        target_agent = self._resolve_agent(recipient_agent)
        if target_agent not in self.entry_points:
            logger.warning(f"Recipient agent '{target_agent.name}' is not a designated entry point.")

        effective_hooks = hooks_override or self.persistence_hooks

        if not chat_id:
            chat_id = f"chat_{uuid.uuid4()}"
            logger.info(f"Initiating new stream chat with agent '{target_agent.name}', chat_id: {chat_id}")

        # Delegate to the target agent's get_response_stream method
        async for event in target_agent.get_response_stream(
            message=message,
            sender_name=None,
            chat_id=chat_id,
            context_override=context_override,
            hooks_override=effective_hooks,
            **kwargs,
        ):
            yield event

    def _resolve_agent(self, agent_ref: Union[str, Agent]) -> Agent:
        """Helper to get an agent instance from a name or instance."""
        if isinstance(agent_ref, Agent):
            # Ensure it's an agent managed by this agency instance
            if agent_ref.name in self.agents and id(self.agents[agent_ref.name]) == id(agent_ref):
                return agent_ref
            else:
                raise ValueError(f"Agent instance {agent_ref.name} is not part of this agency.")
        elif isinstance(agent_ref, str):
            agent_instance = self.agents.get(agent_ref)
            if not agent_instance:
                raise ValueError(f"Agent with name '{agent_ref}' not found in this agency.")
            return agent_instance
        else:
            raise TypeError("recipient_agent must be an Agent instance or agent name string.")

    # --- Deprecated Methods ---
    async def get_completion(
        self,
        message: str,
        recipient_agent: Union[str, Agent],
        **kwargs: Any,  # Pass through other args like chat_id, etc.
    ) -> str:
        """[DEPRECATED] Use get_response instead. Returns final text output."""
        logger.warning("Method 'get_completion' is deprecated. Use 'get_response' instead.")
        run_result = await self.get_response(message=message, recipient_agent=recipient_agent, **kwargs)
        return run_result.final_output_text or ""

    async def stream_completion(
        self, message: str, recipient_agent: Union[str, Agent], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """[DEPRECATED] Use get_response_stream instead. Yields text chunks."""
        logger.warning("Method 'stream_completion' is deprecated. Use 'get_response_stream' instead.")
        async for event in self.get_response_stream(message=message, recipient_agent=recipient_agent, **kwargs):
            # Yield only text events for backward compatibility
            if isinstance(event, dict) and event.get("event") == "text":
                # Check for 'data' field for text events, fallback to 'content' if needed?
                # For consistency with current Agent implementation (yielding {"event": "text", "data": ...})
                # let's prioritize 'data'
                data = event.get("data")
                if data:
                    yield data

    # --- Other Methods (Placeholder/Future) ---
    def is_communication_allowed(self, sender_name: str, recipient_name: str) -> bool:
        """Checks if direct communication is allowed based on the parsed chart."""
        # This check is implicitly handled now by agent._subagents during send_message validation
        # Kept here conceptually, but might not be needed externally.
        allowed_recipients = []
        for entry in self.chart:
            if isinstance(entry, list) and len(entry) == 2:
                if entry[0].name == sender_name:
                    allowed_recipients.append(entry[1].name)
        return recipient_name in allowed_recipients

    # Add run_demo, demo_gradio later if needed
