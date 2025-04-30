# Agency Class Design (Task 6 - Revised)

This document outlines the revised design for the `Agency` class, positioning it as **strictly a setup and configuration layer**. It does **not** handle runtime orchestration, which is managed entirely by the `agency_swarm.Agent` class.

## 1. Goals

*   Provide a familiar entry point (`Agency(...)`) for users migrating from the original Agency Swarm.
*   Manage the collection of `agency_swarm.Agent` instances.
*   Parse the `agency_chart` to understand allowed communication paths.
*   Inject the `send_message` tool into agents based on the `agency_chart`.
*   Initialize and configure the `ThreadManager` with persistence callbacks.
*   Configure individual agents with necessary references (e.g., to the `ThreadManager`, peer lists, `Agency` instance).
*   Provide primary user-facing interaction methods (`get_response`, `get_response_stream`) that **delegate directly** to the appropriate `Agent` method.
*   Provide backward-compatible methods (`get_completion`, `get_completion_stream`) that also delegate to the appropriate `Agent` method.
*   Set up shared resources (instructions, files - TBD) and default settings.
*   Facilitate demo methods (`run_demo`, `demo_gradio`).

## 2. Proposed Class: `Agency`

```python
from typing import List, Dict, Optional, Any, Sequence, Union, Tuple, Callable
import asyncio

# Import core types
from .agent import Agent # Assuming agency_swarm.Agent
from .thread import ConversationThread, ThreadManager, ThreadLoadCallback, ThreadSaveCallback # Revised design
from agents import RunItem, TResponseInputItem, FunctionTool, RunResult, BaseAgent # Placeholder imports

# Type alias for agency chart structure
AgencyChartEntry = Union[Agent, List[Agent]]
AgencyChart = List[AgencyChartEntry]

class Agency:
    """ Manages a collection of Agents, sets up communication paths,
        and provides delegation methods to initiate agent interactions.
        Does NOT handle runtime orchestration directly.
    """

    agents: Dict[str, Agent] # Agent name -> Agent instance
    chart: AgencyChart
    parsed_chart: Dict[str, List[str]] # Parsed chart: sender_name -> list[receiver_names]
    thread_manager: ThreadManager
    shared_instructions: Optional[str] = None
    # ... other Agency Swarm parameters like shared_files_path ...

    def __init__(
        self,
        agency_chart: AgencyChart,
        shared_instructions_path: Optional[str] = None,
        # Persistence Callbacks
        load_callback: Optional[ThreadLoadCallback] = None,
        save_callback: Optional[ThreadSaveCallback] = None,
        # ... other params from original Agency Swarm ...
    ):
        self.chart = agency_chart
        self.agents = {} # Populated in _register_agents
        self._register_agents() # Find all unique agents first
        self.parsed_chart = self._parse_chart(agency_chart)

        self.thread_manager = ThreadManager(load_callback, save_callback)

        if shared_instructions_path:
             self.shared_instructions = self._load_shared_instructions(shared_instructions_path)
            # TODO: Apply shared instructions to agents?

        self._configure_agents()
        # ... load/apply shared files, settings ...

    def _register_agents(self) -> None:
        """Finds all unique agent instances defined in the chart."""
        agent_set = set()
        for entry in self.chart:
            if isinstance(entry, Agent):
                agent_set.add(entry)
            elif isinstance(entry, list) and len(entry) == 2 and all(isinstance(a, Agent) for a in entry):
                agent_set.add(entry[0])
                agent_set.add(entry[1])
            else:
                raise ValueError(f"Invalid agency_chart entry: {entry}. Must contain Agent instances or lists of two Agent instances.")
        self.agents = {agent.name: agent for agent in agent_set}
        print(f"Registered agents: {list(self.agents.keys())}")

    def _parse_chart(self, chart: AgencyChart) -> Dict[str, List[str]]:
        """Parses the agency chart into an easier lookup structure (sender -> receivers)."""
        parsed = {}
        for entry in chart:
            if isinstance(entry, list): # Only communication paths matter here
                sender, receiver = entry
                if sender.name not in self.agents or receiver.name not in self.agents:
                    # This should not happen if _register_agents ran first
                    raise ValueError("Agent in chart path not found in registered agents.")
                if sender.name not in parsed:
                    parsed[sender.name] = []
                if receiver.name not in parsed.get(sender.name, []):
                     parsed[sender.name].append(receiver.name)
        return parsed

    def _load_shared_instructions(self, path: str) -> str:
         # TODO: Implement file loading
         print(f"[Agency] Loading shared instructions from {path}")
         return f"Shared instructions placeholder from {path}"

    def _get_send_message_tool(self) -> FunctionTool:
        """Creates the send_message tool definition.
           The actual function does nothing; it's intercepted by the receiving Agent.
        """
        # Define schema programmatically
        schema = {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Sends a message to another specified agent within the agency.",
                "parameters": {
            "type": "object",
            "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "Name of the recipient agent."
                        },
                        "message": {
                            "type": "string",
                            "description": "The message content to send."
                        }
            },
            "required": ["recipient", "message"]
        }
            }
        }
        # Create a dummy callable because FunctionTool requires one
        def _dummy_send_message_impl(recipient: str, message: str):
             # This function body will NOT be executed by the agent framework
             # The agent's internal loop intercepts calls to "send_message"
             print("[send_message tool] This should not be printed if intercepted correctly.")
             return "This is a dummy return value."

        # Decorate the dummy callable to satisfy FunctionTool creation if needed,
        # or create FunctionTool directly if API allows.
        # Assuming direct creation or a helper is available:
        # return FunctionTool.from_callable(_dummy_send_message_impl)
        # Need to confirm how to create FunctionTool with a specific schema.
        # Placeholder - Assuming a way to create it directly with schema:
        print("[Agency] Creating send_message tool definition (schema only)")
        return FunctionTool(schema=schema) # This likely needs adjustment based on SDK

    def _configure_agents(self) -> None:
        """Configures agents with necessary references and tools."""
        print("[Agency] Configuring agents...")
        send_message_tool = self._get_send_message_tool()

        for agent_name, agent_instance in self.agents.items():
            # Inject ThreadManager reference
            agent_instance._set_thread_manager(self.thread_manager)
            # Inject Agency reference (for accessing peers)
            agent_instance._set_agency_instance(self)

            # Determine allowed peers and inject
            allowed_peers = self.parsed_chart.get(agent_name, [])
            agent_instance._set_agency_peers(allowed_peers)

            # Add send_message tool if this agent can send messages
            if agent_name in self.parsed_chart:
                print(f"[Agency] Adding send_message tool to agent: {agent_name}")
                agent_instance.add_tool(send_message_tool)

            # TODO: Apply shared instructions?
            if self.shared_instructions:
                # How to best combine with agent's own instructions? Prepend?
                # agent_instance.instructions = self.shared_instructions + "\n\n" + (agent_instance.instructions or "")
                pass

    # --- Primary User-Facing Interaction Methods --- (Delegation to Agent)

    async def get_response(
        self,
        message: str, # Or TResponseInputItem?
        chat_id: str,
        recipient_agent: Union[str, Agent],
        text_only: bool = False,
        **kwargs
    ) -> Union[str, RunResult]:
        """Initiates an interaction by sending a message to the specified agent.
           Delegates the actual execution and orchestration to the agent's get_response method.
        """
        print(f"[Agency.get_response] Delegating to Agent {recipient_agent}")
        agent_name = recipient_agent if isinstance(recipient_agent, str) else recipient_agent.name
        if agent_name not in self.agents:
            raise ValueError(f"Recipient agent '{agent_name}' not found in agency.")

        agent_instance = self.agents[agent_name]

        # Direct delegation
        # The agent's get_response handles the full loop, including potential recursion
        result = await agent_instance.get_response(
            message=message, # Pass the initial message content
            chat_id=chat_id,
            sender_name=None, # Initial call originates from user/external
            text_only=text_only,
            **kwargs # Pass any extra kwargs to the agent method
        )

        # Return the result obtained from the agent
        return result # Agent method already returns str or RunResult based on text_only

    async def get_response_stream(
        self,
        message: str, # Or TResponseInputItem?
        chat_id: str,
        recipient_agent: Union[str, Agent],
        **kwargs
    ) -> AsyncGenerator[Any, None]: # Yield type determined by Agent.get_response_stream
        """Initiates a streaming interaction with the specified agent.
           Delegates the actual streaming execution and orchestration to the agent's
           get_response_stream method.
        """
        print(f"[Agency.get_response_stream] Delegating to Agent {recipient_agent}")
        agent_name = recipient_agent if isinstance(recipient_agent, str) else recipient_agent.name
        if agent_name not in self.agents:
            raise ValueError(f"Recipient agent '{agent_name}' not found in agency.")

        agent_instance = self.agents[agent_name]

        # Direct delegation of the stream
        async for chunk in agent_instance.get_response_stream(
            message=message,
            chat_id=chat_id,
            sender_name=None,
            **kwargs
        ):
            yield chunk # Yield chunks directly from the agent's stream

    # --- Backward Compatibility Methods --- (Also delegate)
    async def get_completion(
        self,
        message: str,
        chat_id: str,
        recipient_agent: Union[str, Agent], # Changed from starting_agent for clarity
        text_only: bool = True,
        **kwargs
    ) -> str:
        """Backward compatible method, delegates to the agent's get_response method.
           Note: Uses recipient_agent parameter like newer methods.
        """
        # Simplification: Directly call the agent's get_response via the new Agency method
        print(f"[Agency.get_completion] Delegating via get_response to Agent {recipient_agent}")
        result = await self.get_response(
            message=message,
            chat_id=chat_id,
            recipient_agent=recipient_agent,
            text_only=True, # Ensure text return for backward compatibility
            **kwargs
        )
        # get_response already handles text_only logic
        return result if isinstance(result, str) else str(result) # Ensure string

    async def get_completion_stream(
        self,
        message: str,
        chat_id: str,
        recipient_agent: Union[str, Agent], # Changed from starting_agent
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Backward compatible streaming method, delegates to Agent.get_response_stream.
           Yields strings for compatibility.
           Note: Uses recipient_agent parameter like newer methods.
        """
        # Simplification: Directly call the agent's stream via the new Agency method
        print(f"[Agency.get_completion_stream] Delegating via get_response_stream to Agent {recipient_agent}")
        async for chunk in self.get_response_stream(
            message=message,
            chat_id=chat_id,
            recipient_agent=recipient_agent,
            **kwargs
        ):
             # TODO: Adapt chunk processing to yield only strings as needed based on
             # the actual yield type of Agent.get_response_stream.
             # This requires knowing what Agent.get_response_stream yields (raw chunks, RunItems?).
             # Assuming it yields processable chunks or strings for now.
             yield str(chunk)

    # --- Demo Methods ---
    def run_demo(self, **kwargs):
         """Runs a terminal-based demo of the agency."""
         # TODO: Implement Gradio demo logic, likely initiating calls via get_completion.
         print("[Agency.run_demo] Demo functionality not implemented yet.")
         pass

    def demo_gradio(self, **kwargs):
         """Launches a Gradio web interface for the agency."""
         # TODO: Implement Gradio demo logic.
         print("[Agency.demo_gradio] Gradio demo not implemented yet.")
         pass

```

## 3. Key Design Decisions & Considerations:

*   **Strictly Setup/Delegation:** The `Agency` class **only** handles setup (agent registration, chart parsing, tool injection, ThreadManager config) and **delegates** all runtime execution requests (`get_response`, `get_completion`, etc.) directly to the appropriate `agency_swarm.Agent` instance's methods.
*   **No Runtime Logic:** The `Agency` contains **no** runtime execution loop, state management, or direct interaction with the LLM during a conversation turn.
*   **Setup Responsibility:** Its main jobs remain parsing the `agency_chart`, creating/configuring the `ThreadManager`, adding the `send_message` tool schema to agents, and injecting necessary references (`ThreadManager`, peer lists, `Agency` instance) into each `Agent`.
*   **Agent-Centric Orchestration:** All orchestration logic, including the execution loop, `send_message` interception, and recursive agent calls, resides **entirely** within the `agency_swarm.Agent` class.
*   **`send_message` Tool:** The tool schema is defined in `Agency`, but its interception and handling occur within the `Agent`'s internal execution logic.
*   **Persistence Setup:** Takes `load/save` callbacks and passes them to the `ThreadManager`.
*   **Backward Compatibility:** Provides `get_completion*` methods that now delegate through the primary `get_response*` methods, maintaining the original intent while simplifying the Agency class.

## 4. Open Questions/Refinements:

*   How exactly is the `send_message` `FunctionTool` created with just a schema, bypassing the need for a real callable? (Depends on SDK specifics). Maybe it *needs* a dummy callable even if it's never called.
*   How are shared resources (instructions, files) best applied to agents during configuration?
*   Final implementation details for demo methods.

This revised design **strictly defines** the `Agency` class as a setup/delegation layer, ensuring that the `Agent` class is the sole location for runtime orchestration logic, aligning with the Agent-centric design goal.
