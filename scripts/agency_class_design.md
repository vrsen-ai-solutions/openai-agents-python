# Agency Class Design (Task 6 - Revised)

This document outlines the revised design for the `Agency` class, positioning it as a setup and configuration layer rather than a runtime orchestrator in the Agency Swarm SDK fork.

## 1. Goals

*   Provide a familiar entry point (`Agency(...)`) for users migrating from the original Agency Swarm.
*   Manage the collection of `agency_swarm.Agent` instances.
*   Parse the `agency_chart` to understand allowed communication paths.
*   Inject the `send_message` tool into agents based on the `agency_chart`.
*   Initialize and configure the `ThreadManager` with persistence callbacks.
*   Configure individual agents with necessary references (e.g., to the `ThreadManager`, peer lists, potentially a reference back to the `Agency` instance for accessing shared resources/agents).
*   Provide backward-compatible methods (`get_completion`, `get_completion_stream`) that delegate to the appropriate `Agent.get_response` methods.
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
        and provides backward compatibility methods.
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

    # --- User-Facing Interaction Methods (Backward Compatibility) ---

    async def get_completion(
        self,
        message: str,
        chat_id: str,
        starting_agent: Union[str, Agent],
        text_only: bool = True, # Default to True for original get_completion behavior
        **kwargs
    ) -> str:
        """Backward compatible method, delegates to Agent.get_response."""
        print(f"[Agency.get_completion] Received call. Delegating to Agent {starting_agent}")
        agent_name = starting_agent if isinstance(starting_agent, str) else starting_agent.name
        if agent_name not in self.agents:
            raise ValueError(f"Starting agent '{agent_name}' not found in agency.")

        agent_instance = self.agents[agent_name]

        # Call the agent's primary method
        result = await agent_instance.get_response(
            message=message,
            chat_id=chat_id,
            sender_name=None, # Call originates from user
            text_only=text_only,
            **kwargs
        )

        # Ensure text return for get_completion signature
        if isinstance(result, RunResult):
            return result.final_output_text
                else:
            return str(result) # Should already be string if text_only=True

    async def get_completion_stream(self, message: str, chat_id: str, starting_agent: Union[str, Agent], **kwargs) -> AsyncGenerator[str, None]:
        """Backward compatible streaming method, delegates to Agent.get_response_stream."""
        print(f"[Agency.get_completion_stream] Received call. Delegating to Agent {starting_agent}")
        agent_name = starting_agent if isinstance(starting_agent, str) else starting_agent.name
        if agent_name not in self.agents:
            raise ValueError(f"Starting agent '{agent_name}' not found in agency.")

        agent_instance = self.agents[agent_name]

        async for chunk in agent_instance.get_response_stream(
            message=message,
            chat_id=chat_id,
            sender_name=None,
            **kwargs
        ):
             # Ensure chunks are strings for backward compatibility?
             # The underlying stream might yield different types (RunItems, etc.)
             # TODO: Adapt chunk processing as needed based on get_response_stream's yield type.
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

*   **Thin Layer:** The `Agency` is primarily a configuration and setup class. It does not contain the runtime execution loop.
*   **Setup Responsibility:** Its main jobs are parsing the `agency_chart`, creating/configuring the `ThreadManager`, adding the `send_message` tool to agents, and injecting necessary references (`ThreadManager`, peer lists, `Agency` instance) into each `Agent`.
*   **Agent-Centric Delegation:** User-facing methods (`get_completion*`) delegate directly to the appropriate `Agent.get_response*` method.
*   **`send_message` Tool:** The tool is defined here but relies on the `Agent`'s internal logic to intercept calls to it.
*   **Persistence Setup:** Takes `load/save` callbacks and passes them to the `ThreadManager`.
*   **Backward Compatibility:** Provides `get_completion*` methods with signatures closer to the original Agency Swarm, acting as wrappers around the new agent-level methods.

## 4. Open Questions/Refinements:

*   How exactly is the `send_message` `FunctionTool` created with just a schema, bypassing the need for a real callable? (Depends on SDK specifics). Maybe it *needs* a dummy callable even if it's never called.
*   How are shared resources (instructions, files) best applied to agents during configuration?
*   Final implementation details for demo methods.

This revised design slims down the `Agency` class significantly, aligning with the goal of making `Agent` the core independent unit and keeping `Agency` as a setup/compatibility layer.
