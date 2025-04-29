# Agency Swarm Agent Class Design (Task 5 - Revised)

This document details the revised design for the `agency_swarm.Agent` class, inheriting from `agents.Agent` and acting as the primary execution unit, incorporating features from the original Agency Swarm framework and supporting multi-agent orchestration.

## 1. Goals

*   Provide a familiar interface for users migrating from the original Agency Swarm.
*   Integrate seamlessly with the OpenAI Agents SDK foundation (`agents.Agent`).
*   Act as the main entry point for agent execution (`get_response`).
*   Manage the internal execution loop, including LLM calls, standard tool execution, and handling `send_message` tool calls for invoking other agents.
*   Support both SDK (`@function_tool`) and potentially Agency Swarm tool definitions (loading from folder - postponed for V1).
*   Implement Agency Swarm's file/vector store handling logic without relying on `settings.json`.
*   Interact with the `ThreadManager` to use the correct, isolated `ConversationThread` for each interaction (User <-> Self, Self <-> Other Agent).
*   Remove all dependencies on the OpenAI Assistants API v1/v2.
*   Align with Agent2Agent principles by being a self-contained, executable unit.

## 2. Proposed Class: `agency_swarm.Agent`

```python
from typing import List, Optional, Type, Union, Dict, Any, Callable, AsyncGenerator
from pathlib import Path
import asyncio

# Import base class and necessary types from SDK
from agents import Agent as BaseAgent
from agents import ModelSettings, Tool, RunResult, RunResultStreaming, AgentOutputSchemaBase, TContext, RunItem, TResponseInputItem, FunctionTool, NextStepRunAgain, NextStepFinalOutput

# Import Agency Swarm specific concepts/helpers (placeholders)
from .thread import ConversationThread, ThreadManager # Revised design
# from .tools import ToolFactory # Postponed for V1
# from .util import some_file_handling_utility

# Placeholder for original Agency Swarm parameters to retain
OriginalAgencySwarmParams = Dict[str, Any]

class Agent(BaseAgent):
    """ Extends OpenAI's Agent with Agency Swarm features and orchestration logic. """

    # --- Parameters to Retain/Adapt from Agency Swarm ---
    files_folder: Optional[Union[str, Path]] = None
    tools_folder: Optional[Union[str, Path]] = None # For future tool loading
    response_validator: Optional[Callable[[str], bool]] = None # Or adapt type signature
    # Add description if distinct from handoff_description for compatibility?

    # --- Internal State/Config ---
    _thread_manager: Optional[ThreadManager] = None # Set during Agency init or globally?
    _agency_chart_peers: Optional[List[str]] = None # Allowed agents to call via send_message, set by Agency
    _agency_instance: Optional[Any] = None # Reference back to Agency if needed for shared resources?

    # Inherits base SDK parameters

    def __init__(self, **kwargs):
        # Separate kwargs for BaseAgent and self
        base_agent_params = {k for k, _ in BaseAgent.__annotations__.items()}
        base_kwargs = {k: v for k, v in kwargs.items() if k in base_agent_params}
        swarm_kwargs = {k: v for k, v in kwargs.items() if k not in base_agent_params}

        super().__init__(**base_kwargs)

        # Initialize Agency Swarm specific attributes
        self.files_folder = swarm_kwargs.get('files_folder')
        self.tools_folder = swarm_kwargs.get('tools_folder') # Tool loading postponed
        self.response_validator = swarm_kwargs.get('response_validator')
        # ... initialize others from swarm_kwargs ...

        # Internal state needs to be configured, likely by the Agency
        self._thread_manager = swarm_kwargs.get('_thread_manager')
        self._agency_chart_peers = swarm_kwargs.get('_agency_chart_peers')
        self._agency_instance = swarm_kwargs.get('_agency_instance')

        # --- Initialization Logic ---
        # self._load_tools_from_folder() # Postponed
        self._init_file_handling()

    def _set_thread_manager(self, manager: ThreadManager): # Method for Agency to set this
        self._thread_manager = manager

    def _set_agency_peers(self, peers: List[str]): # Method for Agency to set this
        self._agency_chart_peers = peers

    def _set_agency_instance(self, agency: Any):
        self._agency_instance = agency

    # --- Tool Handling (Simplified for V1) ---

    def add_tool(self, tool: Union[Callable, FunctionTool, Type]) -> None:
        """Adds a tool. Currently primarily supports @function_tool decorated functions."""
        # TODO: Refine this to properly handle different inputs post-V1
        if isinstance(tool, FunctionTool):
            if tool not in self.tools:
                 self.tools.append(tool)
        elif callable(getattr(tool, "_is_function_tool", False)):
            # Attempt to detect @function_tool decorated functions
            # Need a reliable mechanism from the SDK or introspection
            func_tool = FunctionTool.from_callable(tool)
            if func_tool not in self.tools:
                 self.tools.append(func_tool)
        # elif isinstance(tool, type): # Postponed ToolFactory logic
        #     pass
        else:
            # For V1, maybe just try FunctionTool.from_callable and let it raise error?
            try:
                 func_tool = FunctionTool.from_callable(tool)
                 if func_tool not in self.tools:
                     self.tools.append(func_tool)
            except Exception as e:
                 raise ValueError(f"Unsupported tool type or callable for V1: {type(tool)}. Use @function_tool. Error: {e}")

    # _load_tools_from_folder postponed

    async def get_all_tools(self) -> list[Tool]:
        """Returns tools configured for the agent."""
        # For V1, relies on tools added via add_tool/constructor and base SDK logic
        return await super().get_all_tools()

    # --- File Handling ---

    def _init_file_handling(self) -> None:
        """Initializes file handling logic, creating folder if needed."""
        # Adds agent-level parameters and methods (files_folder, upload_file) for managing persistent file storage
        # and vector store associations using specific naming conventions (files_folder_<VS_ID>, myfile_<FILE_ID>.ext),
        # mimicking original Agency Swarm functionality and differing from the base SDK's agent class which lacks
        # these built-in agent-level persistence features.
        if self.files_folder:
            folder_path = Path(self.files_folder)
            # TODO: Implement logic to extract VS_ID if present in folder_path.name
            # vs_id = parse_vs_id(folder_path.name)
            base_folder_path = folder_path # TODO: Adjust if VS_ID was parsed
            base_folder_path.mkdir(parents=True, exist_ok=True)
            # Store VS_ID if found

    def upload_file(self, file_path: str) -> str: # Return File ID
        """Uploads a file, associates with agent's VS (if exists), manages naming."""
        # TODO: Implement file upload logic, VS association, naming convention (`basename_<FILE_ID>.ext`)
        # Check for existence based on base name/hash first.
        file_id = "file_placeholder_id" # Placeholder
        return file_id

    def check_file_exists(self, file_path: str) -> Optional[str]: # Return File ID if found
        """Checks if a file seems to be already uploaded based on naming convention."""
        # TODO: Implement logic to check for `basename_<FILE_ID>.ext` in files_folder.
        return None

    # --- Core Execution Logic ---

    async def get_response(
        self,
        message: str,
        chat_id: str, # Explicit chat_id for thread management
        sender_name: Optional[str] = None, # None if user, or name of calling agent
        current_depth: int = 0, # Tracks recursion depth for inter-agent calls
        text_only: bool = False,
        max_steps: int = 25, # Safety limit for the *internal* loop of this agent
        max_recursion_depth: int = 5, # Safety limit for A->B->C... calls
        **kwargs: Any # Pass additional args to model provider?
    ) -> Union[str, RunResult]:
        """The main entry point for agent execution and orchestration.
        Handles the conversation loop, tool calls, and invoking other agents via send_message.

        Args:
            message: The input message.
            chat_id: Identifier for the overall interaction context.
            sender_name: Name of the calling agent, or None if called by user.
            current_depth: Current depth in the agent call chain.
            text_only: If True, return only the final text output.
            max_steps: Max iterations for this agent's internal loop.
            max_recursion_depth: Max depth allowed for agent-to-agent calls.
            **kwargs: Additional arguments.
        """
        if not self._thread_manager:
            raise RuntimeError("ThreadManager not configured for this agent.")

        # --- Recursion Depth Check ---
        if current_depth > max_recursion_depth:
            print(f"[Agent {self.name}] Error: Max recursion depth ({max_recursion_depth}) exceeded.")
            # Return an error RunResult or raise specific exception
            # Placeholder error result:
            error_output = f"Error: Max recursion depth ({max_recursion_depth}) exceeded."
            return error_output if text_only else RunResult(
                original_input=message,
                generated_items=[], # Or perhaps just the error item?
                model_responses=[],
                final_output=error_output,
                final_output_text=error_output,
                usage=None
            )

        # 1. Get the correct thread for this interaction
        thread = self._thread_manager.get_thread(chat_id, self.name, sender_name)

        # 2. Add the incoming message to the thread
        # TODO: Need to handle RunItem types correctly
        from agents.items import MessageInputItem # Assuming SDK structure
        # Role should be 'user' if sender_name is None, potentially 'agent' or tool otherwise?
        # For simplicity, maybe always treat incoming message as 'user' role within this thread?
        incoming_item = MessageInputItem(role='user', content=message)
        self._thread_manager.add_item_and_save(thread, incoming_item)

        # 3. Execution Loop
        current_step = 0
        final_output = None
        final_output_text = None
        all_items_in_run = [incoming_item] # Track items generated *during this run*

        while current_step < max_steps:
            current_step += 1

            # --- Call LLM ---
            history_for_model = thread.get_history_for_model()
            tools_for_model = await self.get_all_tools()
            # TODO: Prepare other model parameters (system prompt, model settings...)

            # This part needs adaptation from SDK's _run_impl._get_new_response
            # Needs access to a ModelProvider instance
            print(f"[Agent {self.name}] Calling LLM (Step {current_step}). History length: {len(history_for_model)}")
            # model_response = await self._model_provider.get_response(history_for_model, tools=tools_for_model, ...)
            # Placeholder response:
            from agents.items import MessageOutputItem, ToolCallItem # SDK types
            import json
            # Simulate LLM responding with text and maybe a tool call
            llm_output_items = [MessageOutputItem(content=f"Response from {self.name} step {current_step}.")]
            # --- SIMULATE send_message ---
            if current_step == 1 and self._agency_chart_peers and 'AgentB' in self._agency_chart_peers:
                 print(f"[Agent {self.name}] Simulating LLM calling send_message to AgentB")
                 send_call = ToolCallItem(
                     id=f"call_{current_step}",
                     name="send_message",
                     arguments=json.dumps({"recipient": "AgentB", "message": f"Message from {self.name}"})
                 )
                 llm_output_items.append(send_call)
            # --- End Simulate ---

            self._thread_manager.add_items_and_save(thread, llm_output_items)
            all_items_in_run.extend(llm_output_items)

            # --- Process LLM Response ---
            # Logic adapted from _run_impl.process_model_response & execute_tools...
            send_message_call_details = None
            standard_tool_calls = []
            has_text_output = any(isinstance(item, MessageOutputItem) for item in llm_output_items)

            for item in llm_output_items:
                if isinstance(item, ToolCallItem):
                    if item.name == "send_message":
                        # Prioritize send_message
                        send_message_call_details = item
                        break # Handle send_message first
                    else:
                        standard_tool_calls.append(item)

            # --- Handle send_message ---
            if send_message_call_details:
                recipient_name = json.loads(send_message_call_details.arguments).get('recipient')
                message_to_send = json.loads(send_message_call_details.arguments).get('message')

                is_allowed = self._agency_chart_peers and recipient_name in self._agency_chart_peers
                if is_allowed and recipient_name != self.name:
                    print(f"[Agent {self.name}] Calling Agent {recipient_name}")
                    # Find recipient agent instance (needs access to other agents, maybe via Agency ref?)
                    if self._agency_instance and hasattr(self._agency_instance, 'agents'):
                         recipient_agent = self._agency_instance.agents.get(recipient_name)
                         if recipient_agent:
                            # Check recursion depth *before* making the call
                            if current_depth + 1 > max_recursion_depth:
                                print(f"[Agent {self.name}] Error: Max recursion depth ({max_recursion_depth}) would be exceeded by calling {recipient_name}.")
                                # TODO: Add specific error ToolCallOutputItem
                                continue # Skip this tool call and proceed

                            # Recursive call!
                            sub_result = await recipient_agent.get_response(
                                message=message_to_send,
                                chat_id=chat_id,
                                sender_name=self.name, # Pass self as sender
                                current_depth=current_depth + 1, # Increment depth
                                text_only=True # Or get full RunResult?
                            )
                            # Format result as ToolCallOutputItem
                            from agents.items import ToolCallOutputItem
                            output_item = ToolCallOutputItem(tool_call_id=send_message_call_details.id, output=str(sub_result))
                            self._thread_manager.add_item_and_save(thread, output_item)
                            all_items_in_run.append(output_item)
                            continue # Continue *this* agent's loop (call LLM again with tool result)
                         else:
                            print(f"[Agent {self.name}] Error: Recipient agent '{recipient_name}' not found.")
                            # TODO: Add error ToolCallOutputItem
                    else:
                         print(f"[Agent {self.name}] Error: Cannot access recipient agent instance.")
                         # TODO: Add error ToolCallOutputItem
                else:
                    print(f"[Agent {self.name}] Error: Communication to '{recipient_name}' not allowed or is self.")
                    # TODO: Add error ToolCallOutputItem
                # If send_message failed, fall through to check standard tools/final output

            # --- Handle Standard Tools ---
            if standard_tool_calls:
                # TODO: Implement standard tool execution logic (like in _run_impl.execute_function_tool_calls)
                # Execute tools, get ToolCallOutputItems
                # tool_output_items = await self._execute_standard_tools(standard_tool_calls)
                # self._thread_manager.add_items_and_save(thread, tool_output_items)
                # all_items_in_run.extend(tool_output_items)
                # continue # Continue loop to process tool outputs
                print(f"[Agent {self.name}] Standard tool calls detected but execution not implemented yet.")
                pass # Placeholder

            # --- Check for Final Output ---
            # Determine if this turn is final (e.g., text output without pending tool calls)
            # This logic needs refinement based on SDK's NextStep determination.
            is_final = has_text_output and not standard_tool_calls and not send_message_call_details # Simplistic check
            if is_final:
                 final_output_item = next((item for item in reversed(llm_output_items) if isinstance(item, MessageOutputItem)), None)
                 final_output = final_output_item.content if final_output_item else "" # Or structured output
                 final_output_text = str(final_output)
                 print(f"[Agent {self.name}] Reached final output.")
                 break # Exit loop

            # If not final and no tools/send_message handled, maybe loop again? (Depends on SDK logic)
            print(f"[Agent {self.name}] Continuing loop (Step {current_step})")

        # --- End Loop ---
        if final_output is None:
            # Handle max_steps exceeded or other error
            print(f"[Agent {self.name}] Warning: Max steps reached or no final output.")
            final_output = "Error: Max steps reached." # Placeholder error
            final_output_text = final_output

        # --- Response Validation ---
        if self.response_validator and not self.response_validator(final_output_text):
            # Handle validation failure
            print(f"[Agent {self.name}] Response validation failed.")
            final_output = "Error: Response validation failed." # Placeholder error
            final_output_text = final_output

        # --- Construct RunResult ---
        # Need to collect all relevant RunItems for this specific invocation/flow
        # TODO: Refine how 'generated_items' for the final RunResult are collected across recursive calls.
        # Using 'all_items_in_run' is only for *this* agent's part of the flow.
        final_run_result = RunResult(
            original_input=message,
            generated_items=thread.get_full_log(), # This includes history before this run!
            model_responses=[], # TODO: Collect actual model responses
            final_output=final_output,
            final_output_text=final_output_text,
            usage=None # TODO: Collect usage info
        )

        return final_run_result.final_output_text if text_only else final_run_result

    async def get_response_stream(
        self,
        message: str,
        chat_id: str,
        sender_name: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Any, None]: # Yield streaming chunks
        """Gets a streaming response. Handles orchestration internally."""
        # TODO: Implement streaming version of the get_response logic.
        # This requires adapting the loop to yield RunItems as they are generated/processed.
        # Streaming recursive calls needs careful handling.
        print("[Agent.get_response_stream] Streaming not implemented yet.")
        yield "Placeholder streaming chunk"

    # --- Deprecated/Removed Methods ---
    # Document removal of methods tied to Assistants API
```

## 3. Key Design Decisions & Considerations:

*   **Agent-Centric Execution:** `Agent.get_response` is the core method containing the execution loop and orchestration logic.
*   **Recursion for Communication:** Agent-to-agent communication happens via direct, recursive calls to `recipient_agent.get_response`.
*   **Thread Management:** Relies on an external `ThreadManager` (provided during init) to get the correct, isolated `ConversationThread` based on `chat_id` and participants.
*   **`send_message` Interception:** The agent's internal loop detects `send_message` tool calls and triggers the recursive call logic instead of executing a dummy function.
*   **`Agency` Role:** The `Agency` is primarily for setup (injecting `send_message` tool based on `agency_chart`, configuring agents with `ThreadManager` and peer lists) and backward compatibility.
*   **A2A Alignment:** This design makes the `Agent` a more independent unit, suitable for potential A2A integration.
*   **Complexity:** The orchestration logic now resides within the `Agent` class, making it more complex than the base SDK agent.
*   **Dependencies:** Requires the `ThreadManager` and `ConversationThread` designs.

## 4. Open Questions/Refinements:

*   How exactly does the Agent get references to other agents to call their `get_response` method? (Via `_agency_instance.agents` seems feasible but needs confirmation).
*   Detailed implementation of standard tool execution within the `get_response` loop.
*   Robust implementation of the streaming version (`get_response_stream`).
*   Refining the collection of `generated_items` for the final `RunResult` across recursive calls.
*   Access to `ModelProvider` for making LLM calls within `get_response`.
*   Precise handling of `RunItem` roles when adding messages to threads.

This revised Agent design centralizes the execution logic, supports the required multi-agent recursive pattern with isolated threads, and aligns better with the goal of creating independent, potentially A2A-compatible agents.
