from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Type, Union

from agents import (
    Agent as BaseAgent,
    RunResult,
    TContext,
    Tool,
)
from agents.tool import FunctionTool


class ToolFactoryPlaceholder:
    """Placeholder for ToolFactory logic."""

    @staticmethod
    def from_callable(callable_obj: Callable) -> List[FunctionTool]:
        # TODO: Implement actual conversion logic
        # Should inspect callable, generate schema, create FunctionTool
        # Needs to handle @function_tool decorated functions gracefully
        print(f"[ToolFactoryPlaceholder] Converting callable: {callable_obj.__name__}")
        # Dummy implementation - returns empty list
        # Check if it's already a FunctionTool implicitly via decorator?
        if hasattr(callable_obj, "_openai_function_tool"):
            print(f"Callable {callable_obj.__name__} seems to be an SDK @function_tool")
            # How to get the FunctionTool instance from the decorated func?
            # This might require changes in how @function_tool works or access
            # to internal registry if it exists.
            # Returning empty for now.
            return []
        return []

    @staticmethod
    def from_schema(schema_obj: Type) -> List[FunctionTool]:
        # TODO: Implement actual conversion logic from Pydantic model, etc.
        print(f"[ToolFactoryPlaceholder] Converting schema: {schema_obj.__name__}")
        return []

    @staticmethod
    def load_tools_from_folder(folder_path: Union[str, Path]) -> List[FunctionTool]:
        print(f"[ToolFactoryPlaceholder] Loading tools from folder: {folder_path}")
        loaded_tools = []
        if not os.path.isdir(folder_path):
            print(f"Warning: Tools folder not found: {folder_path}")
            return []

        for filename in os.listdir(folder_path):
            if filename.endswith(".py") and not filename.startswith("_"):
                filepath = os.path.join(folder_path, filename)
                # TODO: Implement robust loading and inspection of python modules
                # This is complex: needs to import module safely, find relevant
                # classes/functions/models, and convert them using from_callable/from_schema.
                print(f"  - Found potential tool file: {filename} (Loading logic TBD)")
                # Example conceptual call:
                # module_tools = ToolFactoryPlaceholder.load_from_module(filepath)
                # loaded_tools.extend(module_tools)
                pass
        return loaded_tools


# Use the placeholder for now
ToolFactory = ToolFactoryPlaceholder

if TYPE_CHECKING:
    # Import Agency Swarm specific concepts/helpers (placeholders)
    from .thread import ConversationThread
    # from .tools import ToolFactory # Assuming ToolFactory is adapted/kept

# TODO: Define which original Agency Swarm params are kept explicitly
# This allows for better type checking and clarity
AGENCY_SWARM_PARAMS = ["files_folder", "tools_folder", "description", "response_validator"]


class Agent(BaseAgent[TContext]):
    """
    Extends the OpenAI Agents SDK's Agent class (agents.Agent)
    with features and conventions from the Agency Swarm framework.

    This agent is designed to work within an Agency orchestrator.
    """

    # --- Agency Swarm Specific Parameters ---
    # Define parameters specific to Agency Swarm or those we want to handle differently
    files_folder: Optional[Union[str, Path]]
    tools_folder: Optional[Union[str, Path]]
    description: Optional[str]  # Potentially different from handoff_description?
    response_validator: Optional[Callable[[str], bool]]
    # Note: Add other parameters identified during design/migration

    def __init__(self, **kwargs: Any):
        """
        Initializes the Agency Swarm Agent.

        Args:
            **kwargs: Accepts parameters for both the base OpenAI Agent
                      and Agency Swarm specific features.
        """

        # --- Separate Kwargs for BaseAgent and AgencySwarm Agent ---
        base_agent_params = {}
        agency_swarm_params = {}

        # Get annotations from BaseAgent to identify its parameters
        # Using inspect as BaseAgent.__annotations__ might not capture all inherited params correctly
        try:
            base_sig = inspect.signature(BaseAgent)
            base_param_names = set(base_sig.parameters.keys())
        except (
            ValueError
        ):  # Might happen if BaseAgent is not a standard class/has complex metaclass
            # Fallback or default set - adjust as needed
            base_param_names = {
                "name",
                "instructions",
                "handoff_description",
                "handoffs",
                "model",
                "model_settings",
                "tools",
                "mcp_servers",
                "mcp_config",
                "input_guardrails",
                "output_guardrails",
                "output_type",
                "hooks",
                "tool_use_behavior",
                "reset_tool_choice",
            }

        for key, value in kwargs.items():
            if key in base_param_names:
                base_agent_params[key] = value
            # Check against explicitly defined Agency Swarm params for clarity
            elif key in AGENCY_SWARM_PARAMS:
                agency_swarm_params[key] = value
            else:
                # Decide how to handle unknown parameters: raise error or ignore?
                # For flexibility, maybe ignore for now, or log a warning.
                print(f"Warning: Unknown parameter '{key}' passed to Agent constructor.")

        # --- Initialize BaseAgent ---
        if "name" not in base_agent_params:
            raise ValueError("Agent requires a 'name' parameter.")
        # Ensure 'tools' list exists in base_agent_params before super().__init__
        if "tools" not in base_agent_params:
            base_agent_params["tools"] = []  # Initialize if not provided
        elif not isinstance(base_agent_params["tools"], list):
            raise TypeError("'tools' parameter must be a list.")

        super().__init__(**base_agent_params)

        # --- Initialize Agency Swarm Specific Attributes ---
        # Need to access self.tools *after* super().__init__() has run
        self.files_folder = agency_swarm_params.get("files_folder")
        self.tools_folder = agency_swarm_params.get("tools_folder")
        self.description = agency_swarm_params.get("description")
        self.response_validator = agency_swarm_params.get("response_validator")
        # ... initialize others ...

        # --- Initialization Logic (Now includes loading tools) ---
        self._load_tools_from_folder()  # Load tools from folder now
        # TODO (Task 9): self._init_file_handling()

    # --- Tool Handling ---

    def add_tool(self, tool_input: Union[Callable, FunctionTool, Type]) -> None:
        """Adds a tool to the agent's list of tools.

        Handles SDK FunctionTools directly and attempts to convert
        callables or type schemas using the ToolFactory.
        """
        added = False
        if isinstance(tool_input, FunctionTool):
            if tool_input not in self.tools:
                self.tools.append(tool_input)
                added = True
        elif callable(tool_input):
            # Attempt conversion using ToolFactory
            converted_tools = ToolFactory.from_callable(tool_input)
            for tool in converted_tools:
                if tool not in self.tools:
                    self.tools.append(tool)
                    added = True
            # Special check needed here: if from_callable returned empty because
            # it detected an @function_tool, how do we add the actual tool?
            # This requires ToolFactory/decorator interaction to be resolved.
            if not converted_tools and not added:
                print(
                    f"Warning: Callable '{tool_input.__name__}' could not be converted by ToolFactory (or might be an unretrievable @function_tool)."
                )
        elif isinstance(tool_input, type):
            # Attempt conversion from schema using ToolFactory
            converted_tools = ToolFactory.from_schema(tool_input)
            for tool in converted_tools:
                if tool not in self.tools:
                    self.tools.append(tool)
                    added = True
            if not converted_tools and not added:
                print(
                    f"Warning: Schema type '{tool_input.__name__}' could not be converted by ToolFactory."
                )
        else:
            # Check if it's any other valid SDK Tool type before raising error
            if isinstance(tool_input, Tool):
                if tool_input not in self.tools:
                    self.tools.append(tool_input)  # Add other Tool types like FileSearch, etc.
                    added = True
            else:
                raise ValueError(
                    f"Unsupported tool input type: {type(tool_input)}. Expected FunctionTool, callable, schema, or other agents.Tool."
                )

        # if added:
        #     print(f"Tool '{getattr(tool_input, '__name__', str(tool_input))}' added.") # Debug log

    def _load_tools_from_folder(self) -> None:
        """Loads tools from the specified tools_folder using ToolFactory."""
        if self.tools_folder:
            try:
                folder_path = Path(self.tools_folder).resolve()
                print(f"Loading tools from resolved path: {folder_path}")
                loaded_tools = ToolFactory.load_tools_from_folder(folder_path)
                for tool in loaded_tools:
                    if tool not in self.tools:
                        self.tools.append(tool)
                if loaded_tools:
                    print(f"Loaded {len(loaded_tools)} tool(s) from {folder_path}")
            except Exception as e:
                print(f"Error loading tools from folder {self.tools_folder}: {e}")
        # else:
        #     print("No tools_folder specified.")

    # Override __init__ to call _load_tools_from_folder
    def __init__(self, **kwargs: Any):
        """
        Initializes the Agency Swarm Agent.

        Args:
            **kwargs: Accepts parameters for both the base OpenAI Agent
                      and Agency Swarm specific features.
        """

        # --- Separate Kwargs for BaseAgent and AgencySwarm Agent ---
        base_agent_params = {}
        agency_swarm_params = {}

        # Get annotations from BaseAgent to identify its parameters
        # Using inspect as BaseAgent.__annotations__ might not capture all inherited params correctly
        try:
            base_sig = inspect.signature(BaseAgent)
            base_param_names = set(base_sig.parameters.keys())
        except (
            ValueError
        ):  # Might happen if BaseAgent is not a standard class/has complex metaclass
            # Fallback or default set - adjust as needed
            base_param_names = {
                "name",
                "instructions",
                "handoff_description",
                "handoffs",
                "model",
                "model_settings",
                "tools",
                "mcp_servers",
                "mcp_config",
                "input_guardrails",
                "output_guardrails",
                "output_type",
                "hooks",
                "tool_use_behavior",
                "reset_tool_choice",
            }

        for key, value in kwargs.items():
            if key in base_param_names:
                base_agent_params[key] = value
            # Check against explicitly defined Agency Swarm params for clarity
            elif key in AGENCY_SWARM_PARAMS:
                agency_swarm_params[key] = value
            else:
                # Decide how to handle unknown parameters: raise error or ignore?
                # For flexibility, maybe ignore for now, or log a warning.
                print(f"Warning: Unknown parameter '{key}' passed to Agent constructor.")

        # --- Initialize BaseAgent ---
        if "name" not in base_agent_params:
            raise ValueError("Agent requires a 'name' parameter.")
        # Ensure 'tools' list exists in base_agent_params before super().__init__
        if "tools" not in base_agent_params:
            base_agent_params["tools"] = []  # Initialize if not provided
        elif not isinstance(base_agent_params["tools"], list):
            raise TypeError("'tools' parameter must be a list.")

        super().__init__(**base_agent_params)

        # --- Initialize Agency Swarm Specific Attributes ---
        # Need to access self.tools *after* super().__init__() has run
        self.files_folder = agency_swarm_params.get("files_folder")
        self.tools_folder = agency_swarm_params.get("tools_folder")
        self.description = agency_swarm_params.get("description")
        self.response_validator = agency_swarm_params.get("response_validator")
        # ... initialize others ...

        # --- Initialization Logic (Now includes loading tools) ---
        self._load_tools_from_folder()  # Load tools from folder now
        # TODO (Task 9): self._init_file_handling()

    def _init_file_handling(self) -> None:
        """Initializes file handling logic, checks folder naming for VectorStore ID."""
        # TODO (Task 9): Implement logic
        print("[_init_file_handling] Placeholder")
        pass

    def upload_file(self, file_path: str) -> str:  # Return File ID
        """Uploads a file, associates with agent's VS (if exists), manages naming."""
        # TODO (Task 9): Implement logic
        print("[upload_file] Placeholder")
        return "file_dummy_id"

    def check_file_exists(self, file_path: str) -> Optional[str]:  # Return File ID if found
        """Checks if a file seems to be already uploaded based on naming convention."""
        # TODO (Task 9): Implement logic
        print("[check_file_exists] Placeholder")
        return None

    # --- Response Handling ---
    # These methods provide a way to run this agent standalone, outside of an Agency.
    # For orchestrated multi-agent interactions, use Agency.run_interaction methods.

    async def get_response(
        self,
        message: str,
        message_files: Optional[List[str]] = None,  # Added for potential file uploads with message
        run_config: Optional[Dict[str, Any]] = None,  # Allow passing RunConfig options
        text_only: bool = False,
        **kwargs: Any,  # Pass-through for underlying Runner
    ) -> Union[str, RunResult]:
        """Gets a response for a given message in a *new*, standalone run for this agent.

           Uses the underlying OpenAI Agents SDK Runner.
           For orchestrated multi-agent conversations, use Agency.run_interaction.

        Args:
            message: The user message input.
            message_files: Optional list of file paths to upload and associate with the message.
                           (Requires file handling Task 9 to be implemented).
            run_config: Optional dictionary to configure the run (maps to RunConfig object).
            text_only: If True, return only the final text output, otherwise return the full RunResult.
            **kwargs: Additional arguments passed directly to `Runner.run`.

        Returns:
            The final text output (if text_only=True) or the full RunResult object.
        """
        print(
            f"Warning: Agent.get_response invoked directly on '{self.name}'. \n              Starting a new standalone run using the SDK Runner. \n              For orchestrated multi-agent chat, use Agency.run_interaction."
        )

        # TODO (Task 9): Implement message_files handling: upload files, get IDs, format input
        if message_files:
            print(
                f"Warning: message_files parameter requires Task 9 (File Handling) implementation."
            )
            # Conceptual:
            # uploaded_file_ids = [self.upload_file(fp) for fp in message_files]
            # input_items = [MessageInputItem(role='user', content=message, file_ids=uploaded_file_ids)]
            # input_arg = input_items
            input_arg = message  # Fallback for now
        else:
            input_arg = message

        # Construct RunConfig if provided
        from agents.run import RunConfig  # Local import

        run_config_obj = RunConfig(**run_config) if run_config else None

        from agents.run import Runner  # Local import

        try:
            result = await Runner.run(
                starting_agent=self, input=input_arg, run_config=run_config_obj, **kwargs
            )

            # --- Response Validation (Conceptual - applied after run) ---
            # It might be better to handle this via Runner hooks if possible,
            # but applying it here if `_validate_response` exists.
            if hasattr(self, "_validate_response") and result.final_output_text:
                if not self._validate_response(result.final_output_text):
                    # TODO: Define specific exception for validation failure
                    raise ValueError(f"Response validation failed for agent {self.name}")

            return result.final_output_text if text_only else result
        except Exception as e:
            # TODO: Refine error handling/logging
            print(f"Error during Agent.get_response for {self.name}: {e}")
            raise e

    async def get_response_stream(
        self,
        message: str,
        message_files: Optional[List[str]] = None,
        run_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Gets a streaming response in a *new*, standalone run for this agent.
           Uses the underlying OpenAI Agents SDK Runner.
           For orchestrated multi-agent conversations, use Agency.run_interaction_streamed.

        Args:
            message: The user message input.
            message_files: Optional list of file paths to upload and associate with the message.
                           (Requires file handling Task 9 to be implemented).
            run_config: Optional dictionary to configure the run (maps to RunConfig object).
            **kwargs: Additional arguments passed directly to `Runner.run_streamed`.

        Yields:
            Streaming chunks from the Runner.
        """
        print(
            f"Warning: Agent.get_response_stream invoked directly on '{self.name}'. \n              Starting a new standalone streaming run using the SDK Runner. \n              For orchestrated multi-agent chat, use Agency.run_interaction_streamed."
        )

        # TODO (Task 9): Implement message_files handling
        if message_files:
            print(
                f"Warning: message_files parameter requires Task 9 (File Handling) implementation."
            )
            input_arg = message  # Fallback for now
        else:
            input_arg = message

        # Construct RunConfig if provided
        from agents.run import RunConfig  # Local import

        run_config_obj = RunConfig(**run_config) if run_config else None

        from agents.run import Runner

        try:
            # TODO: Response validation for streaming? This is tricky.
            # Validation typically happens on the complete response.
            # Maybe validate the final assembled text if possible, or rely on output guardrails?
            async for chunk in Runner.run_streamed(
                starting_agent=self, input=input_arg, run_config=run_config_obj, **kwargs
            ):
                yield chunk
        except Exception as e:
            # TODO: Refine error handling/logging
            print(f"Error during Agent.get_response_stream for {self.name}: {e}")
            raise e

    # --- Response Validation Hook (Conceptual) ---
    def _validate_response(self, response_text: str) -> bool:
        """Internal helper to apply response validator if configured."""
        # This method itself doesn't change, but its invocation point might
        # primarily be within the Agency orchestrator for multi-agent flows.
        if self.response_validator:
            try:
                is_valid = self.response_validator(response_text)
                if not is_valid:
                    print(f"Response validation failed for agent {self.name}")  # Log failure
                return is_valid
            except Exception as e:
                print(f"Error during response validation for agent {self.name}: {e}")
                return False  # Treat validation errors as failure
        return True
