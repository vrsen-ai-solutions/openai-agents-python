import inspect
import json
import logging
import re
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from openai import AsyncOpenAI, NotFoundError

from agents import (
    Agent as BaseAgent,
    FileSearchTool,
    FunctionTool,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    RunItem,
    Runner,
    RunResult,
    Tool,
    TResponseInputItem,
)
from agents.exceptions import AgentsException
from agents.items import (
    ItemHelpers,
)
from agents.run import DEFAULT_MAX_TURNS
from agents.stream_events import RunItemStreamEvent

from .context import MasterContext
from .thread import ThreadManager

logger = logging.getLogger(__name__)

# --- Constants / Types ---
# Combine old and new params for easier checking later
AGENT_PARAMS = {
    # New/Current
    "files_folder",
    "tools_folder",
    "description",
    "response_validator",
    # Old/Deprecated (to check in kwargs)
    "id",
    "tool_resources",
    "schemas_folder",
    "api_headers",
    "api_params",
    "file_ids",
    "reasoning_effort",
    "validation_attempts",
    "examples",
    "file_search",
    "refresh_from_id",
    "mcp_servers",
}

# --- Constants for dynamic tool creation ---
SEND_MESSAGE_TOOL_PREFIX = "send_message_to_"
MESSAGE_PARAM = "message"


class Agent(BaseAgent[MasterContext]):  # Context type is MasterContext
    """
    Agency Swarm Agent: Extends the base SDK Agent with capabilities for
    multi-agent collaboration within an Agency.

    Handles subagent registration, file management (optionally linked to Vector Stores),
    and delegates core execution logic to the SDK's Runner.
    """

    # --- Agency Swarm Specific Parameters ---
    files_folder: Optional[Union[str, Path]]
    tools_folder: Optional[Union[str, Path]]  # Placeholder for future ToolFactory
    description: Optional[str]
    response_validator: Optional[Callable[[str], bool]]

    # --- Internal State ---
    _thread_manager: Optional[ThreadManager] = None
    _agency_instance: Optional[Any] = None  # Holds reference to parent Agency
    _associated_vector_store_id: Optional[str] = None
    files_folder_path: Optional[Path] = None
    _subagents: Dict[str, "Agent"]

    def __init__(self, **kwargs: Any):
        """
        Initializes the Agency Swarm Agent.

        Handles backward compatibility with deprecated parameters.
        Separates kwargs for BaseAgent and Agency Swarm specific logic.
        Initializes file handling and subagent dictionary.
        """
        # --- Handle Deprecated Args ---
        deprecated_args_used = {}
        if "id" in kwargs:
            warnings.warn(
                "'id' parameter (OpenAI Assistant ID) is deprecated and no longer used for loading. Agent state is managed via PersistenceHooks.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["id"] = kwargs.pop("id")
        if "tool_resources" in kwargs:
            warnings.warn(
                "'tool_resources' is deprecated. File resources should be managed via 'files_folder' and the 'upload_file' method for Vector Stores.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["tool_resources"] = kwargs.pop("tool_resources")
        if "schemas_folder" in kwargs or "api_headers" in kwargs or "api_params" in kwargs:
            warnings.warn(
                "'schemas_folder', 'api_headers', and 'api_params' related to OpenAPI tools are deprecated. Use standard FunctionTools instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["schemas_folder"] = kwargs.pop("schemas_folder", None)
            deprecated_args_used["api_headers"] = kwargs.pop("api_headers", None)
            deprecated_args_used["api_params"] = kwargs.pop("api_params", None)
        if "file_ids" in kwargs:
            warnings.warn(
                "'file_ids' is deprecated. Use 'files_folder' to associate with Vector Stores or manage files via Agent methods.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["file_ids"] = kwargs.pop("file_ids")
        if "reasoning_effort" in kwargs:
            warnings.warn(
                "'reasoning_effort' is deprecated as a direct Agent parameter. Configure model settings via 'model_settings' if needed.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["reasoning_effort"] = kwargs.pop("reasoning_effort")
        if "validation_attempts" in kwargs:
            val_attempts = kwargs.pop("validation_attempts")
            warnings.warn(
                "'validation_attempts' is deprecated. Use the 'response_validator' callback for validation logic.",
                DeprecationWarning,
                stacklevel=2,
            )
            if val_attempts > 1 and "response_validator" not in kwargs:
                warnings.warn(
                    "Using 'validation_attempts > 1' without a 'response_validator' has no effect. Implement validation logic in the callback.",
                    UserWarning,  # Changed to UserWarning as it's about usage logic
                    stacklevel=2,
                )
            deprecated_args_used["validation_attempts"] = val_attempts
        if "examples" in kwargs:
            examples = kwargs.pop("examples")
            warnings.warn(
                "'examples' parameter is deprecated. Consider incorporating examples directly into the agent's 'instructions'.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Attempt to prepend examples to instructions
            if examples and isinstance(examples, list):
                try:
                    # Basic formatting, might need refinement
                    examples_str = "\\n\\nExamples:\\n" + "\\n".join(f"- {json.dumps(ex)}" for ex in examples)
                    current_instructions = kwargs.get("instructions", "")
                    kwargs["instructions"] = current_instructions + examples_str
                    logger.info("Prepended 'examples' content to agent instructions.")
                except Exception as e:
                    logger.warning(f"Could not automatically prepend 'examples' to instructions: {e}")
            deprecated_args_used["examples"] = examples  # Store original for logging if needed
        if "file_search" in kwargs:
            warnings.warn(
                "'file_search' parameter is deprecated. FileSearchTool is added automatically if 'files_folder' indicates a Vector Store.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["file_search"] = kwargs.pop("file_search")
        if "refresh_from_id" in kwargs:
            warnings.warn(
                "'refresh_from_id' is deprecated as loading by Assistant ID is no longer supported.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["refresh_from_id"] = kwargs.pop("refresh_from_id")
        if "mcp_servers" in kwargs:
            warnings.warn(
                "'mcp_servers' is deprecated and no longer supported in this SDK version.",
                DeprecationWarning,
                stacklevel=2,
            )
            deprecated_args_used["mcp_servers"] = kwargs.pop("mcp_servers")

        # Log if any deprecated args were used
        if deprecated_args_used:
            logger.warning(f"Deprecated Agent parameters used: {list(deprecated_args_used.keys())}")

        # --- Separate Kwargs (Existing Logic) ---
        base_agent_params = {}
        current_agent_params = {}
        # --- SDK Imports --- (Move BaseAgent signature check here)
        try:
            base_sig = inspect.signature(BaseAgent)
            base_param_names = set(base_sig.parameters.keys())
        except ValueError:
            # Fallback if signature inspection fails
            base_param_names = {
                "name",
                "instructions",
                "handoff_description",
                "handoffs",
                "model",
                "model_settings",
                "tools",
                "mcp_servers",
                "mcp_config",  # mcp params are part of BaseAgent, though maybe unused now
                "input_guardrails",
                "output_guardrails",
                "output_type",
                "hooks",
                "tool_use_behavior",
                "reset_tool_choice",
            }

        # Iterate through remaining kwargs after popping deprecated ones
        for key, value in kwargs.items():
            if key in base_param_names:
                base_agent_params[key] = value
            # Use the new combined set AGENT_PARAMS for current/swarm-specific params
            elif key in {"files_folder", "tools_folder", "response_validator", "description"}:
                current_agent_params[key] = value
            else:
                # Only warn if it wasn't a handled deprecated arg
                if key not in deprecated_args_used:
                    logger.warning(f"Unknown parameter '{key}' passed to Agent constructor.")

        # --- BaseAgent Init ---
        if "name" not in base_agent_params:
            raise ValueError("Agent requires a 'name' parameter.")
        if "tools" not in base_agent_params:
            base_agent_params["tools"] = []
        elif not isinstance(base_agent_params["tools"], list):
            raise TypeError("'tools' parameter must be a list.")
        # Remove description from base_agent_params if it was added
        base_agent_params.pop("description", None)
        super().__init__(**base_agent_params)

        # --- Agency Swarm Attrs Init --- (Assign AFTER super)
        self.files_folder = current_agent_params.get("files_folder")
        self.tools_folder = current_agent_params.get("tools_folder")
        self.response_validator = current_agent_params.get("response_validator")
        # Set description directly from current_agent_params, default to None if not provided
        self.description = current_agent_params.get("description")

        # --- Internal State Init ---
        self._subagents = {}
        # _thread_manager and _agency_instance are injected by Agency

        # --- Setup ---
        self._load_tools_from_folder()  # Placeholder call
        self._init_file_handling()

    # --- Properties --- (Example: OpenAI Client)
    @property
    def client(self) -> AsyncOpenAI:
        """Provides access to an initialized AsyncOpenAI client."""
        # Consider making client management more robust if needed
        if not hasattr(self, "_openai_client"):
            self._openai_client = AsyncOpenAI()
        return self._openai_client

    # --- Tool Management ---
    def add_tool(self, tool: Tool) -> None:
        """Adds a tool instance to the agent."""
        # Simplified: Assumes tool is already a valid Tool instance
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected an instance of agents.Tool, got {type(tool)}")

        # Check for existing tool with the same name before adding
        if any(getattr(t, "name", None) == getattr(tool, "name", None) for t in self.tools):
            logger.warning(
                f"Tool with name '{getattr(tool, 'name', '(unknown)')}' already exists for agent '{self.name}'. Skipping."
            )
            return

        self.tools.append(tool)
        logger.debug(f"Tool '{getattr(tool, 'name', '(unknown)')}' added to agent '{self.name}'")

    def _load_tools_from_folder(self) -> None:
        """Placeholder: Loads tools from tools_folder (future Task)."""
        if self.tools_folder:
            logger.warning("Tool loading from folder is not fully implemented yet.")
            # Placeholder logic using ToolFactoryPlaceholder (replace when implemented)
            # try:
            #     folder_path = Path(self.tools_folder).resolve()
            #     loaded_tools = ToolFactory.load_tools_from_folder(folder_path)
            #     for tool in loaded_tools:
            #         self.add_tool(tool)
            # except Exception as e:
            #     logger.error(f"Error loading tools from folder {self.tools_folder}: {e}")

    # --- Subagent Management ---
    def register_subagent(self, recipient_agent: "Agent") -> None:
        """Registers another agent as a subagent and dynamically creates a specific
        SendMessage tool for communication.
        """
        if not isinstance(recipient_agent, Agent):
            raise TypeError(
                f"Expected an instance of Agent, got {type(recipient_agent)}. Ensure agents are initialized before registration."
            )
        if not hasattr(recipient_agent, "name") or not isinstance(recipient_agent.name, str):
            raise TypeError("Subagent must be an Agent instance with a valid name.")

        recipient_name = recipient_agent.name
        if recipient_name == self.name:
            raise ValueError("Agent cannot register itself as a subagent.")

        # Initialize _subagents if it doesn't exist
        if not hasattr(self, "_subagents") or self._subagents is None:
            self._subagents = {}

        if recipient_name in self._subagents:
            logger.warning(
                f"Agent '{recipient_name}' is already registered as a subagent for '{self.name}'. Skipping tool creation."
            )
            return

        self._subagents[recipient_name] = recipient_agent
        logger.info(f"Agent '{self.name}' registered subagent: '{recipient_name}'")

        # --- Dynamically create the specific send_message tool --- #

        tool_name = f"{SEND_MESSAGE_TOOL_PREFIX}{recipient_name}"
        recipient_description = getattr(recipient_agent, "description", "No description provided")
        tool_description = (
            f"Send a message to the {recipient_name} agent. " f"This agent's role is: {recipient_description}"
        )

        # Define the schema for the tool's single parameter: 'message'
        params_schema = {
            "type": "object",
            "properties": {
                MESSAGE_PARAM: {
                    "type": "string",
                    "description": f"The message content to send to the {recipient_name} agent.",
                }
            },
            "required": [MESSAGE_PARAM],
            "additionalProperties": False,  # Enforce only the message parameter
        }

        # Define the async function that will handle the tool invocation
        # This function captures `self` (the sender) and `recipient_agent`
        async def _invoke_send_message(wrapper: RunContextWrapper[MasterContext], args_str: str) -> str:
            master_context: MasterContext = wrapper.context

            # Parse the arguments string
            try:
                parsed_args = json.loads(args_str)
                message_content = parsed_args.get(MESSAGE_PARAM)
            except json.JSONDecodeError:
                logger.error(f"Tool '{tool_name}' invoked with invalid JSON arguments: {args_str}")
                return f"Error: Invalid arguments format for tool {tool_name}."
            except Exception as e:
                logger.error(
                    f"Error parsing arguments for tool '{tool_name}': {e}\nArguments: {args_str}", exc_info=True
                )
                return f"Error: Could not parse arguments for tool {tool_name}."

            if not message_content:
                logger.error(f"Tool '{tool_name}' invoked without '{MESSAGE_PARAM}' parameter in arguments: {args_str}")
                return f"Error: Missing required parameter '{MESSAGE_PARAM}' for tool {tool_name}."

            current_chat_id = master_context.chat_id
            if not current_chat_id:
                logger.error(f"Tool '{tool_name}' invoked without 'chat_id' in MasterContext.")
                return "Error: Internal context error. Missing chat_id for agent communication."

            sender_name = self.name  # Captured from outer scope

            logger.info(
                f"Agent '{sender_name}' invoking tool '{tool_name}'. "
                f"Recipient: '{recipient_name}', ChatID: {current_chat_id}, "
                f'Message: "{message_content[:50]}..."'
            )

            try:
                # Call the recipient agent's get_response method directly
                logger.debug(f"Calling target agent '{recipient_name}'.get_response...")
                # Store the RunResult object
                sub_run_result: RunResult = await recipient_agent.get_response(
                    message=message_content,
                    sender_name=sender_name,
                    chat_id=current_chat_id,
                    context_override=master_context.user_context,  # Pass only user context
                    # Hooks/Config are managed by the Runner called within get_response
                )

                # Extract the final text output from the RunResult
                final_output_text = sub_run_result.final_output or "(No text output from recipient)"
                # Ensure it's a string before logging/returning
                if not isinstance(final_output_text, str):
                    final_output_text = str(final_output_text)

                logger.info(
                    f"Received response via tool '{tool_name}' from '{recipient_name}': \"{final_output_text[:50]}...\""
                )
                # Return the final output text directly as a string
                return final_output_text

            except Exception as e:
                logger.error(
                    f"Error occurred during sub-call via tool '{tool_name}' from '{sender_name}' to '{recipient_name}': {e}",
                    exc_info=True,
                )
                # Return a formatted error string as the tool output
                # The Runner will package this into a ToolCallOutputItem
                error_message = f"Error: Failed to get response from agent '{recipient_name}'. Reason: {e}"
                # Optionally, raise the exception if we want the Runner to handle it more explicitly
                # raise agents.exceptions.ToolError(error_message) from e
                return error_message  # Return error string for now

        # Create the FunctionTool instance directly
        specific_tool = FunctionTool(
            name=tool_name,
            description=tool_description,
            params_json_schema=params_schema,
            on_invoke_tool=_invoke_send_message,  # Pass the nested function
        )

        # Add the specific tool to this agent's tools
        self.add_tool(specific_tool)
        logger.debug(f"Dynamically added tool '{tool_name}' to agent '{self.name}'.")

    # --- File Handling ---
    def _init_file_handling(self) -> None:
        """Initializes file handling: sets up local folder and VS ID if specified."""
        self._associated_vector_store_id = None
        self.files_folder_path = None
        if not self.files_folder:
            return

        try:
            self.files_folder_path = Path(self.files_folder).resolve()
            folder_name = self.files_folder_path.name
            match = re.search(r"_vs_([a-zA-Z0-9\-]+)$", folder_name)
            if match:
                vs_id = match.group(1)
                logger.info(f"Detected Vector Store ID '{vs_id}' in files_folder name.")
                self._associated_vector_store_id = vs_id
                base_folder_name = folder_name[: match.start()]
                # Construct the base path
                base_path = self.files_folder_path.parent / base_folder_name
                # Create the base directory
                base_path.mkdir(parents=True, exist_ok=True)
                # Assign the corrected base path
                self.files_folder_path = base_path
            else:
                # If no VS ID, just ensure the original path exists
                self.files_folder_path.mkdir(parents=True, exist_ok=True)

            # This log message should now show the correct path
            logger.info(f"Agent '{self.name}' local files folder: {self.files_folder_path}")

            if self._associated_vector_store_id:
                self._ensure_file_search_tool()
        except Exception as e:
            logger.error(
                f"Error initializing file handling for path '{self.files_folder}': {e}",
                exc_info=True,
            )
            self.files_folder_path = None  # Reset on error

    def _ensure_file_search_tool(self):
        """Adds or updates the FileSearchTool if a VS ID is associated."""
        if not self._associated_vector_store_id:
            return
        # Remove existing FileSearchTool(s) first to avoid duplicates/conflicts
        self.tools = [t for t in self.tools if not isinstance(t, FileSearchTool)]
        logger.info(f"Adding FileSearchTool for VS ID: {self._associated_vector_store_id}")
        self.add_tool(FileSearchTool(vector_store_ids=[self._associated_vector_store_id]))

    async def upload_file(self, file_path: str) -> str:
        """Uploads a file locally (if configured) and to OpenAI assistants purpose.
        Associates with VS if agent is configured for one.
        """
        source_path = Path(file_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        local_upload_path = source_path  # Default to source if no local copy needed

        # Copy locally if folder is set
        if self.files_folder_path:
            # Simple copy, overwrites allowed for simplicity now.
            # Could add UUID or checks if needed.
            local_destination = self.files_folder_path / source_path.name
            try:
                shutil.copy2(source_path, local_destination)
                logger.info(f"Copied file locally to {local_destination}")
                local_upload_path = local_destination
            except Exception as e:
                logger.error(f"Error copying file {source_path} locally: {e}", exc_info=True)
                # Continue with original path for OpenAI upload

        # Upload to OpenAI
        try:
            logger.info(f"Uploading file '{local_upload_path.name}' to OpenAI...")
            openai_file = await self.client.files.create(file=local_upload_path.open("rb"), purpose="assistants")
            logger.info(f"Uploaded to OpenAI. File ID: {openai_file.id}")

            # Associate with Vector Store if needed
            if self._associated_vector_store_id:
                try:
                    logger.info(f"Adding OpenAI file {openai_file.id} to VS {self._associated_vector_store_id}")
                    # Check if file already exists in VS (optional, API call)
                    # vs_files = await self.client.beta.vector_stores.files.list(vector_store_id=self._associated_vector_store_id, limit=100)
                    # if any(f.id == openai_file.id for f in vs_files.data):
                    #     logger.debug(f"File {openai_file.id} already in VS {self._associated_vector_store_id}.")
                    # else:
                    await self.client.beta.vector_stores.files.create(
                        vector_store_id=self._associated_vector_store_id, file_id=openai_file.id
                    )
                    logger.info(f"Added file {openai_file.id} to VS {self._associated_vector_store_id}.")
                    self._ensure_file_search_tool()  # Ensure tool is present after adding first file
                except NotFoundError:
                    logger.error(
                        f"Vector Store {self._associated_vector_store_id} not found when adding file {openai_file.id}."
                    )
                except Exception as e:
                    logger.error(
                        f"Error adding file {openai_file.id} to VS {self._associated_vector_store_id}: {e}",
                        exc_info=True,
                    )

            return openai_file.id

        except Exception as e:
            logger.error(f"Error uploading file {local_upload_path.name} to OpenAI: {e}", exc_info=True)
            raise AgentsException(f"Failed to upload file to OpenAI: {e}") from e

    async def check_file_exists(self, file_path: str) -> Optional[str]:
        """Checks if a file with the same name exists in the associated VS."""
        if not self._associated_vector_store_id:
            return None
        target_filename = Path(file_path).name
        try:
            logger.debug(f"Checking for file '{target_filename}' in VS {self._associated_vector_store_id}...")
            vs_files_page = await self.client.beta.vector_stores.files.list(
                vector_store_id=self._associated_vector_store_id, limit=100
            )
            for vs_file in vs_files_page.data:
                try:
                    file_object = await self.client.files.retrieve(vs_file.id)
                    if file_object.filename == target_filename:
                        logger.debug(f"Found matching file in VS: {vs_file.id}")
                        return vs_file.id
                except NotFoundError:
                    logger.warning(f"VS file {vs_file.id} not found in OpenAI files.")
                except Exception as e_inner:
                    logger.warning(f"Error retrieving details for file {vs_file.id}: {e_inner}")
            return None
        except NotFoundError:
            logger.error(f"Vector Store {self._associated_vector_store_id} not found during file check.")
            return None
        except Exception as e:
            logger.error(
                f"Error checking file existence in VS {self._associated_vector_store_id}: {e}",
                exc_info=True,
            )
            return None

    # --- Core Execution Methods ---
    async def get_response(
        self,
        message: Union[str, List[Dict[str, Any]]],  # Allow raw message or OpenAI format
        sender_name: Optional[str] = None,
        chat_id: Optional[str] = None,  # Made optional, Agency might manage this
        context_override: Optional[Dict[str, Any]] = None,
        hooks_override: Optional[RunHooks] = None,
        run_config: Optional[RunConfig] = None,
        **kwargs: Any,  # Pass-through to Runner
    ) -> RunResult:
        """Runs the agent's turn using the SDK Runner, returning the full result."""
        # with custom_span(f"Agent Turn: {self.name}.get_response") as agent_turn_span: # Commented out tracing
        # 1. Validate Prerequisites & Get Thread
        if not self._thread_manager:
            raise RuntimeError(f"Agent '{self.name}' missing ThreadManager.")
        if not self._agency_instance or not hasattr(self._agency_instance, "agents"):
            raise RuntimeError(f"Agent '{self.name}' missing Agency instance or agents map.")

        # Determine chat_id if not provided (e.g., for user interaction)
        effective_chat_id = chat_id
        if sender_name is None and not effective_chat_id:
            effective_chat_id = f"chat_{uuid.uuid4()}"
            logger.info(f"New user interaction, generated chat_id: {effective_chat_id}")
        elif sender_name is not None and not effective_chat_id:
            # This case should be prevented by the check below, but handle defensively
            raise ValueError("chat_id is required for agent-to-agent communication within get_response.")

        logger.info(f"Agent '{self.name}' handling get_response for chat_id: {effective_chat_id}")
        thread = self._thread_manager.get_thread(effective_chat_id)
        # agent_turn_span.set_attribute("chat_id", chat_id) # Commented out tracing

        # --- ADDED: Add user message to thread before run --- #
        if sender_name is None:  # Only add if it's initial user input
            try:
                items_to_add = ItemHelpers.input_to_new_input_list(message)
                # Add items to the thread object in memory
                # self._thread_manager.add_items_and_save(thread, items_to_add)
                thread.add_items(items_to_add)
                logger.debug(f"Added initial user message to thread {thread.thread_id} before run.")
            except Exception as e:
                logger.error(f"Error processing initial input message for get_response: {e}", exc_info=True)
        # --- END ADDED --- #

        # 3. Prepare Context (History is handled internally by Runner now)
        # history_for_runner = thread.get_history() # Don't need to get history here
        master_context = self._prepare_master_context(context_override, effective_chat_id)

        # 4. Prepare Hooks & Config
        hooks_to_use = hooks_override or self.hooks
        effective_run_config = run_config or RunConfig()

        # 5. Execute via Runner
        try:
            logger.debug(f"Calling Runner.run for agent '{self.name}'...")
            # Call Runner.run as a class method, passing the initial input
            run_result: RunResult = await Runner.run(
                starting_agent=self,
                input=message,  # Runner handles adding this initial input
                context=master_context,
                hooks=hooks_to_use,
                run_config=effective_run_config,
                max_turns=kwargs.get("max_turns", DEFAULT_MAX_TURNS),
                previous_response_id=kwargs.get("previous_response_id"),
            )
            # Log completion based on presence of final_output
            completion_info = (
                f"Output Type: {type(run_result.final_output).__name__}"
                if run_result.final_output is not None
                else "No final output"
            )
            logger.info(f"Runner.run completed for agent '{self.name}'. {completion_info}")
            # ... tracing commented out ...

        except Exception as e:
            logger.error(f"Error during Runner.run for agent '{self.name}': {e}", exc_info=True)
            # ... tracing commented out ...
            raise AgentsException(f"Runner execution failed for agent {self.name}") from e

        # 6. Optional: Validate Response
        response_text_for_validation = ""
        if run_result.new_items:
            # Use ItemHelpers to extract text from message output items in the result
            response_text_for_validation = ItemHelpers.text_message_outputs(run_result.new_items)

        if response_text_for_validation and self.response_validator:
            if not self._validate_response(response_text_for_validation):
                logger.warning(f"Response validation failed for agent '{self.name}'")

        # 7. Add final result items to thread ONLY if it's a top-level call (from user/agency)
        if sender_name is None:
            if self._thread_manager and run_result.new_items:
                thread = self._thread_manager.get_thread(effective_chat_id)
                items_to_save: List[TResponseInputItem] = []
                logger.debug(f"Preparing to save {len(run_result.new_items)} new items to thread {thread.thread_id}")
                for i, run_item in enumerate(run_result.new_items):
                    item_dict = self._run_item_to_tresponse_input_item(run_item)
                    if item_dict:
                        items_to_save.append(item_dict)
                        logger.debug(f"  Item {i+1}/{len(run_result.new_items)} converted for saving: {item_dict}")
                    else:
                        logger.debug(
                            f"  Item {i+1}/{len(run_result.new_items)} ({type(run_item).__name__}) skipped or failed conversion."
                        )

                if items_to_save:
                    logger.info(f"Saving {len(items_to_save)} converted items to thread {thread.thread_id}")
                    self._thread_manager.add_items_and_save(thread, items_to_save)

        # 8. Return Result
        return run_result

    async def get_response_stream(
        self,
        message: Union[str, List[Dict[str, Any]]],
        sender_name: Optional[str] = None,
        chat_id: Optional[str] = None,
        context_override: Optional[Dict[str, Any]] = None,
        hooks_override: Optional[RunHooks] = None,
        run_config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Runs the agent's turn using the SDK Runner, yielding stream events."""
        # ... initial setup and validation ...
        if not self._thread_manager:
            raise RuntimeError(f"Agent '{self.name}' missing ThreadManager.")
        # ... other checks ...
        effective_chat_id = chat_id
        if sender_name is None and not effective_chat_id:
            effective_chat_id = f"chat_{uuid.uuid4()}"
            logger.info(f"New user stream interaction, generated chat_id: {effective_chat_id}")
        elif sender_name is not None and not effective_chat_id:
            raise ValueError("chat_id is required for agent-to-agent stream communication.")

        logger.info(f"Agent '{self.name}' handling get_response_stream for chat_id: {effective_chat_id}")

        # Add user message to thread *before* starting the run
        # This assumes ThreadManager uses TResponseInputItem dicts now.
        if sender_name is None:
            try:
                thread = self._thread_manager.get_thread(effective_chat_id)
                items_to_add = ItemHelpers.input_to_new_input_list(message)  # Convert string to dict list
                thread.add_items(items_to_add)
                self._thread_manager.add_items_and_save(thread, items_to_add)
            except Exception as e:
                logger.error(f"Error processing input message for stream: {e}", exc_info=True)
                yield {"error": f"Invalid input message format: {e}"}  # Yield error event
                return  # Stop the generator

        # history_for_runner = thread.get_history() # Not needed for Runner input
        master_context = self._prepare_master_context(context_override, effective_chat_id)
        hooks_to_use = hooks_override or self.hooks
        effective_run_config = run_config or RunConfig()
        final_result_items = []  # To capture items for saving at the end

        # Execute via Runner stream
        try:
            logger.debug(f"Calling Runner.run_streamed for agent '{self.name}'...")
            # Use Runner.run_streamed
            async for event in Runner.run_streamed(
                starting_agent=self,
                input=message,  # Runner handles adding initial input
                context=master_context,
                hooks=hooks_to_use,
                run_config=effective_run_config,
                max_turns=kwargs.get("max_turns", DEFAULT_MAX_TURNS),
                previous_response_id=kwargs.get("previous_response_id"),
            ):
                yield event
                # Collect RunItems from the stream events if needed for final processing
                if isinstance(event, RunItemStreamEvent):
                    final_result_items.append(event.item)

            logger.info(f"Runner.run_streamed completed for agent '{self.name}'.")

        except Exception as e:
            logger.error(f"Error during Runner.run_streamed for agent '{self.name}': {e}", exc_info=True)
            # Yield an error event if streaming fails
            yield {"error": f"Runner execution failed: {e}"}
            # Optional: re-raise or handle differently
            return  # Stop the generator after yielding error

        # 6. Optional: Validate Response (using collected items)
        response_text_for_validation = ""
        if final_result_items:
            response_text_for_validation = ItemHelpers.text_message_outputs(final_result_items)

        if response_text_for_validation and self.response_validator:
            if not self._validate_response(response_text_for_validation):
                logger.warning(f"Response validation failed for agent '{self.name}' after stream.")

        # 7. Add final result items to thread (using collected items)
        if self._thread_manager and final_result_items:
            thread = self._thread_manager.get_thread(effective_chat_id)
            items_to_save: List[TResponseInputItem] = []
            for run_item in final_result_items:
                item_dict = self._run_item_to_tresponse_input_item(run_item)
                if item_dict:
                    items_to_save.append(item_dict)
            if items_to_save:
                self._thread_manager.add_items_and_save(thread, items_to_save)

    # --- Helper Methods ---
    def _run_item_to_tresponse_input_item(self, item: RunItem) -> Optional[TResponseInputItem]:
        """Converts a RunItem into the TResponseInputItem dictionary format for history.
        Returns None if the item type shouldn't be added to history directly.
        """
        # Import necessary types locally within the function if needed, or ensure they are available
        # Adjust imports based on SDK structure
        from openai.types.responses import (
            ResponseComputerToolCall,
            ResponseFileSearchToolCall,
            ResponseFunctionToolCall,
            ResponseFunctionWebSearch,  # Removed FunctionCallOutput, ComputerCallOutput here
        )

        # Import nested types from correct location
        from openai.types.responses.response_input_item_param import (
            ComputerCallOutput,
            FunctionCallOutput,
        )

        from agents.items import ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem

        if isinstance(item, MessageOutputItem):
            # Extract text content for simplicity; complex content needs more handling
            content = ItemHelpers.text_message_output(item)
            logger.debug(
                f"Converting MessageOutputItem to history: role=assistant, content='{content[:50]}...'"
            )  # DEBUG
            return {"role": "assistant", "content": content}

        elif isinstance(item, ToolCallItem):
            # Construct tool_calls list
            tool_calls = []
            # Handle different raw_item types within ToolCallItem
            if isinstance(item.raw_item, ResponseFunctionToolCall):
                # Access attributes directly from ResponseFunctionToolCall
                call_id = getattr(item.raw_item, "call_id", None)
                func_name = getattr(item.raw_item, "name", None)
                func_args = getattr(item.raw_item, "arguments", None)

                if not call_id or not func_name:
                    logger.warning(f"Missing call_id or name in ResponseFunctionToolCall: {item.raw_item}")
                    return None

                # Need to handle potential serialization issues with func_args
                # It's often a string already, but might be dict/list
                if isinstance(func_args, (dict, list)):
                    args_str = json.dumps(func_args)
                elif isinstance(func_args, str):
                    args_str = func_args  # Assume it's valid JSON string if it's a string
                else:
                    args_str = str(func_args)  # Fallback

                logger.debug(
                    f"Converting ToolCallItem (Function) to history: id={call_id}, name={func_name}, args='{args_str[:50]}...'"
                )  # DEBUG
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": func_name, "arguments": args_str},
                    }
                )
            # Add elif blocks here for other tool call types if needed
            # elif isinstance(item.raw_item, ResponseComputerToolCall): ...
            # elif isinstance(item.raw_item, ResponseFileSearchToolCall): ...
            else:
                logger.warning(f"Unhandled raw_item type in ToolCallItem: {type(item.raw_item)}")
                return None  # Or handle appropriately

            if tool_calls:
                # Ensure content is None when tool_calls are present
                return {"role": "assistant", "content": None, "tool_calls": tool_calls}
            else:
                return None  # No valid tool calls extracted

        elif isinstance(item, ToolCallOutputItem):
            # Construct tool call output item
            tool_call_id = None
            output_content = str(item.output)  # Use the processed output
            # Check structure instead of isinstance for TypedDict
            if isinstance(item.raw_item, dict) and item.raw_item.get("type") == "function_call_output":
                tool_call_id = item.raw_item.get("call_id")
                logger.debug(
                    f"Converting ToolCallOutputItem (Function) to history: tool_call_id={tool_call_id}, content='{output_content[:50]}...'"
                )  # DEBUG

            # Add similar checks here if handling ComputerCallOutput, etc.
            # elif isinstance(item.raw_item, dict) and item.raw_item.get('type') == 'computer_call_output':
            #     tool_call_id = item.raw_item.get("call_id")
            #     logger.debug(f"Converting ToolCallOutputItem (Computer) to history: tool_call_id={tool_call_id}, content='{output_content[:50]}...'") # DEBUG

            if tool_call_id:
                # Content should be stringified output
                return {"role": "tool", "tool_call_id": tool_call_id, "content": output_content}
            else:
                logger.warning(f"Could not determine tool_call_id for ToolCallOutputItem: {item.raw_item}")
                return None

        # Add handling for other RunItem types if needed (e.g., HandoffOutputItem?)
        # elif isinstance(item, UserInputItem) -> Should already be handled when initially added

        else:
            logger.debug(f"Skipping RunItem type {type(item).__name__} for thread history saving.")
            return None

    def _prepare_master_context(
        self, context_override: Optional[Dict[str, Any]], chat_id: Optional[str]
    ) -> MasterContext:
        """Constructs the MasterContext for the current run."""
        if not self._agency_instance or not hasattr(self._agency_instance, "agents"):
            raise RuntimeError("Cannot prepare context: Agency instance or agents map missing.")
        if not self._thread_manager:
            raise RuntimeError("Cannot prepare context: ThreadManager missing.")

        # Start with base user context from agency, if it exists
        base_user_context = getattr(self._agency_instance, "user_context", {})
        merged_user_context = base_user_context.copy()
        if context_override:
            merged_user_context.update(context_override)

        return MasterContext(
            thread_manager=self._thread_manager,
            agents=self._agency_instance.agents,
            user_context=merged_user_context,
            current_agent_name=self.name,
            chat_id=chat_id,  # Pass chat_id to context
        )

    def _validate_response(self, response_text: str) -> bool:
        """Internal helper to apply response validator if configured."""
        if self.response_validator:
            try:
                is_valid = self.response_validator(response_text)
                if not is_valid:
                    logger.warning(f"Response validation failed for agent {self.name}")
                return is_valid
            except Exception as e:
                logger.error(f"Error during response validation for agent {self.name}: {e}", exc_info=True)
                return False  # Treat validation errors as failure
        return True  # No validator means always valid

    # --- Agency Configuration Methods --- (Called by Agency)
    def _set_thread_manager(self, manager: ThreadManager):
        """Allows the Agency to inject the ThreadManager instance."""
        self._thread_manager = manager

    def _set_agency_instance(self, agency: Any):
        """Allows the Agency to inject a reference to itself and its agent map."""
        if not hasattr(agency, "agents"):
            raise TypeError("Provided agency instance must have an 'agents' dictionary.")
        self._agency_instance = agency
