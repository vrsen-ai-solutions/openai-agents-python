from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, NotFoundError

from agents import (
    Agent as BaseAgent,
    AgentOutputSchema,
    AgentOutputSchemaBase,
    FileSearchTool,
    # AgentState, # Not directly used?
    FunctionTool,
    ModelResponse,
    RunItem,
    Runner,
    RunResult,
    TContext,
    Tool,
    TResponseInputItem,
)
from agents.exceptions import AgentsException
from agents.items import (
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.models.interface import Model
from agents.tool import FunctionTool
from agents.tracing import trace

# Import Agency Swarm specific concepts/helpers
from .thread import ConversationThread, ThreadManager

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Configure logger if not already configured elsewhere
# logging.basicConfig(level=logging.INFO) # Avoid double config if Agency does it

# --- Constants ---
SEND_MESSAGE_TOOL_NAME = "SendMessage"


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
    from .thread import (
        ConversationThread,
        ThreadManager,  # Already imported
    )
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

    # --- Internal State/Config ---
    _thread_manager: Optional[ThreadManager] = None  # Keep this
    _agency_chart_peers: Optional[List[str]] = (
        None  # Allowed agents to call via send_message, set by Agency
    )
    _agency_instance: Optional[Any] = None  # Reference back to Agency
    _associated_vector_store_id: Optional[str] = None  # Added in Task 9
    files_folder_path: Optional[Path] = None  # Added in Task 9

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

        # --- Internal state needs to be configured, likely by the Agency ---
        # These are typically set *after* init using setter methods by the Agency
        self._thread_manager = None
        self._agency_chart_peers = None
        self._agency_instance = None

        # --- Initialization Logic (Now includes loading tools) ---
        self._load_tools_from_folder()  # Load tools from folder now
        self._init_file_handling()  # Call the implemented method
        # TODO (Task 9): self._init_file_handling()

    # --- OpenAI Client ---
    # Helper property to get an initialized client
    # Assumes OPENAI_API_KEY is set in the environment
    @property
    def client(self) -> AsyncOpenAI:
        if not hasattr(self, "_openai_client"):
            self._openai_client = AsyncOpenAI()
        return self._openai_client

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

    # --- File Handling (Task 9) ---

    def _init_file_handling(self) -> None:
        """Initializes file handling logic based on files_folder parameter.
        Creates the folder if it doesn't exist, parses Vector Store ID if present,
        and potentially syncs local files with the associated Vector Store.
        """
        self._associated_vector_store_id: Optional[str] = None
        self.files_folder_path: Optional[Path] = None  # Store the resolved Path

        if self.files_folder:
            self.files_folder_path = Path(self.files_folder).resolve()  # Resolve path early
            folder_name = self.files_folder_path.name

            # Regex to find _vs_XXX pattern at the end of the folder name
            match = re.search(r"_vs_([a-zA-Z0-9]+)$", folder_name)
            if match:
                vs_id = match.group(1)
                print(f"Folder name suggests Vector Store ID: {vs_id}")
                # TODO: Validate VS ID exists via API? For now, assume it's correct.
                self._associated_vector_store_id = vs_id
                # Use the path without the VS ID suffix for local storage
                base_folder_name = folder_name[: match.start()]
                parent_dir = self.files_folder_path.parent
                self.files_folder_path = parent_dir / base_folder_name  # Update path

            # Ensure the local folder exists
            self.files_folder_path.mkdir(parents=True, exist_ok=True)

            if not self.files_folder_path.is_dir():
                print(
                    f"Warning: files_folder path exists but is not a directory: {self.files_folder_path}"
                )
            else:
                print(
                    f"File handling initialized. Agent local files folder: {self.files_folder_path}"
                )
                # Potentially sync local folder with VS on init? (Could be slow)
                # asyncio.run(self._sync_local_folder_to_vs()) # Example - consider implications

            # Ensure FileSearchTool is configured if VS ID is present
            if self._associated_vector_store_id:
                self._ensure_file_search_tool()
        else:
            print("No files_folder specified for this agent.")

    async def _sync_local_folder_to_vs(self):
        """(Conceptual) Uploads files from local folder that aren't in the VS."""
        if not self.files_folder_path or not self._associated_vector_store_id:
            return

        print(
            f"Checking sync status for local folder {self.files_folder_path} and VS {self._associated_vector_store_id}"
        )
        try:
            # Check if VS exists
            try:
                vector_store = await self.client.beta.vector_stores.retrieve(
                    self._associated_vector_store_id
                )
                print(f"Found existing Vector Store: {vector_store.id}")
            except NotFoundError:
                print(f"Vector Store {self._associated_vector_store_id} not found. Cannot sync.")

        except NotFoundError:
            print(f"Error: Vector Store {self._associated_vector_store_id} not found during sync.")
        except Exception as e:
            print(f"Error during folder sync: {e}")

    def _ensure_file_search_tool(self):
        """Adds or updates the FileSearchTool if a VS ID is associated."""
        if not self._associated_vector_store_id:
            return

        fs_tool_exists = False
        for i, tool in enumerate(self.tools):
            if isinstance(tool, FileSearchTool):
                # Update existing tool if VS ID differs or not set
                if tool.vector_store_ids != [self._associated_vector_store_id]:
                    print(
                        f"Updating existing FileSearchTool with VS ID: {self._associated_vector_store_id}"
                    )
                    self.tools[i] = FileSearchTool(
                        vector_store_ids=[self._associated_vector_store_id]
                    )
                else:
                    print("FileSearchTool already configured correctly.")
                fs_tool_exists = True
                break

        if not fs_tool_exists:
            print(f"Adding new FileSearchTool for VS ID: {self._associated_vector_store_id}")
            self.add_tool(FileSearchTool(vector_store_ids=[self._associated_vector_store_id]))

    async def upload_file(self, file_path: str) -> str:  # Return OpenAI File ID
        """Uploads a file locally (if files_folder is set) following naming convention,
        uploads the file to OpenAI, and associates it with the agent's Vector Store if configured.

        Returns:
            The OpenAI File ID (e.g., 'file-XXXXXXXX').

        Raises:
            FileNotFoundError: If the source file_path does not exist.
            ValueError: If file upload to OpenAI fails.
            IOError: If local file copy fails.
        """
        source_path = Path(file_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        base_name = source_path.stem
        extension = source_path.suffix
        local_file_id_part = f"local_{uuid.uuid4()}"  # Use a distinct local ID part for clarity

        # --- Local File Management (if files_folder is set) ---
        local_destination_path: Optional[Path] = None
        if self.files_folder_path:
            # Check local existence first using the base name only (more robust than convention)
            # Note: This check is simplified; doesn't guarantee content match.
            # Consider hashing if content-based duplication check is needed.
            existing_local_file = next(
                self.files_folder_path.glob(f"{re.escape(base_name)}_*.*"), None
            )

            if existing_local_file:
                print(
                    f"File with base name '{base_name}' already exists locally: {existing_local_file.name}. Reusing."
                )
                # TODO: Should we still upload to OpenAI if it exists locally?
                # Yes, upload to OpenAI is the primary goal. Local is cache/reference.
                local_destination_path = existing_local_file  # Use existing local path for upload
            else:
                # Construct the new local filename: basename_local_UUID.ext
                new_filename = f"{base_name}_{local_file_id_part}{extension}"
                local_destination_path = self.files_folder_path / new_filename
                try:
                    shutil.copy2(source_path, local_destination_path)  # copy2 preserves metadata
                    print(f"File '{source_path.name}' copied locally to '{local_destination_path}'")
                except Exception as e:
                    print(f"Error copying file {source_path} locally: {e}")
                    # Don't raise immediately, OpenAI upload is primary. Log and continue.
                    # raise IOError(f"Failed to copy file locally: {e}")
        else:
            # No local folder, use the original source path for upload
            local_destination_path = source_path

        # --- OpenAI File Upload ---
        print(f"Uploading file '{local_destination_path.name}' to OpenAI...")
        try:
            # Use 'assistants' purpose for FileSearchTool compatibility
            openai_file = await self.client.files.create(
                file=local_destination_path.open("rb"), purpose="assistants"
            )
            print(f"File uploaded successfully to OpenAI. File ID: {openai_file.id}")

            # --- Associate with Vector Store (if applicable) ---
            if self._associated_vector_store_id:
                print(
                    f"Adding OpenAI file {openai_file.id} to VS {self._associated_vector_store_id}..."
                )
                try:
                    # Check if file is already in the VS to avoid errors
                    # Note: Listing all files can be slow for large VS.
                    # Consider alternative strategies if performance is critical.
                    vs_files = await self.client.beta.vector_stores.files.list(
                        vector_store_id=self._associated_vector_store_id
                    )
                    if any(f.id == openai_file.id for f in vs_files.data):
                        print(
                            f"File {openai_file.id} already exists in VS {self._associated_vector_store_id}."
                        )
                    else:
                        vs_file = await self.client.beta.vector_stores.files.create(
                            vector_store_id=self._associated_vector_store_id, file_id=openai_file.id
                        )
                        print(
                            f"Successfully added file {vs_file.id} to VS {vs_file.vector_store_id}."
                        )
                        # Ensure the FileSearchTool is up-to-date (might have been added after init)
                        self._ensure_file_search_tool()
                except NotFoundError:
                    print(
                        f"Error: Vector Store {self._associated_vector_store_id} not found when trying to add file."
                    )
                    # Should this re-raise or just warn? Warn for now.
                except Exception as e:
                    print(
                        f"Error adding file {openai_file.id} to VS {self._associated_vector_store_id}: {e}"
                    )
                    # Warn, but proceed returning the OpenAI file ID

            return openai_file.id  # Return the OpenAI file ID

        except Exception as e:
            print(f"Error uploading file {local_destination_path.name} to OpenAI: {e}")
            raise ValueError(f"Failed to upload file to OpenAI: {e}")

    async def check_file_exists(
        self, file_path: str
    ) -> Optional[str]:  # Return OpenAI File ID if found
        """Checks if a file with the same name has already been uploaded to OpenAI
           files associated with this agent's Vector Store (if configured).
           NOTE: This method is now async.

        Args:
            file_path: The path to the local file to check.

        Returns:
            The OpenAI File ID if a matching file (by name) is found in the VS, otherwise None.
            Returns None if no Vector Store is configured for the agent.
        """
        if not self._associated_vector_store_id:
            return None  # No VS configured

        source_path = Path(file_path)
        target_filename = source_path.name  # Check based on original filename in OpenAI

        try:
            # This is inefficient as it lists all files.
            # OpenAI API doesn't currently support filtering files by name directly.
            # Consider alternative strategies for large numbers of files (e.g., local cache/db).
            print(
                f"Checking for file '{target_filename}' in VS {self._associated_vector_store_id}..."
            )
            vs_files_page = await self.client.beta.vector_stores.files.list(
                vector_store_id=self._associated_vector_store_id,
                limit=100,  # Adjust limit as needed, might need pagination
            )
            # TODO: Implement pagination if needed

            for vs_file in vs_files_page.data:
                # Need to retrieve the original filename from the File object
                try:
                    file_object = await self.client.files.retrieve(vs_file.id)
                    if file_object.filename == target_filename:
                        print(f"Found matching file in VS: {target_filename} (ID: {vs_file.id})")
                        return vs_file.id
                except NotFoundError:
                    print(
                        f"Warning: File ID {vs_file.id} listed in VS {self._associated_vector_store_id} but not found in OpenAI files."
                    )
                except Exception as e:
                    print(f"Error retrieving details for file {vs_file.id}: {e}")

            print(f"File '{target_filename}' not found in VS {self._associated_vector_store_id}.")
            return None
        except NotFoundError:
            print(f"Error: Vector Store {self._associated_vector_store_id} not found during check.")
            return None
        except Exception as e:
            print(f"Error checking file existence in VS {self._associated_vector_store_id}: {e}")
        return None

    # --- Response Handling ---
    # These methods provide a way to run this agent standalone, outside of an Agency.
    # For orchestrated multi-agent interactions, use Agency.run_interaction methods.

    async def get_response(
        self,
        message: str,  # Initial message content
        chat_id: str,  # Explicit chat_id for thread management
        sender_name: Optional[str] = None,  # None if user, or name of calling agent
        # message_files handled internally now by upload_file if needed before calling
        run_config: Optional[Dict[str, Any]] = None,  # Keep for potential future use?
        text_only: bool = False,
        max_steps: Optional[int] = 25,  # Max steps for this agent's execution loop
        # current_depth: int = 0, # Add later for recursion check
        **kwargs: Any,  # Pass-through? Maybe remove if not used?
    ) -> Union[str, RunResult]:
        """Runs this agent's turn within a conversation thread, handling orchestration.

        Args:
            message: The user message input.
            chat_id: The explicit chat_id for thread management.
            sender_name: The name of the calling agent or None if user.
            run_config: Optional dictionary to configure the run (maps to RunConfig object).
            text_only: If True, return only the final text output, otherwise return the full RunResult.
            max_steps: The maximum number of steps for this agent's execution loop.
            **kwargs: Additional arguments passed directly to `Runner.run`.

        Returns:
            The final text output (if text_only=True) or the full RunResult object.
        """
        # --- Start: Refactored Orchestration Logic ---
        if not self._thread_manager:
            raise RuntimeError(f"Agent '{self.name}' cannot execute: ThreadManager not configured.")
        if not self._agency_instance:
            # This check might be too strict if running standalone is sometimes ok?
            # For now, assume orchestration requires the agency context.
            raise RuntimeError(
                f"Agent '{self.name}' cannot execute: Agency instance not configured."
            )

        logger.info(f"Agent '{self.name}' starting get_response for chat_id: {chat_id}")
        thread = self._thread_manager.get_thread(chat_id)

        # Add initial message if this is the start of the interaction (sender_name is None)
        if sender_name is None:
            thread.add_user_message(message)
            self._thread_manager.add_item_and_save(thread, thread.items[-1])  # Save added message

        current_step = 0
        max_steps_local = max_steps or 25  # Use provided max_steps or default

        # Track items generated specifically during this agent's execution run
        # This helps construct the final RunResult for this agent's scope
        run_items: List[RunItem] = []
        model_responses: List[ModelResponse] = []  # Track underlying model responses
        final_output: Any = None
        final_output_text: Optional[str] = None

        # --- Core Agent Execution Loop ---
        while current_step < max_steps_local:
            current_step += 1
            logger.info(
                f"Agent '{self.name}' - Step {current_step}/{max_steps_local} in chat {chat_id}"
            )

            # 1. Prepare for LLM call
            history_for_model = thread.get_history()
            tools_for_model = await self.get_all_tools()  # Includes SendMessage if allowed
            tool_schemas = [t.openai_schema for t in tools_for_model if hasattr(t, "openai_schema")]

            # TODO: Handle model settings (self.model, self.model_settings)
            model_to_use = self.model or "gpt-4o"  # Default if not set

            # 2. Call LLM
            try:
                logger.debug(f"Calling LLM ({model_to_use}) for agent '{self.name}'")
                # Note: Need to map our history format to ChatCompletionMessageParam list
                # The get_history method already does this conversion.
                response = await self.client.chat.completions.create(
                    model=model_to_use,
                    messages=history_for_model,  # Already formatted
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto",  # Or specific choice if needed
                    # TODO: Add temperature, max_tokens from self.model_settings?
                )
                model_responses.append(response)  # Track raw response

                # Process response message (content, tool calls)
                response_message = response.choices[0].message

                # Create corresponding RunItems (mimicking SDK structure)
                llm_output_items: List[RunItem] = []
                if response_message.content:
                    text_item = MessageOutputItem(content=response_message.content)
                    llm_output_items.append(text_item)
                    final_output = text_item.content  # Keep track of last text output
                    final_output_text = text_item.content

                if response_message.tool_calls:
                    for tc in response_message.tool_calls:
                        # Assuming all are function tools for now
                        tool_call_item = ToolCallItem(
                            id=tc.id,
                            name=tc.function.name,
                            args_raw=tc.function.arguments,  # Store raw args
                            # Attempt to parse JSON, fallback to raw string
                            args_json=json.loads(tc.function.arguments)
                            if tc.function.arguments
                            else {},
                        )
                        llm_output_items.append(tool_call_item)

                if not llm_output_items:
                    # Should not happen with valid API response, but handle defensively
                    logger.warning("LLM response had no content or tool calls.")
                    # Consider breaking or returning error?
                    # For now, add a generic empty message to avoid loop errors
                    llm_output_items.append(MessageOutputItem(content=""))

                # Add LLM output items to thread and current run log
                thread.add_items(llm_output_items)
                self._thread_manager.add_items_and_save(thread, llm_output_items)  # Save changes
                run_items.extend(llm_output_items)

            except Exception as e:
                logger.error(f"Error calling LLM for agent '{self.name}': {e}", exc_info=True)
                # TODO: Handle LLM error - maybe retry or return error RunResult?
                raise AgentsException(f"LLM call failed for agent {self.name}") from e

            # 3. Process Tool Calls (including SendMessage)
            send_message_call: Optional[ToolCallItem] = None
            standard_tool_calls: List[ToolCallItem] = []

            # First, find if SendMessage was called
            for item in llm_output_items:
                if isinstance(item, ToolCallItem) and item.name == SEND_MESSAGE_TOOL_NAME:
                    send_message_call = item
                    break  # Prioritize SendMessage

            # If SendMessage was not called, collect standard tool calls
            if not send_message_call:
                for item in llm_output_items:
                    if isinstance(item, ToolCallItem):
                        standard_tool_calls.append(item)

            # --- Handle SendMessage (if called) --- (Correct Indentation)
            if send_message_call:
                logger.info(f"Agent '{self.name}' initiating SendMessage.")
                args = send_message_call.args_json
                recipient_name = args.get("recipient")
                message_content = args.get("message")

                is_allowed = self._agency_instance.is_communication_allowed(
                    self.name, recipient_name
                )
                recipient_agent = self._agency_instance.agents.get(recipient_name)

                if (
                    not recipient_name
                    or message_content is None
                    or not is_allowed
                    or not recipient_agent
                ):
                    # Construct specific error message
                    if not recipient_name or message_content is None:
                        error_msg = f"SendMessage failed: Missing recipient ('{recipient_name}') or message in args: {args}"
                    elif not is_allowed:
                        error_msg = f"Communication DENIED: Agent '{self.name}' cannot send to '{recipient_name}'."
                    else:  # not recipient_agent
                        error_msg = (
                            f"SendMessage failed: Recipient agent '{recipient_name}' not found."
                        )

                    logger.error(error_msg)
                    tool_output_item = ToolCallOutputItem(
                        tool_call_id=send_message_call.id,
                        name=send_message_call.name,
                        content=error_msg,
                        is_error=True,
                    )
                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    # Continue loop after failed SendMessage
                    continue
                else:
                    # --- Recursive Call ---
                    logger.info(
                        f"Agent '{self.name}' calling agent '{recipient_name}' via SendMessage."
                    )
                    try:
                        recursive_result: RunResult = await recipient_agent.get_response(
                            message=message_content,
                            chat_id=thread.thread_id,
                            sender_name=self.name,
                            max_steps=max_steps_local - current_step,
                            text_only=False,  # Always get RunResult internally
                            # TODO: Pass current_depth+1
                        )

                        output_content = getattr(recursive_result, "final_output_text", None)
                        if output_content is None:
                            output_content = "(No text output from recipient)"

                        tool_output_item = ToolCallOutputItem(
                            tool_call_id=send_message_call.id,
                            name=send_message_call.name,
                            content=output_content,
                        )
                        logger.info(f"SendMessage call to '{recipient_name}' completed.")

                    except Exception as e_recurse:
                        error_msg = (
                            f"Error during recursive call to agent '{recipient_name}': {e_recurse}"
                        )
                        logger.error(error_msg, exc_info=True)
                        tool_output_item = ToolCallOutputItem(
                            tool_call_id=send_message_call.id,
                            name=send_message_call.name,
                            content=error_msg,
                            is_error=True,
                        )

                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    # After successful/failed SendMessage handling, continue the loop
                    continue

            # --- Handle Standard Tool Calls --- (Correct Indentation)
            elif standard_tool_calls:
                logger.info(
                    f"Agent '{self.name}' executing standard tools: {[tc.name for tc in standard_tool_calls]}"
                )
                tool_outputs_generated = False
                for tool_call in standard_tool_calls:
                    tool_to_execute = next(
                        (
                            t
                            for t in tools_for_model
                            if hasattr(t, "name") and t.name == tool_call.name
                        ),
                        None,
                    )
                    output_content = ""
                    is_error = False

                    if not tool_to_execute or not hasattr(tool_to_execute, "on_invoke_tool"):
                        error_msg = f"Tool '{tool_call.name}' not found or not executable."
                        logger.error(error_msg)
                        output_content = error_msg
                        is_error = True
                    else:
                        try:
                            logger.debug(
                                f"Executing tool: {tool_call.name} with args: {tool_call.args_json}"
                            )
                            if asyncio.iscoroutinefunction(tool_to_execute.on_invoke_tool):
                                output_content = await tool_to_execute.on_invoke_tool(
                                    None, tool_call.args_json
                                )
                            else:
                                output_content = tool_to_execute.on_invoke_tool(
                                    None, tool_call.args_json
                                )
                            is_error = False
                            logger.debug(f"Tool '{tool_call.name}' execution successful.")
                        except Exception as e_tool:
                            error_msg = f"Error executing tool '{tool_call.name}': {e_tool}"
                            logger.error(error_msg, exc_info=True)
                            output_content = error_msg
                            is_error = True

                    tool_output_item = ToolCallOutputItem(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=str(output_content),  # Ensure content is string
                        is_error=is_error,
                    )
                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    tool_outputs_generated = True

                # If standard tools were executed, continue the loop to process their output
                if tool_outputs_generated:
                    continue

            # --- No Tool Calls were made by LLM this step --- (Correct Indentation)
            else:
                if final_output_text is not None:
                    logger.info(f"Agent '{self.name}' produced final text output. Ending loop.")
                    break  # Exit the while loop
                else:
                    logger.warning(
                        f"Agent '{self.name}' finished step {current_step} with no text or tool calls. Breaking loop."
                    )
                    break  # Exit the while loop

        # --- End of Loop ---

        # 4. Construct Final RunResult
        final_run_result = RunResult(
            run_id=f"as_run_{uuid.uuid4()}",  # Generate a run ID
            thread_id=thread.thread_id,
            items=run_items,  # Items generated during *this specific agent run*
            model_responses=model_responses,
            final_output=final_output,  # Last text or final object if structured output used
            final_output_text=final_output_text,  # Last text output
            usage=None,  # TODO: Aggregate usage from model_responses if available
        )

        logger.info(f"Agent '{self.name}' finished get_response for chat_id: {chat_id}")

        # 5. Response Validation (Using the internal helper)
        if final_output_text is not None:
            if not self._validate_response(final_output_text):
                # TODO: Decide how to handle validation failure - raise error? Return specific result?
                # Raise ValueError for now, mirroring previous conceptual logic
                raise ValueError(f"Response validation failed for agent '{self.name}'")

        # 6. Return Result
        return final_output_text if text_only else final_run_result
        # --- End: Refactored Orchestration Logic ---

    async def get_response_stream(
        self,
        message: str,  # Initial message content
        chat_id: str,  # Explicit chat_id for thread management
        sender_name: Optional[str] = None,  # None if user, or name of calling agent
        run_config: Optional[Dict[str, Any]] = None,  # Keep for potential future use?
        max_steps: Optional[int] = 25,  # Max steps for this agent's execution loop
        # current_depth: int = 0, # Add later for recursion check
        **kwargs: Any,  # Pass-through? Maybe remove if not used?
    ) -> AsyncGenerator[Any, None]:  # TODO: Define precise yield type (e.g., SDK StreamEvent?)
        """Runs this agent's turn within a thread, yielding streaming events."""
        # --- Start: Streaming Orchestration Logic ---
        if not self._thread_manager:
            raise RuntimeError(f"Agent '{self.name}' cannot execute: ThreadManager not configured.")
        if not self._agency_instance:
            raise RuntimeError(
                f"Agent '{self.name}' cannot execute: Agency instance not configured."
            )

        logger.info(f"Agent '{self.name}' starting get_response_stream for chat_id: {chat_id}")
        thread = self._thread_manager.get_thread(chat_id)

        # Add initial message if this is the start of the interaction (sender_name is None)
        if sender_name is None:
            # Convert string to list of input items using helper
            try:
                user_input_list = ItemHelpers.input_to_new_input_list(message)
                if user_input_list:  # Ensure list is not empty
                    user_item = user_input_list[0]  # Get the dict item
                    thread.add_item(user_item)
                    self._thread_manager.add_item_and_save(thread, user_item)  # Save added message
                else:
                    logger.warning(f"Could not convert initial user message for chat {chat_id}")
            except Exception as e:
                logger.error(
                    f"Error converting initial user message for chat {chat_id}: {e}", exc_info=True
                )
                # Decide if we should raise or just log and continue?
                # For now, log and continue without adding initial message
            # yield RunItemStreamEvent(run_item=user_item) # Example yield

        current_step = 0
        max_steps_local = max_steps or 25

        run_items: List[RunItem] = []
        model_responses: List[ModelResponse] = []

        # --- Core Agent Streaming Loop ---
        while current_step < max_steps_local:
            current_step += 1
            logger.info(
                f"Agent '{self.name}' - STREAM Step {current_step}/{max_steps_local} in chat {chat_id}"
            )

            # 1. Prepare for LLM call (same as non-streaming)
            history_for_model = thread.get_history()
            tools_for_model = await self.get_all_tools()
            tool_schemas = [t.openai_schema for t in tools_for_model if hasattr(t, "openai_schema")]
            model_to_use = self.model or "gpt-4o"

            # 2. Call LLM (Streaming)
            accumulated_content = ""
            accumulated_tool_calls = []
            current_tool_call_chunks = {}  # {tool_call_id: {"name": ..., "args": ...}}
            try:
                logger.debug(f"Calling LLM stream ({model_to_use}) for agent '{self.name}'")
                stream = await self.client.chat.completions.create(
                    model=model_to_use,
                    messages=history_for_model,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto",
                    stream=True,
                )

                # Process the stream chunks
                async for chunk in stream:
                    yield chunk  # Yield raw chunk for maximum compatibility?
                    # Or process into RunItemStreamEvent?
                    # Let's yield raw for now.

                    # --- Accumulate content and tool calls from chunks ---
                    # This logic is complex and depends heavily on the chunk structure
                    # provided by the openai client when streaming tool calls.
                    # Placeholder logic - needs refinement based on actual stream format.
                    delta = chunk.choices[0].delta
                    if delta.content:
                        accumulated_content += delta.content
                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            idx = tool_call_chunk.index
                            call_id = tool_call_chunk.id
                            if call_id:
                                if call_id not in current_tool_call_chunks:
                                    current_tool_call_chunks[call_id] = {
                                        "id": call_id,
                                        "index": idx,
                                        "name": "",
                                        "args": "",
                                    }
                                if tool_call_chunk.function:
                                    if tool_call_chunk.function.name:
                                        current_tool_call_chunks[call_id]["name"] += (
                                            tool_call_chunk.function.name
                                        )
                                    if tool_call_chunk.function.arguments:
                                        current_tool_call_chunks[call_id]["args"] += (
                                            tool_call_chunk.function.arguments
                                        )
                                # TODO: Handle other tool types if necessary

                # Reconstruct full response items after stream ends
                llm_output_items: List[RunItem] = []
                if accumulated_content:
                    text_item = MessageOutputItem(content=accumulated_content)
                    llm_output_items.append(text_item)

                for call_id, chunk_data in current_tool_call_chunks.items():
                    # Attempt to parse args JSON
                    try:
                        args_json = json.loads(chunk_data["args"]) if chunk_data["args"] else {}
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not decode tool args JSON for call {call_id}: {chunk_data['args']}"
                        )
                        args_json = {"raw_args": chunk_data["args"]}  # Fallback

                    tool_call_item = ToolCallItem(
                        id=call_id,
                        name=chunk_data["name"],
                        args_raw=chunk_data["args"],
                        args_json=args_json,
                    )
                    llm_output_items.append(tool_call_item)
                    accumulated_tool_calls.append(tool_call_item)

                # Save reconstructed items to thread
                if llm_output_items:
                    thread.add_items(llm_output_items)
                    self._thread_manager.add_items_and_save(thread, llm_output_items)
                    run_items.extend(llm_output_items)
            except Exception as e:
                logger.error(
                    f"Error calling or processing LLM stream for '{self.name}': {e}", exc_info=True
                )
                yield {"error": str(e)}
                raise AgentsException(f"LLM stream failed for agent {self.name}") from e

            # Now process tool calls outside the try block
            send_message_call: Optional[ToolCallItem] = None
            standard_tool_calls: List[ToolCallItem] = []  # Start with empty list

            # Populate standard_tool_calls from accumulated calls
            standard_tool_calls = accumulated_tool_calls

            # Check if SendMessage was called
            send_message_call = next(
                (tc for tc in standard_tool_calls if tc.name == SEND_MESSAGE_TOOL_NAME), None
            )
            if send_message_call:
                # Remove SendMessage from standard calls if found
                standard_tool_calls = [
                    tc for tc in standard_tool_calls if tc.name != SEND_MESSAGE_TOOL_NAME
                ]

            # --- Handle SendMessage --- (Correct Indentation)
            if send_message_call:
                logger.info(f"Agent '{self.name}' STREAM initiating SendMessage.")
                args = send_message_call.args_json
                recipient_name = args.get("recipient")
                message_content = args.get("message")

                # Combine validation checks
                is_allowed = self._agency_instance.is_communication_allowed(
                    self.name, recipient_name
                )
                recipient_agent = self._agency_instance.agents.get(recipient_name)

                if (
                    not recipient_name
                    or message_content is None
                    or not is_allowed
                    or not recipient_agent
                ):
                    # Construct specific error message
                    if not recipient_name or message_content is None:
                        error_msg = f"SendMessage failed: Missing recipient ('{recipient_name}') or message in args: {args}"
                    elif not is_allowed:
                        error_msg = f"Communication DENIED: Agent '{self.name}' cannot send to '{recipient_name}'."
                    else:  # not recipient_agent
                        error_msg = (
                            f"SendMessage failed: Recipient agent '{recipient_name}' not found."
                        )

                    logger.error(error_msg)
                    tool_output_item = ToolCallOutputItem(
                        tool_call_id=send_message_call.id,
                        name=send_message_call.name,
                        content=error_msg,
                        is_error=True,
                    )
                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    # Continue loop after failed SendMessage
                    continue
                else:
                    # --- Recursive Streaming Call ---
                    logger.info(f"Agent '{self.name}' STREAM calling agent '{recipient_name}'.")
                    try:
                        output_content = ""  # Reset accumulated content for sub-stream
                        async for sub_chunk in recipient_agent.get_response_stream(
                            message=message_content,
                            chat_id=thread.thread_id,
                            sender_name=self.name,
                            max_steps=max_steps_local - current_step,
                            # TODO: Pass depth
                        ):
                            yield sub_chunk  # Pass through sub-agent's stream
                            # Accumulate final text from sub-agent if needed
                            if isinstance(sub_chunk, dict):  # Example check for OpenAI-like chunk
                                choices = sub_chunk.get("choices")
                                if (
                                    choices
                                    and isinstance(choices, list)
                                    and len(choices) > 0
                                    and choices[0].get("delta")
                                ):
                                    delta_content = choices[0]["delta"].get("content")
                                    if delta_content:
                                        output_content += delta_content
                            elif isinstance(sub_chunk, str):  # Simplistic fallback
                                output_content += sub_chunk

                        if not output_content:
                            output_content = "(No text output from recipient stream)"

                        tool_output_item = ToolCallOutputItem(
                            tool_call_id=send_message_call.id,
                            name=send_message_call.name,
                            content=output_content,
                        )
                        logger.info(f"SendMessage STREAM call to '{recipient_name}' completed.")
                    except Exception as e_recurse:
                        error_msg = (
                            f"Error during recursive stream call to '{recipient_name}': {e_recurse}"
                        )
                        logger.error(error_msg, exc_info=True)
                        tool_output_item = ToolCallOutputItem(
                            tool_call_id=send_message_call.id,
                            name=send_message_call.name,
                            content=error_msg,
                            is_error=True,
                        )

                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    # After successful/failed SendMessage handling, continue the loop
                    continue

            # --- Handle Standard Tool Calls --- (Correct Indentation)
            elif standard_tool_calls:
                logger.info(
                    f"Agent '{self.name}' STREAM executing standard tools: {[tc.name for tc in standard_tool_calls]}"
                )
                tool_outputs_generated = False
                for tool_call in standard_tool_calls:
                    tool_to_execute = next(
                        (
                            t
                            for t in tools_for_model
                            if hasattr(t, "name") and t.name == tool_call.name
                        ),
                        None,
                    )
                    output_content = ""
                    is_error = False

                    if not tool_to_execute or not hasattr(tool_to_execute, "on_invoke_tool"):
                        error_msg = f"Tool '{tool_call.name}' not found or not executable."
                        logger.error(error_msg)
                        output_content = error_msg
                        is_error = True
                    else:
                        try:
                            logger.debug(
                                f"Executing tool: {tool_call.name} with args: {tool_call.args_json}"
                            )
                            if asyncio.iscoroutinefunction(tool_to_execute.on_invoke_tool):
                                output_content = await tool_to_execute.on_invoke_tool(
                                    None, tool_call.args_json
                                )
                            else:
                                output_content = tool_to_execute.on_invoke_tool(
                                    None, tool_call.args_json
                                )
                            is_error = False
                            logger.debug(f"Tool '{tool_call.name}' execution successful.")
                        except Exception as e_tool:
                            error_msg = f"Error executing tool '{tool_call.name}': {e_tool}"
                            logger.error(error_msg, exc_info=True)
                            output_content = error_msg
                            is_error = True

                    tool_output_item = ToolCallOutputItem(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=str(output_content),
                        is_error=is_error,
                    )
                    thread.add_item(tool_output_item)
                    self._thread_manager.add_item_and_save(thread, tool_output_item)
                    run_items.append(tool_output_item)
                    tool_outputs_generated = True

                # If standard tools were executed, continue the loop to process their output
                if tool_outputs_generated:
                    continue

            # --- No Tool Calls were made by LLM this step --- (Correct Indentation)
            else:
                if accumulated_content:
                    logger.info(
                        f"Agent '{self.name}' STREAM produced final text output. Ending loop."
                    )
                    break  # Exit the while loop
                else:
                    logger.warning(
                        f"Agent '{self.name}' STREAM finished step {current_step} with no text or tool calls. Breaking loop."
                    )
                    break  # Exit the while loop

        # --- End of Loop ---
        logger.info(f"Agent '{self.name}' finished get_response_stream for chat_id: {chat_id}")
        # TODO: Final validation for streaming? Difficult.
        # TODO: Yield a final RunResult summary event?
        # yield {"event": "run_complete", "final_result": ...} # Example
        # --- End: Streaming Orchestration Logic ---

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

    def _set_thread_manager(self, manager: ThreadManager):
        """Allows the Agency to inject the ThreadManager instance."""
        self._thread_manager = manager

    def _set_agency_peers(self, peers: List[str]):
        """Allows the Agency to set the list of allowed communication peers."""
        self._agency_chart_peers = peers

    def _set_agency_instance(self, agency: Any):
        """Allows the Agency to inject a reference to itself."""
        self._agency_instance = agency
