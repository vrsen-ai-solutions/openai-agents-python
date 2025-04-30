from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from agents import RunItem, RunResult, TResponseInputItem
from agents.items import ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem

if TYPE_CHECKING:
    from .agent import Agent  # Use forward reference

logger = logging.getLogger(__name__)


@dataclass
class ConversationThread:
    """Manages the history of a multi-agent conversation.
    Does NOT manage execution state (handled by program call stack).
    """

    thread_id: str = field(default_factory=lambda: f"as_thread_{uuid.uuid4()}")
    items: List[RunItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_item(self, item: RunItem) -> None:
        """Appends a RunItem (message, tool call, etc.) to the history."""
        if not isinstance(item, RunItem):
            logger.warning(
                f"Attempted to add non-RunItem type {type(item)} to thread {self.thread_id}"
            )
            return
        self.items.append(item)
        logger.debug(f"Added item {type(item).__name__} to thread {self.thread_id}")

    def add_items(self, items: Sequence[RunItem]) -> None:
        """Appends multiple RunItems to the history."""
        added_count = 0
        for item in items:
            if isinstance(item, RunItem):
                self.items.append(item)
                added_count += 1
            else:
                logger.warning(
                    f"Skipping non-RunItem type {type(item)} during add_items in thread {self.thread_id}"
                )
        logger.debug(f"Added {added_count} items to thread {self.thread_id}")

    def add_run_result(self, result: RunResult) -> None:
        """Adds all items from an agent's RunResult to the thread history."""
        if not hasattr(result, "items") or not result.items:
            logger.warning(
                f"RunResult provided to add_run_result in thread {self.thread_id} has no items."
            )
            return

        if not isinstance(result.items, Sequence):
            logger.error(
                f"RunResult.items is not a sequence (type: {type(result.items)}) in thread {self.thread_id}. Cannot add items."
            )
            return

        self.add_items(result.items)
        logger.info(f"Added {len(result.items)} items from RunResult to thread {self.thread_id}")

    def add_user_message(self, message: Union[str, TResponseInputItem]) -> None:
        """Adds a user message to the thread history."""
        item_to_add = None
        if isinstance(message, str):
            item_to_add = UserInputItem(content=message)
        elif isinstance(message, UserInputItem):
            item_to_add = message
        elif isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                item_to_add = UserInputItem(content=content)
            else:
                logger.error(
                    f"User message dict has invalid content type: {type(content)} in thread {self.thread_id}"
                )
                raise TypeError(f"Invalid content type in user message dict: {type(content)}")
        else:
            logger.error(
                f"Invalid type for add_user_message: {type(message)} in thread {self.thread_id}"
            )
            raise TypeError(
                f"Unsupported message type for add_user_message: {type(message)}. Expecting str or UserInputItem."
            )

        self.add_item(item_to_add)
        logger.info(f"Added user message to thread {self.thread_id}")

    def _convert_item_to_openai_format(self, item: RunItem) -> Optional[Dict[str, Any]]:
        """Converts a single RunItem to the OpenAI API message dictionary format."""
        if isinstance(item, UserInputItem):
            return {"role": "user", "content": item.content}
        elif isinstance(item, MessageOutputItem):
            if item.content:
                if isinstance(item.content, list):
                    return {"role": "assistant", "content": item.content}
                elif isinstance(item.content, str):
                    return {"role": "assistant", "content": item.content}
                else:
                    logger.warning(
                        f"MessageOutputItem content is not string or list: {type(item.content)}"
                    )
                    return {
                        "role": "assistant",
                        "content": str(item.content),
                    }
            else:
                return None
        elif isinstance(item, ToolCallItem):
            try:
                args_str = (
                    json.dumps(item.args_json)
                    if isinstance(item.args_json, dict)
                    else str(item.args_json)
                )
            except (TypeError, ValueError) as e:
                logger.error(
                    f"Could not serialize tool call args for {item.name} (id: {item.id}): {e}"
                )
                args_str = str(item.args_json)

            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": args_str},
                    }
                ],
            }
        elif isinstance(item, ToolCallOutputItem):
            return {
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "name": item.name,
                "content": item.content,
            }
        else:
            logger.warning(
                f"Unrecognized RunItem type for OpenAI conversion: {type(item)} in thread {self.thread_id}"
            )
            return None

    def get_history(
        self,
        perspective_agent: Optional["Agent"] = None,
        max_items: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[TResponseInputItem]:
        """Gets the history formatted for the OpenAI API (or compatible provider).
        Args:
            perspective_agent: The agent for whom the history is being prepared.
                               (Future use for filtering based on agency_chart).
            max_items: Optional limit on the number of recent items to include.
            max_tokens: Optional approximate token limit for the history.

        Returns:
            A list of items formatted for the model provider.
        """
        selected_items = self.items
        if max_items is not None and len(selected_items) > max_items:
            logger.debug(
                f"Truncating history to last {max_items} items for thread {self.thread_id}"
            )
            selected_items = selected_items[-max_items:]

        formatted_history: List[TResponseInputItem] = []
        current_tokens = 0

        for item in reversed(selected_items):
            converted_item = self._convert_item_to_openai_format(item)
            if not converted_item:
                continue

            item_tokens = len(json.dumps(converted_item)) // 3

            if max_tokens is not None and (current_tokens + item_tokens > max_tokens):
                if not formatted_history:
                    formatted_history.insert(0, converted_item)
                    current_tokens += item_tokens
                    logger.warning(
                        f"First item added already exceeds token limit ({item_tokens}/{max_tokens}) in thread {self.thread_id}"
                    )
                else:
                    logger.debug(
                        f"Token limit ({max_tokens}) reached. Stopping history inclusion for thread {self.thread_id}."
                    )
                    break
            else:
                formatted_history.insert(0, converted_item)
                current_tokens += item_tokens

        logger.debug(
            f"Generated history with {len(formatted_history)} items, estimated {current_tokens} tokens for thread {self.thread_id}"
        )
        return formatted_history

    def get_full_log(self) -> List[RunItem]:
        """Returns the complete, raw list of RunItems."""
        return list(self.items)

    def __len__(self) -> int:
        """Returns the number of items in the thread history."""
        return len(self.items)

    def clear(self) -> None:
        """Clears the history of the thread."""
        self.items.clear()
        logger.info(f"Cleared items from thread {self.thread_id}")


# Placeholder imports for callbacks
ThreadLoadCallback = Callable[[str], Optional[ConversationThread]]
ThreadSaveCallback = Callable[[ConversationThread], None]


class ThreadManager:
    """Manages multiple ConversationThreads and persistence."""

    _threads: Dict[str, ConversationThread]  # In-memory storage for now
    _load_callback: Optional[ThreadLoadCallback]
    _save_callback: Optional[ThreadSaveCallback]

    def __init__(
        self,
        load_callback: Optional[ThreadLoadCallback] = None,
        save_callback: Optional[ThreadSaveCallback] = None,
    ):
        self._threads = {}
        self._load_callback = load_callback
        self._save_callback = save_callback
        logger.info("ThreadManager initialized.")

    def get_thread(self, thread_id: Optional[str] = None) -> ConversationThread:
        """Retrieves or creates a ConversationThread.
        Handles loading from persistence if callback is provided.
        """
        if thread_id and thread_id in self._threads:
            logger.debug(f"Returning existing thread {thread_id} from memory.")
            return self._threads[thread_id]

        if thread_id and self._load_callback:
            logger.debug(f"Attempting to load thread {thread_id} using callback...")
            loaded_thread = self._load_callback(thread_id)
            if loaded_thread:
                logger.info(f"Successfully loaded thread {thread_id} from persistence.")
                self._threads[thread_id] = loaded_thread
                return loaded_thread
            else:
                logger.warning(f"Load callback provided but failed to load thread {thread_id}.")
                # Fall through to create a new thread

        # If no ID provided, or not in memory, or loading failed, create new
        new_thread = ConversationThread(thread_id=thread_id)  # Pass ID if provided
        actual_id = new_thread.thread_id  # Get potentially generated ID
        self._threads[actual_id] = new_thread
        logger.info(f"Created new thread: {actual_id}. Storing in memory.")
        # Optionally save the newly created thread immediately
        if self._save_callback:
            self._save_thread(new_thread)
        return new_thread

    def add_item_and_save(self, thread: ConversationThread, item: RunItem):
        """Adds an item to the thread and triggers the save callback if configured."""
        thread.add_item(item)
        if self._save_callback:
            self._save_thread(thread)

    def add_items_and_save(self, thread: ConversationThread, items: Sequence[RunItem]):
        """Adds multiple items to the thread and triggers the save callback if configured."""
        thread.add_items(items)
        if self._save_callback:
            self._save_thread(thread)

    def _save_thread(self, thread: ConversationThread):
        """Internal method to save a thread using the callback."""
        if self._save_callback:
            try:
                logger.debug(f"Saving thread {thread.thread_id} using callback...")
                self._save_callback(thread)
                logger.info(f"Successfully saved thread {thread.thread_id}.")
            except Exception as e:
                logger.error(
                    f"Error saving thread {thread.thread_id} using callback: {e}", exc_info=True
                )
                # Decide if this should raise an error or just log

    def delete_thread(self, thread_id: str) -> bool:
        """Deletes a thread from memory.
        TODO: Implement deletion from persistence if needed.
        """
        if thread_id in self._threads:
            del self._threads[thread_id]
            logger.info(f"Deleted thread {thread_id} from memory.")
            # TODO: Call persistence deletion callback if available
            return True
        else:
            logger.warning(f"Attempted to delete non-existent thread {thread_id} from memory.")
            return False


# --- Helper Functions (Consider moving to a utils module) ---
