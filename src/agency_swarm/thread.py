import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from agents import RunResult, TResponseInputItem

if TYPE_CHECKING:
    from .agent import Agent  # Use forward reference

logger = logging.getLogger(__name__)


@dataclass
class ConversationThread:
    """Manages the history of a multi-agent conversation.
    Does NOT manage execution state (handled by program call stack).
    """

    thread_id: str = field(default_factory=lambda: f"as_thread_{uuid.uuid4()}")
    # Store TResponseInputItem dictionaries directly
    items: List[TResponseInputItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_item(self, item: TResponseInputItem) -> None:
        """Appends a TResponseInputItem dictionary to the history."""
        # Basic validation: ensure it's a dict with a 'role'
        if not isinstance(item, dict) or "role" not in item:
            logger.warning(
                f"Attempted to add non-TResponseInputItem-like dict {type(item)} to thread {self.thread_id}"
            )
            return
        self.items.append(item)
        logger.debug(f"Added item with role '{item.get('role')}' to thread {self.thread_id}")

    def add_items(self, items: Sequence[TResponseInputItem]) -> None:
        """Appends multiple TResponseInputItem dictionaries to the history."""
        added_count = 0
        for item in items:
            # Basic validation: ensure it's a dict with a 'role'
            if isinstance(item, dict) and "role" in item:
                self.items.append(item)
                added_count += 1
            else:
                logger.warning(
                    f"Skipping non-TResponseInputItem-like dict {type(item)} during add_items in thread {self.thread_id}"
                )
        logger.debug(f"Added {added_count} items to thread {self.thread_id}")

    def add_run_result(self, result: RunResult) -> None:
        """Adds all items from an agent's RunResult to the thread history.
        Converts RunItems to TResponseInputItem dictionaries before adding.
        """
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

        # Convert RunItems to TResponseInputItem before adding
        items_to_add: List[TResponseInputItem] = []
        for run_item in result.items:
            if hasattr(run_item, "raw_item") and isinstance(run_item.raw_item, dict):
                if run_item.raw_item.get("role") in ["user", "assistant", "tool"]:
                    items_to_add.append(run_item.raw_item)
                else:
                    logger.warning(
                        f"Skipping RunItem with invalid role in raw_item: {run_item.raw_item.get('role')}"
                    )
            else:
                logger.warning(f"Skipping RunItem without a valid raw_item dict: {type(run_item)}")

        self.add_items(items_to_add)
        logger.info(f"Added {len(items_to_add)} items from RunResult to thread {self.thread_id}")

    def add_user_message(self, message: Union[str, TResponseInputItem]) -> None:
        """Adds a user message to the thread history as a TResponseInputItem dict."""
        item_dict: TResponseInputItem
        if isinstance(message, str):
            item_dict = {"role": "user", "content": message}
        elif isinstance(message, dict) and message.get("role") == "user":
            if "content" not in message:
                logger.error(f"User message dict missing 'content' key in thread {self.thread_id}")
                raise ValueError("User message dict must have a 'content' key.")
            item_dict = message
        else:
            logger.error(
                f"Invalid type for add_user_message: {type(message)} in thread {self.thread_id}"
            )
            raise TypeError(
                f"Unsupported message type for add_user_message: {type(message)}. Expecting str or TResponseInputItem dict."
            )
        # Add the dictionary directly
        self.add_item(item_dict)
        logger.info(f"Added user message to thread {self.thread_id}")

    def get_history(
        self,
        perspective_agent: Optional["Agent"] = None,  # Kept for future use
        max_items: Optional[int] = None,
    ) -> List[TResponseInputItem]:
        """Gets the history as a list of TResponseInputItem dictionaries suitable for the Runner.
        Args:
            perspective_agent: The agent for whom the history is being prepared.
                               (Future use for filtering).
            max_items: Optional limit on the number of recent items to include.

        Returns:
            A list of TResponseInputItem dictionaries.
        """
        selected_items = self.items
        if max_items is not None and len(selected_items) > max_items:
            logger.debug(
                f"Truncating history to last {max_items} items for thread {self.thread_id}"
            )
            selected_items = selected_items[-max_items:]

        # History is already List[TResponseInputItem]
        formatted_history: List[TResponseInputItem] = list(selected_items)

        logger.debug(
            f"Generated history with {len(formatted_history)} items for thread {self.thread_id}"
        )
        return formatted_history

    def get_full_log(self) -> List[TResponseInputItem]:  # Return type changed
        """Returns the complete, raw list of TResponseInputItem dictionaries."""
        return list(self.items)

    def __len__(self) -> int:
        """Returns the number of items in the thread history."""
        return len(self.items)

    def clear(self) -> None:
        """Clears the history of the thread."""
        self.items.clear()
        logger.info(f"Cleared items from thread {self.thread_id}")


# Placeholder imports for callbacks - Update Typehint
ThreadLoadCallback = Callable[[str], Optional[ConversationThread]]
# Save callback expects the full dictionary of threads
ThreadSaveCallback = Callable[[Dict[str, ConversationThread]], None]


class ThreadManager:
    """Manages multiple ConversationThreads and persistence."""

    _threads: Dict[str, ConversationThread]  # In-memory storage
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
        # Fix 1: Explicitly check type if thread_id is provided
        if thread_id is not None and not isinstance(thread_id, str):
            raise TypeError(f"thread_id must be a string or None, not {type(thread_id)}")

        effective_thread_id = thread_id

        # Fix 2: Generate ID here if None
        if effective_thread_id is None:
            effective_thread_id = f"as_thread_{uuid.uuid4()}"
            logger.info(f"No thread_id provided, generated new ID: {effective_thread_id}")

        if effective_thread_id in self._threads:
            logger.debug(f"Returning existing thread {effective_thread_id} from memory.")
            return self._threads[effective_thread_id]

        # Load attempt uses effective_thread_id (which might be the generated one)
        if self._load_callback:
            logger.debug(f"Attempting to load thread {effective_thread_id} using callback...")
            # Assuming load callback should only be called if an ID was initially provided
            # or if we expect a specific generated ID format might exist.
            # Let's refine: Only attempt load if thread_id was explicitly provided.
            if thread_id is not None:
                loaded_thread = self._load_callback(thread_id)  # Use original provided id for load
                if loaded_thread:
                    logger.info(f"Successfully loaded thread {thread_id} from persistence.")
                    # Ensure the loaded thread uses the requested ID
                    if loaded_thread.thread_id != thread_id:
                        logger.warning(
                            f"Loaded thread ID '{loaded_thread.thread_id}' differs from requested ID '{thread_id}'. Using requested ID."
                        )
                        loaded_thread.thread_id = thread_id
                    self._threads[thread_id] = loaded_thread
                    return loaded_thread
                else:
                    logger.warning(f"Load callback provided but failed to load thread {thread_id}.")
            else:
                logger.debug("Skipping load callback as no specific thread_id was requested.")

        # Create new thread using the effective_thread_id (original or generated)
        new_thread = ConversationThread(thread_id=effective_thread_id)
        # actual_id = new_thread.thread_id # Not needed anymore as we use effective_thread_id
        self._threads[effective_thread_id] = new_thread
        logger.info(f"Created new thread: {effective_thread_id}. Storing in memory.")
        if self._save_callback:
            # Save only if it was newly created (i.e., not loaded)
            self._save_thread(new_thread)
        return new_thread

    def add_item_and_save(self, thread: ConversationThread, item: TResponseInputItem):
        """Adds an item to the thread and triggers the save callback if configured."""
        thread.add_item(item)
        if self._save_callback:
            self._save_thread(thread)

    def add_items_and_save(self, thread: ConversationThread, items: Sequence[TResponseInputItem]):
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
