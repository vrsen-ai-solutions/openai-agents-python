from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

# --- SDK Imports ---
from agents import RunHooks, RunResult

# --- Agency Swarm Imports ---
from .context import MasterContext
from .thread import ConversationThread

logger = logging.getLogger(__name__)

# --- Callback Type Hints ---
# Load callback expects a thread ID string, returns Optional[ConversationThread]
ThreadLoadCallback = Callable[[str], Optional[ConversationThread]]
# Save callback expects the full dictionary of threads
ThreadSaveCallback = Callable[[Dict[str, ConversationThread]], None]


# --- Persistence Hooks ---
class PersistenceHooks(RunHooks[MasterContext]):  # Context type is MasterContext
    """Custom RunHooks implementation for loading/saving ThreadManager state."""

    def __init__(self, load_callback: ThreadLoadCallback, save_callback: ThreadSaveCallback):
        if not callable(load_callback) or not callable(save_callback):
            raise TypeError("load_callback and save_callback must be callable.")
        self._load_callback = load_callback
        self._save_callback = save_callback
        logger.info("PersistenceHooks initialized.")

    def on_run_start(self, *, context: MasterContext, **kwargs) -> None:
        """Loads threads into the ThreadManager at the start of a run."""
        logger.debug("PersistenceHooks: on_run_start triggered.")
        try:
            loaded_threads = self._load_callback()
            if loaded_threads:
                if isinstance(loaded_threads, dict):
                    # Assume callback returns Dict[str, ConversationThread]
                    context.thread_manager._threads = loaded_threads
                    logger.info(f"Loaded {len(loaded_threads)} threads via load_callback.")
                else:
                    logger.error(f"load_callback returned unexpected type: {type(loaded_threads)}")
            else:
                logger.info("load_callback returned no threads to load.")
        except Exception as e:
            logger.error(f"Error during load_callback execution: {e}", exc_info=True)
            # Decide if this should halt execution?
            # For now, log and continue with potentially empty threads.

    def on_run_end(self, *, context: MasterContext, result: RunResult, **kwargs) -> None:
        """Saves threads from the ThreadManager at the end of a run."""
        logger.debug("PersistenceHooks: on_run_end triggered.")
        try:
            self._save_callback(context.thread_manager._threads)
            logger.info("Saved threads via save_callback.")
        except Exception as e:
            logger.error(f"Error during save_callback execution: {e}", exc_info=True)
            # Log error but don't prevent run completion.
