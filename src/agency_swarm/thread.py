from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from agents import RunItem, RunResult, TResponseInputItem
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem, UserInputItem

if TYPE_CHECKING:
    from .agent import Agent  # Use forward reference

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnState:
    """Represents the state of a single agent's turn within the recursive flow."""

    agent: "Agent"
    caller_agent: Optional["Agent"] = None  # Agent that initiated this turn via SendMessage
    # TODO: Add other relevant state, e.g., tool_call_id that triggered SendMessage
    # triggering_tool_call_id: Optional[str] = None


@dataclass
class ConversationThread:
    """Manages the history and execution state of a multi-agent conversation."""

    thread_id: str = field(default_factory=lambda: f"as_thread_{uuid.uuid4()}")
    items: List[RunItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _execution_stack: List[AgentTurnState] = field(
        default_factory=list, init=False, repr=False
    )  # Manages recursive calls

    def add_item(self, item: RunItem) -> None:
        """Appends a RunItem (message, tool call, etc.) to the history."""
        if not isinstance(item, RunItem):
            # Basic type check, might need refinement based on RunItem definition
            logger.warning(
                f"Attempted to add non-RunItem type {type(item)} to thread {self.thread_id}"
            )
            # Optionally raise TypeError
            # raise TypeError(f"Expected RunItem, got {type(item)}")
            return
        self.items.append(item)
        logger.debug(f"Added item {type(item).__name__} to thread {self.thread_id}")
        # TODO: Add persistence callback logic here if needed

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
        # TODO: Add persistence callback logic here if needed

    def add_run_result(self, result: RunResult) -> None:
        """Adds all items from an agent's RunResult to the thread history."""
        if not hasattr(result, "items") or not result.items:
            logger.warning(
                f"RunResult provided to add_run_result in thread {self.thread_id} has no items."
            )
            return

        # Check if items is a sequence before iterating
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
            # If it's already the correct type, use it directly
            item_to_add = message
        elif isinstance(message, dict) and message.get("role") == "user":
            # Handle simple dict format if needed, though TResponseInputItem is preferred
            content = message.get("content")
            if isinstance(content, str):
                item_to_add = UserInputItem(content=content)
            else:
                logger.error(
                    f"User message dict has invalid content type: {type(content)} in thread {self.thread_id}"
                )
                raise TypeError(f"Invalid content type in user message dict: {type(content)}")
        # TODO: Handle other TResponseInputItem types if they represent user input?
        # For now, only UserInputItem or string is directly supported.
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
            # Handle potential tool calls associated with assistant message
            # The SDK might represent this differently (e.g., RunResult bundles them),
            # but if MessageOutputItem itself has tool_calls, include them.
            # For now, assume MessageOutputItem primarily holds text content.
            if item.content:
                # Check if item.content is already a list (e.g. multi-modal)
                if isinstance(item.content, list):
                    return {"role": "assistant", "content": item.content}
                elif isinstance(item.content, str):
                    return {"role": "assistant", "content": item.content}
                else:
                    logger.warning(
                        f"MessageOutputItem content is not string or list: {type(item.content)}"
                    )
                    # Decide how to handle non-text/list content (e.g., skip, convert to string)
                    return {
                        "role": "assistant",
                        "content": str(item.content),
                    }  # Fallback to string conversion
            else:
                # If content is empty, it might be just a container for tool calls
                # Handled by ToolCallItem below. Return None here.
                return None
        elif isinstance(item, ToolCallItem):
            # Convert tool call parameters to JSON string if they are not already
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
                args_str = str(item.args_json)  # Fallback

            return {
                "role": "assistant",
                "content": None,  # Standard practice for tool call messages
                "tool_calls": [
                    {
                        "id": item.id,
                        "type": "function",  # Assuming all tools are functions for now
                        "function": {"name": item.name, "arguments": args_str},
                    }
                ],
            }
        elif isinstance(item, ToolCallOutputItem):
            # Tool results need to be associated with their call via ID
            return {
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "name": item.name,  # Often included for clarity, matches ToolCallItem name
                "content": item.content,  # Content is the output of the tool
            }
        else:
            logger.warning(
                f"Unrecognized RunItem type for OpenAI conversion: {type(item)} in thread {self.thread_id}"
            )
            return None  # Skip unknown types

    def get_history(
        self,
        perspective_agent: Optional["Agent"] = None,
        max_items: Optional[int] = None,
        max_tokens: Optional[int] = None,  # Added max_tokens based on plan detail
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

        # --- History Selection ---
        selected_items = self.items
        if max_items is not None and len(selected_items) > max_items:
            logger.debug(
                f"Truncating history to last {max_items} items for thread {self.thread_id}"
            )
            selected_items = selected_items[-max_items:]

        # TODO: Implement filtering logic based on perspective_agent and agency_chart

        # --- Formatting & Token Limiting ---
        formatted_history: List[TResponseInputItem] = []
        current_tokens = 0

        # Process in reverse to keep recent items if token limit is hit
        for item in reversed(selected_items):
            converted_item = self._convert_item_to_openai_format(item)
            if not converted_item:
                continue  # Skip items that couldn't be converted or are None

            # Estimate tokens (very basic estimation)
            # TODO: Replace with a proper tokenizer (e.g., tiktoken) if accuracy is needed
            item_tokens = len(json.dumps(converted_item)) // 3  # Rough estimate

            if max_tokens is not None and (current_tokens + item_tokens > max_tokens):
                # Only add if it fits, unless history is currently empty
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
                    break  # Stop adding items
            else:
                formatted_history.insert(0, converted_item)  # Insert at beginning
                current_tokens += item_tokens

        # TODO: Implement summarization if needed when truncated due to tokens

        logger.debug(
            f"Generated history with {len(formatted_history)} items, estimated {current_tokens} tokens for thread {self.thread_id}"
        )
        return formatted_history

    def get_full_log(self) -> List[RunItem]:
        """Returns the complete, raw list of RunItems."""
        # Return a copy to prevent external modification
        return list(self.items)

    # --- State Management for Recursive Calls ---

    def push_turn(self, current_agent: "Agent", caller: Optional["Agent"] = None) -> None:
        """Pushes the current agent's state onto the execution stack.
           Called by the Agency orchestrator before invoking an agent, especially
           when handling a SendMessage call.

        Args:
            current_agent: The agent whose turn is starting.
            caller: The agent that initiated this turn via SendMessage, if applicable.
        """
        state = AgentTurnState(agent=current_agent, caller_agent=caller)
        self._execution_stack.append(state)

    def pop_turn(self) -> Optional[AgentTurnState]:
        """Pops the latest agent state from the stack.
           Called by the Agency orchestrator when an agent's turn (or SendMessage call) completes.

        Returns:
            The state of the turn that just ended, or None if the stack was empty.
        """
        if self._execution_stack:
            return self._execution_stack.pop()
        return None

    def current_turn_state(self) -> Optional[AgentTurnState]:
        """Gets the state of the currently executing agent turn (top of the stack)."""
        if self._execution_stack:
            return self._execution_stack[-1]
        return None

    def get_caller(self) -> Optional["Agent"]:
        """Gets the agent that initiated the current turn via SendMessage, if any."""
        state = self.current_turn_state()
        return state.caller_agent if state else None

    def __len__(self) -> int:
        """Returns the number of items in the thread history."""
        return len(self.items)

    def clear(self) -> None:
        """Clears the history and execution stack of the thread."""
        self.items.clear()
        self._execution_stack.clear()
        # Consider clearing metadata or not?
