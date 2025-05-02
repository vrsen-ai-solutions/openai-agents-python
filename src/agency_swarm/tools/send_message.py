import logging
from typing import TYPE_CHECKING

from agents import FunctionTool, RunContextWrapper, RunResult, function_tool

from ..context import MasterContext

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)
SEND_MESSAGE_TOOL_NAME = "SendMessage"


# --- Tool Definition using @function_tool --- #
@function_tool(
    name_override=SEND_MESSAGE_TOOL_NAME,
    description_override=(
        "Send a message to another agent to delegate tasks or share information. "
        "Use this when you need another agent's expertise or action to proceed."
    ),
)
async def send_message(
    wrapper: RunContextWrapper[MasterContext],
    recipient: str,  # Description inferred from docstring
    message: str,  # Description inferred from docstring
) -> str:
    """Sends a message to a specified recipient agent.

    Args:
        wrapper: The run context wrapper providing access to agent and shared context.
        recipient: The exact name of the recipient agent that is registered.
        message: The message content to send to the recipient agent.
    """
    # @custom_span("SendMessageTool.on_invoke_tool") # Keep tracing commented out for now
    master_context: MasterContext = wrapper.context
    # hooks = wrapper.hooks # Hooks are not available directly in the tool wrapper

    # Get current agent name and agency agent map from context
    current_agent_name = master_context.current_agent_name
    agency_agents = master_context.agents

    if not current_agent_name or current_agent_name not in agency_agents:
        logger.error(f"Could not determine current agent ('{current_agent_name}') from context.")
        return "Error: Internal configuration error. Could not determine sending agent."

    current_agent: Agent = agency_agents[current_agent_name]

    recipient_name = recipient
    message_content = message
    current_chat_id = master_context.chat_id  # Get chat_id from context

    logger.info(
        f"Agent '{current_agent_name}' invoking {SEND_MESSAGE_TOOL_NAME} tool. "
        f"Recipient: '{recipient_name}', Message: \"{message_content[:50]}...\""
    )

    # --- Validation (using current_agent instance obtained from context) ---
    if not hasattr(current_agent, "_subagents"):
        logger.error(f"Current agent '{current_agent_name}' lacks '_subagents' attribute.")
        return "Error: Internal configuration error. Agent cannot determine valid recipients."

    if recipient_name not in current_agent._subagents:
        valid_recipients = ", ".join(current_agent._subagents.keys())
        logger.warning(
            f"Invalid recipient '{recipient_name}' specified by agent '{current_agent_name}'. "
            f"Valid recipients: {valid_recipients}"
        )
        return (
            f"Error: Invalid recipient '{recipient_name}'. Valid recipients are: {valid_recipients}"
        )

    # Find target agent in the agency map from context
    if recipient_name not in agency_agents:
        logger.error(f"Recipient agent '{recipient_name}' not found in master context agents map.")
        return f"Error: Recipient agent '{recipient_name}' not found in the agency."

    target_agent: Agent = agency_agents[recipient_name]

    # --- Call Target Agent --- (Using target_agent.get_response as per PRD)
    try:
        logger.debug(f"Calling target agent '{target_agent.name}'.get_response...")
        # Pass down the master context. Hooks will be handled by the Runner for the sub-call.
        sub_result: RunResult = await target_agent.get_response(
            message=message_content,
            sender_name=current_agent_name,  # Pass name from context
            chat_id=current_chat_id,  # Pass the current chat_id
            # Pass only the user_context part as override
            context_override=master_context.user_context,
            # hooks_override=hooks, # Don't pass hooks from here
        )
        final_output_text = sub_result.final_output or "(No text output from agent)"
        logger.info(
            f"Received response from '{target_agent.name}': \"{final_output_text[:50]}...\""
        )

        # --- Log Communication (Optional but good practice) ---
        # Logging might be better handled within the Runner or specific hooks.

        return final_output_text

    except Exception as e:
        logger.error(
            f"Error occurred during sub-call from '{current_agent_name}' to '{target_agent.name}': {e}",
            exc_info=True,
        )
        return f"Error: Failed to get response from agent '{target_agent.name}'. Reason: {e}"


# --- Export the decorated tool instance --- #
send_message_tool: FunctionTool = send_message
