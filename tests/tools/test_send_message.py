import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agency_swarm.agent import Agent
from agency_swarm.context import MasterContext
from agency_swarm.tools.send_message import send_message_tool
from agents import RunContextWrapper, RunResult


@pytest.fixture
def mock_master_context():
    """Creates a mock MasterContext with mock agents."""
    mock_agent_sender = MagicMock(spec=Agent)
    mock_agent_sender.name = "SenderAgent"
    mock_agent_sender._subagents = {"RecipientAgent": None}  # Sender knows Recipient

    mock_agent_recipient = MagicMock(spec=Agent)
    mock_agent_recipient.name = "RecipientAgent"
    # Mock the get_response method as an async function
    mock_response = MagicMock(spec=RunResult)
    mock_response.final_output = "Response from RecipientAgent"
    mock_agent_recipient.get_response = AsyncMock(return_value=mock_response)

    mock_agent_other = MagicMock(spec=Agent)
    mock_agent_other.name = "OtherAgent"

    context = MasterContext(
        agents={
            "SenderAgent": mock_agent_sender,
            "RecipientAgent": mock_agent_recipient,
            "OtherAgent": mock_agent_other,
        },
        thread_manager=MagicMock(),  # Mock ThreadManager if needed
        current_agent_name="SenderAgent",
        chat_id="test_chat_123",
        user_context={},
    )
    return context


@pytest.fixture
def mock_wrapper(mock_master_context):
    """Creates a mock RunContextWrapper."""
    wrapper = MagicMock(spec=RunContextWrapper)
    wrapper.context = mock_master_context
    return wrapper


# --- Test Cases ---


@pytest.mark.asyncio
async def test_send_message_success(mock_wrapper, mock_master_context):
    """Tests successful message sending and response retrieval."""
    recipient_name = "RecipientAgent"
    message_content = "Hello, Recipient!"
    expected_response = "Response from RecipientAgent"

    args_json = json.dumps({"recipient": recipient_name, "message": message_content})

    result = await send_message_tool.on_invoke_tool(ctx=mock_wrapper, input=args_json)

    # Assert target agent's get_response was called correctly
    mock_agent_recipient = mock_master_context.agents["RecipientAgent"]
    mock_agent_recipient.get_response.assert_awaited_once_with(
        message=message_content,
        sender_name="SenderAgent",
        chat_id="test_chat_123",
        context_override={},
    )

    # Assert the result is the expected output
    assert result == expected_response


@pytest.mark.asyncio
async def test_send_message_invalid_recipient(mock_wrapper):
    """Tests sending to a recipient not registered with the sender."""
    recipient_name = "UnknownAgent"
    message_content = "This should fail."

    args_json = json.dumps({"recipient": recipient_name, "message": message_content})

    result = await send_message_tool.on_invoke_tool(ctx=mock_wrapper, input=args_json)

    # Assert an error message is returned indicating the invalid recipient
    assert "Error: Invalid recipient" in result
    assert recipient_name in result
    assert "RecipientAgent" in result  # Check that valid recipients are listed


@pytest.mark.asyncio
async def test_send_message_recipient_not_found_in_agency(mock_wrapper, mock_master_context):
    """Tests sending to a recipient known by sender but not in agency context."""
    recipient_name = "MissingAgent"
    message_content = "This agent doesn't exist in context."

    # Add the missing agent to the sender's subagents but not to the main context
    mock_master_context.agents["SenderAgent"]._subagents[recipient_name] = None

    args_json = json.dumps({"recipient": recipient_name, "message": message_content})

    result = await send_message_tool.on_invoke_tool(ctx=mock_wrapper, input=args_json)

    # Assert an error message is returned indicating the recipient wasn't found
    assert f"Error: Recipient agent '{recipient_name}' not found in the agency." in result


@pytest.mark.asyncio
async def test_send_message_target_agent_error(mock_wrapper, mock_master_context):
    """Tests error handling when the target agent raises an exception."""
    recipient_name = "RecipientAgent"
    message_content = "Trigger error in recipient."
    error_message = "Simulated error in get_response"

    # Configure the mock recipient agent to raise an exception
    mock_agent_recipient = mock_master_context.agents[recipient_name]
    mock_agent_recipient.get_response.side_effect = Exception(error_message)

    args_json = json.dumps({"recipient": recipient_name, "message": message_content})

    result = await send_message_tool.on_invoke_tool(ctx=mock_wrapper, input=args_json)

    # Assert an error message is returned containing the exception details
    assert f"Error: Failed to get response from agent '{recipient_name}'." in result
    assert error_message in result
