import logging
import os

# --- Adjust path to import agency_swarm ---
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agency_swarm.agency import Agency  # Needed for context setup
from agency_swarm.agent import Agent
from agency_swarm.thread import ThreadManager
from agency_swarm.tools.send_message import SEND_MESSAGE_TOOL_NAME
from agents import Tool
from agents.tool import FileSearchTool


# --- Test Fixtures (Optional but helpful) ---
@pytest.fixture
def mock_thread_manager():
    """Provides a mocked ThreadManager instance."""
    manager = MagicMock(spec=ThreadManager)
    # Mock any methods needed by Agent during init or tests
    manager.get_thread = MagicMock(return_value=MagicMock())
    return manager


@pytest.fixture
def mock_agency():
    """Provides a mocked Agency instance."""
    agency = MagicMock(spec=Agency)
    agency.agents = {}  # Mock the agents dictionary
    return agency


# --- Test Class for Agent ---
class TestAgencySwarmAgent:
    def test_agent_initialization_minimal(self):
        """Test basic agent initialization with minimal parameters."""
        agent = Agent(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.instructions is None
        assert agent.tools == []
        assert agent._subagents == {}
        assert agent.files_folder is None
        assert agent._thread_manager is None  # Not set until configured by Agency
        assert agent._agency_instance is None

    def test_agent_initialization_with_instructions(self):
        """Test agent initialization with instructions."""
        instructions = "Be helpful."
        agent = Agent(name="HelpfulAgent", instructions=instructions)
        assert agent.name == "HelpfulAgent"
        assert agent.instructions == instructions

    def test_agent_initialization_with_tools(self):
        """Test agent initialization with a list of tools."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "MockTool"
        tools_list = [mock_tool]
        agent = Agent(name="ToolAgent", tools=tools_list)
        assert agent.name == "ToolAgent"
        assert agent.tools == tools_list
        # Verify SendMessageTool is NOT added automatically if no subagents registered yet
        assert not any(t.name == SEND_MESSAGE_TOOL_NAME for t in agent.tools if hasattr(t, "name"))

    def test_agent_initialization_with_files_folder(self, tmp_path):
        """Test agent initialization with a files_folder path."""
        files_folder = tmp_path / "agent_files"
        # files_folder does not exist yet
        agent = Agent(name="FileAgent", files_folder=str(files_folder))
        assert agent.name == "FileAgent"
        assert agent.files_folder == str(files_folder)
        assert agent.files_folder_path == files_folder
        assert files_folder.is_dir()  # Check folder was created
        assert agent._associated_vector_store_id is None  # No VS ID in name
        # FileSearchTool should not be added without a VS ID
        assert not any(isinstance(t, FileSearchTool) for t in agent.tools)

    def test_agent_initialization_with_files_folder_and_vs_id(self, tmp_path):
        """Test agent initialization with a files_folder containing a VS ID."""
        vs_id = "vs_123abc"
        folder_name_with_vs = f"agent_files_vs_{vs_id}"
        files_folder_with_vs = tmp_path / folder_name_with_vs
        # files_folder_with_vs does not exist yet
        agent = Agent(name="VSFileAgent", files_folder=str(files_folder_with_vs))

        # Correct expected base path (agent code strips the VS ID suffix)
        expected_base_path = tmp_path / "agent_files_vs"

        assert agent.name == "VSFileAgent"
        assert agent.files_folder == str(files_folder_with_vs)
        assert agent._associated_vector_store_id == vs_id[3:]  # Revert: Check ID without prefix
        assert agent.files_folder_path == expected_base_path
        assert expected_base_path.is_dir()  # Check base folder was created
        # FileSearchTool SHOULD be added
        assert any(isinstance(t, FileSearchTool) for t in agent.tools)
        # Check the VS ID on the added tool
        fs_tool = next((t for t in agent.tools if isinstance(t, FileSearchTool)), None)
        assert fs_tool is not None
        assert fs_tool.vector_store_ids == [vs_id[3:]]  # Check against ID without prefix

    # --- Subagent Registration Tests ---
    def test_register_subagent_success(self):
        """Test successfully registering a subagent."""
        agent = Agent(name="MainAgent")
        subagent = Agent(name="SubAgent")

        agent.register_subagent(subagent)

        assert "SubAgent" in agent._subagents
        assert agent._subagents["SubAgent"] == subagent
        # Check if SendMessageTool was added
        assert any(t.name == SEND_MESSAGE_TOOL_NAME for t in agent.tools if hasattr(t, "name"))

    def test_register_subagent_adds_tool_only_once(self):
        """Test that SendMessageTool is added only once even if multiple subagents are registered."""
        agent = Agent(name="MainAgent")
        subagent1 = Agent(name="SubAgent1")
        subagent2 = Agent(name="SubAgent2")

        agent.register_subagent(subagent1)
        agent.register_subagent(subagent2)

        send_message_tools = [
            t for t in agent.tools if hasattr(t, "name") and t.name == SEND_MESSAGE_TOOL_NAME
        ]
        assert len(send_message_tools) == 1
        assert "SubAgent1" in agent._subagents
        assert "SubAgent2" in agent._subagents

    def test_register_subagent_duplicate(self, caplog):
        """Test registering the same subagent again logs a warning and doesn't duplicate."""
        agent = Agent(name="MainAgent")
        subagent = Agent(name="SubAgent")

        agent.register_subagent(subagent)
        tool_count_before = len(agent.tools)

        with caplog.at_level(logging.WARNING):
            agent.register_subagent(subagent)  # Register again

        assert "already registered as a subagent" in caplog.text
        assert "SubAgent" in agent._subagents
        assert len(agent.tools) == tool_count_before  # Tool count shouldn't change

    def test_register_subagent_self_raises_error(self):
        """Test that an agent cannot register itself as a subagent."""
        agent = Agent(name="SelfAgent")
        with pytest.raises(ValueError, match="Agent cannot register itself"):
            agent.register_subagent(agent)

    # --- File Handling Method Tests (upload_file, check_file_exists) ---
    # Needs mocking for client calls

    # --- Add more tests below for other methods etc. ---

    # --- Add more tests below for other initialization params, methods, etc. ---
    # test_agent_initialization_with_tools()
    # test_agent_initialization_with_files_folder()
    # test_register_subagent()
    # test_register_subagent_adds_send_message_tool()
    # test_init_file_handling_no_folder()
    # test_init_file_handling_with_folder()
    # test_init_file_handling_with_vs_id()
    # test_upload_file_no_folder()
    # test_upload_file_with_folder()
    # test_upload_file_with_vs()
    # test_check_file_exists_no_vs()
    # test_check_file_exists_vs_found()
    # test_check_file_exists_vs_not_found()
    # test_get_response_setup() # Test context prep
    # test_get_response_calls_runner()
    # test_get_response_stream_setup()
    # test_get_response_stream_calls_runner_streamed()
    # test_set_thread_manager()
    # test_set_agency_instance()
