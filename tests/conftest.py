from __future__ import annotations

import pathlib
import pathlib as _pl
import sys

import pytest

# Add project src directory to sys.path for tests
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agents.models import _openai_shared
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tracing import set_trace_processors
from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

from .testing_processor import SPAN_PROCESSOR_TESTING


# This fixture will run once before any tests are executed
@pytest.fixture(scope="session", autouse=True)
def setup_span_processor():
    set_trace_processors([SPAN_PROCESSOR_TESTING])


# This fixture will run before each test
@pytest.fixture(autouse=True)
def clear_span_processor():
    SPAN_PROCESSOR_TESTING.force_flush()
    SPAN_PROCESSOR_TESTING.shutdown()
    SPAN_PROCESSOR_TESTING.clear()


# This fixture will run before each test
@pytest.fixture(autouse=True)
def clear_openai_settings():
    _openai_shared._default_openai_key = None
    _openai_shared._default_openai_client = None
    _openai_shared._use_responses_by_default = True


# This fixture will run after all tests end
@pytest.fixture(autouse=True, scope="session")
def shutdown_trace_provider():
    yield
    GLOBAL_TRACE_PROVIDER.shutdown()


@pytest.fixture(autouse=True)
def disable_real_model_clients(monkeypatch, request):
    # If the test is marked to allow the method call, don't override it.
    if request.node.get_closest_marker("allow_call_model_methods"):
        return

    def failing_version(*args, **kwargs):
        pytest.fail("Real models should not be used in tests!")

    monkeypatch.setattr(OpenAIResponsesModel, "get_response", failing_version)
    monkeypatch.setattr(OpenAIResponsesModel, "stream_response", failing_version)
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", failing_version)
    monkeypatch.setattr(OpenAIChatCompletionsModel, "stream_response", failing_version)


# Stub modules if missing to avoid import errors in tests
try:
    import inline_snapshot  # type: ignore
except ImportError:  # pragma: no cover
    import types

    _snapshot_mod = types.ModuleType("inline_snapshot")

    def _default_snapshot(value=None):
        return value

    _snapshot_mod.snapshot = _default_snapshot  # type: ignore
    import sys as _sys

    _sys.modules["inline_snapshot"] = _snapshot_mod

try:
    import graphviz  # type: ignore
except ImportError:  # pragma: no cover
    import types as _types

    _graphviz_mod = _types.ModuleType("graphviz")

    class _Digraph:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

        def node(self, *args, **kwargs):
            pass

        def edge(self, *args, **kwargs):
            pass

        def source(self):
            return ""

    _graphviz_mod.Digraph = _Digraph  # type: ignore
    import sys as _sys

    _sys.modules["graphviz"] = _graphviz_mod


def pytest_ignore_collect(path, config):  # noqa: D401
    """Skip collecting tests that have external dependencies or need major rework."""
    p = _pl.Path(str(path))
    parts = p.parts

    # Skip old tool factory tests (requires langchain)
    if p.name == "test_tool_factory.py" and "old_agency_swarm_tests" in parts:
        return True

    # Skip MCP server integration tests (requires running server & mcp package)
    # Also skip old_agency_swarm_tests/test_mcp.py which depends on old structure/mcp package
    if p.name == "test_mcp.py" and "old_agency_swarm_tests" in parts:
        return True
    if "tests" in parts and "mcp" in parts:
        return True

    # Skip map model tests (requires litellm)
    if p.name == "test_map.py" and "models" in parts:
        return True

    # Skip visualization tests (requires full graphviz support)
    if p.name == "test_visualization.py":
        return True

    # Skip voice tests for now (require audio frameworks)
    # if "voice" in parts:
    #    return True

    return False
