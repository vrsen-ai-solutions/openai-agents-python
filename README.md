# Agency Swarm SDK (Based on OpenAI Agents SDK)

The Agency Swarm SDK is a framework for building robust, multi-agent workflows, based on a fork of the lightweight OpenAI Agents SDK. It enhances the base SDK with specific features for defining, orchestrating, and managing persistent conversations between multiple agents.

It is provider-agnostic, supporting the OpenAI Responses and Chat Completions APIs, as well as 100+ other LLMs via the underlying SDK capabilities.

<!-- TODO: Add Agency Swarm specific diagram/image -->
<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

### Core concepts:

1.  **[Agents](https://openai.github.io/openai-agents-python/agents)**: LLMs configured with instructions, tools, guardrails. In Agency Swarm, `agency_swarm.Agent` extends the base agent, managing sub-agent communication links and file handling.
2.  **[Agency](link-to-agency-docs)**: A builder class that defines the overall structure of collaborating agents using an `agency_chart`. It registers agents, sets up communication paths (via the `send_message` tool), manages shared state (`ThreadManager`), and configures persistence.
3.  **`agency_chart`**: A list defining the agents and allowed communication flows within an `Agency`.
4.  **[ThreadManager & Persistence](link-to-persistence-docs)**: Manages conversation state (`ConversationThread`) for each agent interaction. Persistence is handled via `load_callback` and `save_callback` functions provided to the `Agency`.
5.  **`send_message` Tool**: A specialized `FunctionTool` automatically added to agents, enabling them to invoke other agents they are connected to in the `agency_chart`.
6.  **[Tools](https://openai.github.io/openai-agents-python/tools/)**: Standard SDK tools (like `FunctionTool`) used by agents to perform actions.
7.  **[Guardrails](https://openai.github.io/openai-agents-python/guardrails/)**: Configurable safety checks for input and output validation (inherited from SDK).
8.  **[Tracing](https://openai.github.io/openai-agents-python/tracing/)**: Built-in tracking of agent runs via the SDK, allowing viewing and debugging.

Explore the [examples](./examples/) directory to see the SDK in action, and read our documentation (including the [Migration Guide](./docs/migration_guide.md)) for more details.

## Get started

1.  Set up your Python environment

```bash
python -m venv env
source env/bin/activate
```

2.  Install Agency Swarm SDK

```bash
# TODO: Update with actual package name
pip install agency-swarm-sdk
```

<!-- For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'` (Check if applicable). -->

## Hello world example (Single Agent)

```python
# Demonstrates basic SDK Agent usage
from agents import Agent, Runner # Use base SDK components

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

## Agency Swarm Example (Two Agents)

```python
import asyncio
from agency_swarm import Agent, Agency
from agency_swarm.thread import ThreadManager, ConversationThread # Import thread components
from typing import Optional
import uuid

# Define two agents
agent_one = Agent(
    name="AgentOne",
    instructions="You are agent one. If asked about the weather, send a message to AgentTwo.",
    # tools=[...] # send_message is added automatically by Agency if needed
)

agent_two = Agent(
    name="AgentTwo",
    instructions="You are agent two. You report the weather is always sunny.",
)

# Define the agency structure and communication
# AgentOne can talk to AgentTwo
agency_chart = [
    agent_one,
    agent_two,
    [agent_one, agent_two] # Defines communication flow: agent_one -> agent_two
]

# --- Simple In-Memory Persistence Example ---
# (Replace with file/database callbacks for real use)
memory_threads: dict[str, ConversationThread] = {}

def memory_save_callback(thread: ConversationThread):
    print(f"[Persistence] Saving thread: {thread.thread_id}")
    memory_threads[thread.thread_id] = thread

def memory_load_callback(thread_id: str) -> Optional[ConversationThread]:
    print(f"[Persistence] Loading thread: {thread_id}")
    return memory_threads.get(thread_id)
# -------------------------------------------

# Create the agency
agency = Agency(
    agency_chart=agency_chart,
    load_callback=memory_load_callback,
    save_callback=memory_save_callback
)

async def main():
    chat_id = f"chat_{uuid.uuid4()}"
    print(f"--- Turn 1: Asking AgentOne about weather (ChatID: {chat_id}) ---")
    result1 = await agency.get_response(
        message="What is the weather like?",
        recipient_agent="AgentOne", # Start interaction with AgentOne
        chat_id=chat_id
    )
    print(f"\nFinal Output from AgentOne: {result1.final_output}")
    # Expected: AgentOne calls send_message -> AgentTwo responds -> AgentOne returns AgentTwo's response
    # Output should be similar to: The weather is always sunny.

    print(f"\n--- Turn 2: Follow up (ChatID: {chat_id}) ---")
    # Use the *same* chat_id to continue the conversation
    result2 = await agency.get_response(
        message="Thanks!",
        recipient_agent="AgentOne", # Continue with AgentOne
        chat_id=chat_id
    )
    print(f"\nFinal Output from AgentOne: {result2.final_output}")

    print(f"\n--- Checking Persisted History for {chat_id} ---")
    final_thread = memory_load_callback(chat_id)
    if final_thread:
        print(f"Thread {final_thread.thread_id} items:")
        for item in final_thread.items:
            print(f"- Role: {item.get('role')}, Content: {item.get('content')}")
    else:
        print("Thread not found in memory.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Functions example

```python
import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
```

## The agent loop

When you call `Runner.run()`, we run a loop until we get a final output.

1. We call the LLM, using the model and settings on the agent, and the message history.
2. The LLM returns a response, which may include tool calls.
3. If the response has a final output (see below for more on this), we return it and end the loop.
4. If the response has a handoff, we set the agent to the new agent and go back to step 1.
5. We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `max_turns` parameter that you can use to limit the number of times the loop executes.

### Final output

Final output is the last thing the agent produces in the loop.

1.  If you set an `output_type` on the agent, the final output is when the LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `output_type` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

As a result, the mental model for the agent loop is:

1. If the current agent has an `output_type`, the loop runs until the agent produces structured output matching that type.
2. If the current agent does not have an `output_type`, the loop runs until the current agent produces a message without any tool calls/handoffs.

## Agency Swarm Orchestration

Agency Swarm uses the `Agency` class to define and manage groups of collaborating agents. The structure and communication paths are defined in the `agency_chart`.

- **Agent Registration:** Agents are automatically registered when the `Agency` is initialized based on the chart.
- **Communication:** Agents communicate using the built-in `send_message` tool, which is automatically added to agents that have permission to talk to others (defined by `[SenderAgent, ReceiverAgent]` pairs in the chart).
- **Entry Points:** Interactions are typically started by calling `agency.get_response()` or `agency.get_response_stream()`, targeting a specific agent designated as an entry point (though calling non-entry points is possible).

## State & Persistence

Conversation state is managed per interaction context (identified by `chat_id`) using `ConversationThread` objects held by a central `ThreadManager`.

- **`ConversationThread`**: Stores the sequence of messages and tool interactions for a specific chat.
- **`ThreadManager`**: Creates, retrieves, and manages `ConversationThread` instances.
- **Persistence Callbacks:** The `Agency` accepts optional `load_callback(thread_id)` and `save_callback(thread)` functions during initialization. These functions are responsible for loading and saving `ConversationThread` state to your desired backend (e.g., files, database). If provided, they are automatically invoked at the start and end of agent runs via `PersistenceHooks`.

## Common agent patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide range of LLM workflows including deterministic flows, iterative loops, and more. See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

The Agents SDK automatically traces your agent runs, making it easy to track and debug the behavior of your agents. Tracing is extensible by design, supporting custom spans and a wide variety of external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). For more details about how to customize or disable tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Development (only needed if you need to edit the SDK/examples)

0. Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv --version
```

1. Install dependencies

```bash
make sync
```

2. (After making changes) lint/test

```
make tests  # run tests
make mypy   # run typechecker
make lint   # run linter
```

## Acknowledgements

We'd like to acknowledge the excellent work of the open-source community, especially:

-   [Pydantic](https://docs.pydantic.dev/latest/) (data validation) and [PydanticAI](https://ai.pydantic.dev/) (advanced agent framework)
-   [MkDocs](https://github.com/squidfunk/mkdocs-material)
-   [Griffe](https://github.com/mkdocstrings/griffe)
-   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We're committed to continuing to build the Agents SDK as an open source framework so others in the community can expand on our approach.
