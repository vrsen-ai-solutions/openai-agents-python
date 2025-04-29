# OpenAI Agents SDK Core Component Analysis (Task 1)

This document summarizes the analysis of the core components of the OpenAI Agents SDK (`src/agents/`) relevant to integrating Agency Swarm's orchestration features.

## Key Files Analyzed:

*   `agent.py`: Defines the `Agent` class and its configuration.
*   `run.py`: Contains the `Runner` class, the main entry point for execution.
*   `_run_impl.py`: Holds the internal logic for executing a single agent turn.
*   `handoffs.py`: Defines the `Handoff` mechanism for transferring control between agents.
*   `tool.py`: Defines `Tool` types (FunctionTool, hosted tools) and the `@function_tool` decorator.
*   `items.py`: Defines `RunItem` and its subclasses, representing events in the execution history.

## Core Concepts & Execution Flow:

1.  **`Agent` Configuration:**
    *   Holds static configuration (instructions, tools, model settings, handoffs, output type).
    *   `instructions` act as the system prompt.
    *   `tools` can be `FunctionTool` or hosted tools (FileSearch, WebSearch, Computer).
    *   `handoffs` list potential target agents for delegation.
    *   `output_type` defines the expected structured output (if not `str`).

2.  **`Runner` Orchestration:**
    *   `Runner.run`/`run_sync`/`run_streamed` starts the workflow.
    *   Manages the overall loop (`while True` until `max_turns` or final output/error).
    *   Maintains the list of generated `RunItem` objects (conversation history).
    *   Handles global configuration (`RunConfig`).
    *   Calls `_run_single_turn` repeatedly.

3.  **`_run_impl._run_single_turn` (The Core Loop):**
    *   Takes the current `agent` and the list of `generated_items` (history).
    *   **Gets Model Input:** Prepares the input message list (`TResponseInputItem`) from `generated_items`.
    *   **Invokes Model:** Calls the `ModelProvider` (`_get_new_response`) with the input history, system prompt, tools, handoffs, and output schema.
    *   **Receives `ModelResponse`:** Gets the LLM's response (text, tool calls).
    *   **Processes Response (`process_model_response`):** Parses the `ModelResponse` into `RunItem`s (like `MessageOutputItem`, `ToolCallItem`, `HandoffCallItem`) and categorizes tool calls (handoffs, functions, computer actions).
    *   **Executes Tools/Side Effects (`execute_tools_and_side_effects`):**
        *   Calls local `FunctionTool` implementations (`execute_function_tool_calls`). Tool outputs become `ToolCallOutputItem`.
        *   Calls `ComputerTool` actions (`execute_computer_actions`). Outputs become `RunItem`s.
        *   Handles **Handoffs (`execute_handoffs`):**
            *   If a handoff tool was called, this resolves the target agent.
            *   Applies the `input_filter` to potentially modify the history (`RunItem` list) passed to the next agent.
            *   Returns `NextStepHandoff(new_agent=target_agent)`.
        *   Checks `agent.tool_use_behavior` to see if function tool outputs constitute a final result (`_check_for_final_output_from_tools`).
    *   **Determines Next Step:**
        *   If handoff occurred: `NextStepHandoff`.
        *   If `output_type` is met or text output has no tools/handoffs: `NextStepFinalOutput`.
        *   If tools were run and need LLM review: `NextStepRunAgain`.
    *   The `Runner` uses the `NextStep` result to either loop with the same/new agent or terminate.

4.  **History Management (`RunItem`):**
    *   The conversation history is implicitly managed as a growing list of `RunItem` objects (`generated_items` in `Runner`).
    *   `RunItem` subtypes include `MessageInputItem`, `MessageOutputItem`, `ToolCallItem`, `ToolCallOutputItem`, `HandoffCallItem`, `HandoffOutputItem`, etc.
    *   This list is passed to the model on each turn.
    *   Handoffs can modify this list via `input_filter` before passing it to the next agent.
    *   There's no separate, persistent `Thread` object managing this history across different `Runner.run` calls or providing complex filtering/scoping beyond the handoff filter.

5.  **Handoffs vs. `agent.as_tool()`:**
    *   **Handoff:** Transfers *control* and *history* (potentially filtered) to a new agent. The new agent continues the main loop. Implemented as a specific tool type recognized by `_run_impl`.
    *   **`agent.as_tool()`:** Wraps an agent run within a *single* `FunctionTool` call. The called agent runs in isolation with only the provided `input` string (no history transfer). The *result* is returned to the calling agent as a tool output within the *same turn*.

## Summary:

The SDK provides a solid foundation for single-agent execution and simple linear handoffs. History is managed as a list of `RunItem`s passed turn-by-turn. Integrating Agency Swarm's orchestration requires introducing a dedicated history manager (`Thread`) and a central orchestrator (`Agency`) to manage the control flow between agents, replacing or significantly adapting the current `Runner` logic. Communication tools need special handling to interact with the orchestrator.
