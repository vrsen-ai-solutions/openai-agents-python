# Agent Orchestration Logic Design (Task 4 - Revised)

This document outlines the revised design for the orchestration logic, which now resides **exclusively** *within* the `agency_swarm.Agent.get_response` and `get_response_stream` methods, enabling fully Agent-centric execution and multi-agent communication. The `Agency` class does *not* contain runtime orchestration logic.

## 1. Goals

*   Implement the **complete** core execution loop within `Agent.get_response`.
*   Handle LLM calls, processing of responses, and standard tool execution based on the base SDK's patterns (e.g., `_run_impl.py`).
*   Intercept `send_message` tool calls specifically within the `Agent` loop.
*   Initiate recursive calls to other agents' `get_response`/`get_response_stream` methods directly from the sending agent upon detecting `send_message`.
*   Utilize the `ThreadManager` and correctly scoped `ConversationThread` objects for history management during self-execution and recursive calls.
*   Return results (including those from sub-agents) correctly formatted as `ToolCallOutputItem` to the **calling agent's** execution context.
*   Support both synchronous (`get_response`) and eventually streaming (`get_response_stream`) modes **within the Agent class**.

## 2. Orchestration Logic Flow (within `Agent.get_response`)

The `Agent.get_response` method **is the orchestrator**. It manages the agent's own execution turn and initiates interactions with other agents when required by the LLM's use of the `send_message` tool.

**High-Level Steps:**

1.  **Initialization:**
    *   Verify `ThreadManager` is configured.
    *   Get the appropriate `ConversationThread` for the current interaction (e.g., User<->Self or CallerAgent<->Self) using `ThreadManager.get_thread(chat_id, self.name, sender_name)`.
    *   Add the incoming `message` as a `MessageInputItem` to the thread and save via `ThreadManager` **(if this agent is the entry point, otherwise the message comes from the SendMessage call)**.
2.  **Execution Loop (synchronous `get_response` example):**
    *   Loop until a final output is determined or `max_steps` is reached.
    *   **Prepare for LLM:**
        *   Fetch the current history using `thread.get_history_for_model()`.
        *   Fetch the agent's tools using `self.get_all_tools()` (including the injected `send_message` if applicable).
        *   Prepare system prompt (combining agent instructions and potentially shared instructions from Agency).
        *   Prepare other model settings.
    *   **Call LLM:** Invoke the `ModelProvider` (needs access to this) to get the next response (text, tool calls). Log the request/response `RunItem`s.
    *   **Add LLM Output to Thread:** Add the `MessageOutputItem` and any `ToolCallItem`s from the LLM response to the current `ConversationThread` and save via `ThreadManager`.
    *   **Process Tool Calls:**
        *   Scan the `ToolCallItem`s from the LLM response.
        *   **If `send_message` is detected:**
            *   Extract recipient name and message content.
            *   Validate if communication is allowed using `self._agency_chart_peers`.
            *   If allowed:
                *   Get the recipient `Agent` instance (e.g., via `self._agency_instance.agents`).
                *   Make a *recursive call*: `result = await recipient_agent.get_response(message=content, chat_id=chat_id, sender_name=self.name)`.
                *   Format `result` as a `ToolCallOutputItem` for the `send_message` call ID.
                *   Add the `ToolCallOutputItem` to the *current* agent's thread and save.
                *   **Important:** Check `current_depth` against `max_recursion_depth` *before* making the recursive call.
                *   `continue` the loop (the current agent will call the LLM again with the result of the `send_message` call).
            *   If not allowed or recipient not found:
                *   Create an error `ToolCallOutputItem`.
                *   Add it to the current thread and save.
                *   `continue` the loop.
        *   **If standard tool calls are detected:**
            *   Execute them using logic similar to `_run_impl.execute_function_tool_calls`.
            *   Generate `ToolCallOutputItem`s for their results.
            *   Add these output items to the current thread and save.
            *   `continue` the loop (the agent will call the LLM again with the results of the standard tools).
    *   **Determine Next Step:**
        *   Analyze the LLM response and executed tools (similar to `_run_impl` logic determining `NextStepFinalOutput`, `NextStepRunAgain`).
        *   If `NextStepFinalOutput` is determined (e.g., LLM provided text and no tools need further processing):
            *   Extract the final output content.
            *   Perform `response_validator` check if configured.
            *   Break the loop.
        *   If `NextStepRunAgain` (e.g., after tools were executed):
            *   `continue` the loop (LLM will be called again).
3.  **Finalization:**
    *   If the loop finished due to `NextStepFinalOutput`, construct the `RunResult` object.
        *   `generated_items` in the `RunResult` should ideally represent the items generated *within this specific run/context*, but getting the full thread log might be simpler for V1 (`thread.get_full_log()`). Needs refinement.
        *   Include usage data if available.
    *   If the loop finished due to `max_steps` or error, construct an appropriate error `RunResult`.
    *   Return the `RunResult` or just the `final_output_text` based on the `text_only` flag.

**Streaming (`get_response_stream`):**

*   The overall logic is similar, but the implementation needs to yield `RunItem`s as they are added to the thread.
*   When a recursive `send_message` call is made, the **calling agent** awaits the response (or stream) from the sub-agent (`recipient_agent.get_response` / `recipient_agent.get_response_stream`) and processes the result (or yields its chunks) *before* continuing its own execution.
*   This requires careful management of asynchronous calls/generators within the `Agent.get_response`/`Agent.get_response_stream` methods.

## 3. Key Concepts Reiteration

*   **Agent Responsibility:** The `Agent` **is solely responsible** for managing its own turn-by-turn execution and initiating calls to other agents via `send_message` interception.
*   **Recursion:** `send_message` detected within an Agent's run triggers a direct recursive call to another agent's `get_response` method **from the sending agent**.
*   **State Isolation:** The `ThreadManager` ensures that the correct, isolated `ConversationThread` is used for each step (User<->A, A<->B, B<->C, etc.).
*   **Result Propagation:** Results from recursive calls are returned to the **calling agent** and injected back into the **calling agent's** thread as `ToolCallOutputItem`s, allowing the caller to react to the sub-agent's output.

## 4. Comparison to Base SDK (`_run_impl.py`)

*   This internal Agent orchestration loop adapts concepts from the SDK's `_run_impl.py` but significantly extends it.
*   Key differences include:
    *   Orchestration logic resides within `Agent.get_response`, not an external `Runner`.
    *   Special interception logic for the `send_message` tool call.
    *   Executing recursive calls to other agents.
    *   Integration with `ThreadManager` for saving state after adding items.
    *   Precise implementation of LLM calling (`ModelProvider` access **within the Agent**).
    *   Exact implementation of standard tool execution logic within the agent loop.
    *   Robust implementation of the streaming version (`get_response_stream`), especially handling streamed recursive calls.
    *   Accurate collection of `generated_items` and `usage` for the final `RunResult` across nested calls.
    *   Error handling for LLM failures or tool execution errors within the agent's loop.

This design places the orchestration logic firmly and **exclusively** within the `Agent`, supporting the desired recursive communication pattern while leveraging the concept of isolated, persistent threads. The `Agency` class acts purely as a setup and delegation layer.

## 5. Open Questions/Refinements

*   Precise implementation of LLM calling (`ModelProvider` access).
*   Exact implementation of standard tool execution logic within the agent loop.
*   Robust implementation of the streaming version, especially handling streamed recursive calls.
*   Accurate collection of `generated_items` and `usage` for the final `RunResult` across nested calls.
*   Error handling for LLM failures or tool execution errors within the loop.
