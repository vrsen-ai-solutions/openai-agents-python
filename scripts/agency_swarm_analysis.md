# Agency Swarm Communication Pattern Analysis (Task 2)

This document analyzes the communication and orchestration patterns of Agency Swarm based on the provided documentation (`docs-agency-swarm/core-framework/agencies/`).

## Core Concepts:

1.  **`Agency` Class (`overview.mdx`):**
    *   Acts as a container for multiple `Agent` instances.
    *   Configures shared resources (instructions, files) and default settings (temperature, tokens).
    *   The primary definition of agent interaction structure is delegated to the `agency_chart` parameter.

2.  **`agency_chart` (`communication-flows.mdx`, `overview.mdx`):**
    *   Defines allowed communication pathways and user interaction points.
    *   **Structure:** A list containing:
        *   Individual `Agent` objects: These agents can interact directly with the user.
        *   Nested lists `[AgentA, AgentB]`: Defines a *unidirectional* communication path where `AgentA` can initiate communication with `AgentB`.
    *   **Flexibility:** Allows defining arbitrary, non-hierarchical communication graphs (e.g., `[A, B]`, `[B, C]`, `[C, A]`).
    *   **Example:**
        ```python
        agency = Agency([
            ceo, dev          # ceo, dev can talk to user
            [ceo, dev],       # ceo can send message to dev
            [ceo, va],        # ceo can send message to va
            [dev, va]         # dev can send message to va
        ])
        ```

3.  **Communication Mechanism (`communication-flows.mdx`):**
    *   Relies on a specialized tool, implicitly named `SendMessage`.
    *   Defining `[AgentA, AgentB]` in the `agency_chart` configures `AgentA`'s `SendMessage` tool to accept `AgentB` as a valid recipient.
    *   When `AgentA` calls `SendMessage` targeting `AgentB`, it triggers `AgentB` to process the message.

4.  **`Thread` Concept (`communication-flows.mdx`, `overview.mdx`):**
    *   The documentation mentions agents responding "in the same thread" and provides `threads_callbacks` for persistence.
    *   **Ambiguity:** The *specific implementation* of how threads manage conversation history, context isolation between different agent pairs, or how they relate to the `RunItem` list in the target SDK is **not detailed** in the analyzed files.
    *   **Inference:** It's highly likely the original Agency Swarm heavily relied on the underlying OpenAI Assistants API's thread management. This needs to be explicitly designed and implemented in the SDK fork.

## Comparison to OpenAI Agents SDK Analysis (Task 1):

*   **Orchestration:**
    *   **Agency Swarm:** Declarative graph (`agency_chart`) + specialized tool (`SendMessage`). Enables flexible, multi-directional flows initiated by agents.
    *   **Agents SDK:** Linear loop (`Runner`) + control transfer (`Handoff`). Handoffs are unidirectional transfers of control and full history (modifiable by filter).
*   **History Management:**
    *   **Agency Swarm:** Relies on an abstracted `Thread` concept (likely via Assistants API threads) for context persistence and isolation.
    *   **Agents SDK:** Explicitly manages history as a linear list of `RunItem` objects within a single run context.
*   **Agent-to-Agent Communication:**
    *   **Agency Swarm:** Primary method is via the orchestrated `SendMessage` tool, maintaining separate contexts/threads.
    *   **Agents SDK:** `Handoff` (transfers control) or `agent.as_tool()` (synchronous call, no history transfer, result returned in the same turn).

## Conclusion & Next Steps (for Task 2):

Agency Swarm defines communication pathways using the `agency_chart` and executes them via a `SendMessage` tool. The core orchestration logic seems embedded within this tool's interaction with the underlying (and currently undocumented) `Thread` management system.

To proceed with the integration:

1.  The `Thread` mechanism needs explicit design (Task 3). It must replicate the necessary context management and isolation implied by Agency Swarm's structure, but using the SDK's `RunItem`s or a similar approach instead of relying on the Assistants API.
2.  The `Agency` class design (Task 6) needs to incorporate the `agency_chart` parsing and setup of the `SendMessage` tool equivalents for permitted agent pairs.
3.  The `SendMessage` tool needs special handling within the execution loop (Task 4/Task 7/Task 15) to trigger the orchestrator logic rather than being treated as a standard function or handoff.

This completes the analysis required for Task 2.

### Required Changes/Refactoring

1.  The `agents.Agent` needs to be extended (Task 5) to include Agency Swarm features (file handling, dual tool support, response methods like `get_response`/`get_response_stream`, potentially validator). The `SendMessage` logic will be baked into these new response methods, intercepting the tool call.
2.  The `Agency` class design (Task 6) needs to incorporate the `agency_chart` parsing and setup of the `send_message` tool equivalents for permitted agent pairs.
3.  The `send_message` tool needs special handling within the execution loop (Task 4/Task 7/Task 15/Task 11) to trigger the orchestrator logic rather than being treated as a standard function or handoff.
