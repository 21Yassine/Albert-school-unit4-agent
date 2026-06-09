# Albert School — Unit 4: Customer Support Agent with LangGraph

> *"An agent is an LLM that controls its own loop: it decides what to do, does it, sees what happened, decides again."*

---

## Architecture: The Four Pillars

This agent is built on the four principles from the Unit 4 curriculum:

| Pillar | Implementation | Role |
|--------|---------------|------|
| **LLM** | `ChatOpenAI(gpt-4o-mini)` | The engine — provides reasoning |
| **Loop** | `StateGraph` with conditional edges | The control loop — Reason → Act → Observe |
| **Tools** | Specialist nodes (billing / technical / general) | Reach into the world — each wraps a domain-scoped LLM call |
| **Harness** | `SqliteSaver` + `interrupt_before` + `stream_mode="updates"` | Safety, persistence, observability |

---

## Graph Flow

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │  sequential edge
                    ┌──────▼──────┐
                    │  classify   │  ← LLM call: sets state["category"]
                    └──────┬──────┘
                           │  conditional edge via route()
           ┌───────────────┼──────────────────┐
           ▼               ▼                  ▼                  ▼
    ┌─────────────┐ ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    │   billing   │ │  technical  │  │   general    │  │  [INTERRUPT] │
    │ specialist  │ │ specialist  │  │  specialist  │  │    human     │
    └──────┬──────┘ └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
           │               │                │                  │
           └───────────────┴────────────────┴──────────────────┘
                                            │
                                     ┌──────▼──────┐
                                     │     END     │
                                     └─────────────┘
```

**Key design rule from the curriculum:**  
> *"The classifier does the thinking; the router does the routing. Keeping them separate is what lets the LLM steer the graph through the State, without entangling the two responsibilities."*

---

## State: The Shared Contract

```python
class CustomerSupportState(TypedDict):
    messages: Annotated[list, add_messages]   # chat history — ACCUMULATES (reducer)
    category: str                              # routing decision — OVERWRITES
```

- `add_messages` is LangGraph's built-in reducer for chat history. It **appends** new messages instead of replacing the list.
- `category` uses default overwrite behaviour — the classifier sets it once per turn.
- **The classifier writes only `category`. Every specialist writes only `messages`.** Each node has exactly one responsibility.

---

## Setup

```bash
# 1. Navigate to the project
cd ~/Desktop/albert_school_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 4. Run the demo
python app.py
```

---

## Thread ID and State Lifecycle

Each conversation is identified by a `thread_id` passed in the config:

```python
config = {"configurable": {"thread_id": "session_user_42"}}
```

The `SqliteSaver` checkpointer writes the full `CustomerSupportState` to `support.db` **after every node**. This means:

- **Robustness**: if the process crashes mid-run, the conversation resumes exactly where it stopped.
- **Multi-turn memory**: a follow-up question on the same `thread_id` automatically includes the full prior conversation.
- **Observability**: every intermediate state snapshot is queryable via `compiled_graph.get_state(config)`.

```python
# Resume a previous conversation on the same thread
compiled_graph.stream(
    {"messages": [HumanMessage(content="And what about the Pro plan?")]},
    config={"configurable": {"thread_id": "session_user_42"}},
    stream_mode="updates"
)
```

---

## Human-in-the-Loop

When the classifier outputs `category = "human"`, the conditional router sends execution to the `human` node. Because the graph is compiled with `interrupt_before=["human"]`, **execution pauses before that node runs**.

The harness pattern (from section 4.17 of the class notes):

```
Checkpointer + interrupt + streaming = the harness pattern for supervised agents.
Together, they let an agent run in production without flying blind.
```

### Resume Protocol

```python
# Step 1 — graph pauses here
events = compiled_graph.stream(
    {"messages": [HumanMessage(content="I want to speak to a manager!")]},
    config,
    stream_mode="updates"
)
for event in events:
    print(event)   # streams: classify deciding…

# Check interrupt
snapshot = compiled_graph.get_state(config)
print(snapshot.next)   # ('human',)  ← graph is waiting

# Step 2 — human agent injects their reply
from langchain_core.messages import AIMessage
compiled_graph.update_state(
    config,
    {"messages": [AIMessage(content="Hello, I'm the manager. How can I help?")]},
    as_node="human",   # marks the human node as complete
)

# Step 3 — resume; graph continues to END
for event in compiled_graph.stream(None, config, stream_mode="updates"):
    print(event)
```

---

## Observability: stream_mode="updates"

`stream_mode="updates"` yields only the **state diff** (changed fields) after each node — not the full state. This is the right default for agent UIs:

- `"classifier deciding…"` appears immediately when `classify` writes to `category`
- `"routing to billing specialist…"` appears when the conditional edge fires
- The response streams token-by-token if you switch to `stream_mode="messages"`

---

## File Structure

```
albert_school_agent/
├── app.py             ← Complete agent: State, Nodes, Edges, Harness, CLI demo
├── requirements.txt   ← Production dependencies
├── README.md          ← This file
└── support.db         ← Auto-created at runtime (SQLite checkpoint store)
```
