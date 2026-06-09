#!/usr/bin/env python3
"""
Albert School – Unit 4: Agents with LangGraph
Customer Support Agent – Production-Ready Implementation

Architecture: the four pillars from the curriculum
  - LLM    : ChatOpenAI (gpt-4o-mini) – the engine
  - Loop   : StateGraph with conditional edges – the control loop
  - Tools  : Specialist nodes – domain-scoped LLM calls
  - Harness: SqliteSaver + interrupt_before + stream_mode="updates"

Usage:
    export OPENAI_API_KEY="sk-..."
    python app.py
"""

import os
import sqlite3
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# =============================================================================
# PILLAR 1 — SHARED STATE
#
# The State is the shared contract every node reads from and writes to.
# It is defined once; LangGraph merges partial updates returned by each node.
#
# add_messages: built-in LangGraph reducer for chat history.
#   Default behaviour  → overwrite (new value replaces old)
#   With add_messages  → append   (new messages join the list)
# This matters: without it, each specialist would erase the user's question.
# =============================================================================

class CustomerSupportState(TypedDict):
    messages: Annotated[list, add_messages]   # accumulates the full chat history
    category: str                              # routing decision written by classify()


# =============================================================================
# LLM
# "The model is the engine. The harness is the rest of the car."
# temperature=0 ensures deterministic classification decisions.
# =============================================================================

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# =============================================================================
# PILLAR 2 — ATOMIC NODES
#
# Each node is a function:  (CustomerSupportState) -> dict
# The dict is a PARTIAL state update — only the fields this node is responsible for.
# LangGraph merges it into the global State; untouched fields are preserved.
#
# Design rule: one responsibility per node.
#   classify()          → writes ONLY  category
#   *_specialist()      → writes ONLY  messages
# =============================================================================

def classify(state: CustomerSupportState) -> dict:
    """
    Classify the last user message into one of four routing categories.
    Writes ONLY the 'category' field.
    """
    last_content = state["messages"][-1].content
    prompt = (
        "You are a customer support router.\n"
        "Classify the message below into exactly one category.\n\n"
        "Categories:\n"
        "  billing   – payment, invoice, subscription, refund, charge\n"
        "  technical – bug, error, configuration, installation, API, integration\n"
        "  general   – product info, account, pricing, anything else\n"
        "  human     – customer explicitly requests a human, manager, or escalation\n\n"
        "Reply with one lowercase word only (billing / technical / general / human).\n\n"
        f"Message: {last_content}"
    )
    response = llm.invoke(prompt)
    raw = response.content.strip().lower()
    valid_categories = {"billing", "technical", "general", "human"}
    category = raw if raw in valid_categories else "general"
    return {"category": category}


def billing_specialist(state: CustomerSupportState) -> dict:
    """
    Billing expert node.
    Prepends a domain system prompt, invokes the LLM with the full message history,
    and returns a partial state update modifying ONLY 'messages'.
    """
    response = llm.invoke([
        SystemMessage(content=(
            "You are a billing specialist at a SaaS company. "
            "Help the customer with payments, invoices, subscriptions, and refunds. "
            "Be concise, empathetic, and always offer a concrete next step. "
            "If you need to check an account, ask for the account email."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def tech_specialist(state: CustomerSupportState) -> dict:
    """
    Technical support expert node.
    Handles bugs, configuration problems, API errors, and integration issues.
    """
    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior technical support engineer. "
            "Resolve bugs, configuration problems, API errors, and integration issues. "
            "Provide numbered, step-by-step instructions when possible. "
            "If the problem is unclear, ask one precise clarifying question before guessing."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def general_specialist(state: CustomerSupportState) -> dict:
    """
    General customer support node.
    Handles product questions, account queries, and anything outside billing/technical.
    """
    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly and knowledgeable general customer support agent. "
            "Help the customer with product questions, account management, and general inquiries. "
            "Be warm, clear, and solution-oriented. "
            "If a question requires billing or technical expertise, say so and offer to transfer."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def human_review(state: CustomerSupportState) -> dict:
    """
    Human-in-the-loop placeholder node.

    The graph is compiled with interrupt_before=["human"], so execution
    PAUSES before this node ever executes.  A human agent then:
      1. Reads the interrupted state via compiled_graph.get_state(config)
      2. Injects their reply:
            compiled_graph.update_state(config,
                {"messages": [AIMessage(content="...")]},
                as_node="human")
      3. Resumes execution: compiled_graph.stream(None, config)

    After update_state(..., as_node="human"), this node is marked as complete.
    stream(None, config) advances the graph from human → END without re-running it.
    This function is therefore a no-op in the interrupt-and-resume pattern.
    """
    return {}


# =============================================================================
# PILLAR 3 — ROUTING FUNCTION
#
# The router is NOT a node. It is a pure routing function:
#   (State) -> node_name_string
#
# It does no work and makes no LLM calls. It simply reads state["category"]
# and returns the name of the next node to execute.
#
# "The classifier does the thinking; the router does the routing."
# Keeping them separate lets the LLM steer the graph through the State
# without entangling two different responsibilities in the same function.
# =============================================================================

def route(state: CustomerSupportState) -> Literal["billing", "technical", "general", "human"]:
    """Map state['category'] → next node name. Deterministic, no side effects."""
    category = state.get("category", "general")
    if category in ("billing", "technical", "general", "human"):
        return category  # type: ignore[return-value]
    return "general"


# =============================================================================
# GRAPH CONSTRUCTION
#
# Edges declare which node runs after which.
# Sequential edge : after A always run B
# Conditional edge: after A, call route(state) to decide which node runs next
#
# Flow:
#   START  ──sequential──>  classify
#   classify  ──conditional via route()──>  billing | technical | general | human
#   billing / technical / general / human  ──>  END
# =============================================================================

graph = StateGraph(CustomerSupportState)

# Register nodes
graph.add_node("classify",  classify)
graph.add_node("billing",   billing_specialist)
graph.add_node("technical", tech_specialist)
graph.add_node("general",   general_specialist)
graph.add_node("human",     human_review)

# Sequential entry edge: START → classify
graph.add_edge(START, "classify")

# Conditional branching: route() reads state["category"] and returns a node name
# LangGraph resolves the returned string to the registered node with that name.
graph.add_conditional_edges("classify", route)

# All specialist paths terminate at END
graph.add_edge("billing",   END)
graph.add_edge("technical", END)
graph.add_edge("general",   END)
graph.add_edge("human",     END)


# =============================================================================
# PILLAR 4 — PRODUCTION HARNESS
#
# Three components:
#
#   SqliteSaver (persistence)
#     Writes the full CustomerSupportState to support.db after every node.
#     Each conversation is keyed by thread_id.
#     The agent can pause, crash, or restart — the thread resumes exactly
#     where it stopped on the next stream() call with the same thread_id.
#
#   interrupt_before=["human"] (human-in-the-loop)
#     Execution pauses BEFORE the human node runs.
#     The host application gets the pause signal, presents the conversation
#     to a human agent, receives their reply, and resumes via stream(None, config).
#
#   stream_mode="updates" (observability)
#     Each iteration of the event loop yields only the STATE DIFF —
#     the fields that changed in that node — not the full state.
#     This lets a UI display "classifier deciding…" and "routing to billing…"
#     in real time, without flooding output with the entire message history.
#
# "Checkpointer + interrupt + streaming is the harness pattern for supervised agents."
# =============================================================================

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support.db")
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)

compiled_graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human"],
)


# =============================================================================
# CLI DEMO
# Shows three automated paths + one complete human-in-the-loop escalation.
# =============================================================================

def _print_stream_event(event: dict) -> None:
    """Pretty-print a single stream update diff."""
    for node_name, node_output in event.items():
        print(f"\n    [NODE: {node_name}]")
        if "category" in node_output:
            print(f"      routing decision → \"{node_output['category']}\"")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                role = type(msg).__name__.replace("Message", "")
                snippet = msg.content[:300].replace("\n", " ")
                print(f"      {role}: {snippet}{'…' if len(msg.content) > 300 else ''}")


def demo_automated(thread_id: str, user_message: str, scenario_label: str) -> None:
    """Run a scenario that resolves automatically without human intervention."""
    print(f"\n{'═' * 65}")
    print(f"  {scenario_label}")
    print(f"  Thread : {thread_id}")
    print(f"  User   : \"{user_message}\"")
    print(f"{'─' * 65}")

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=user_message)]}

    print("  Streaming state updates:")
    for event in compiled_graph.stream(input_state, config, stream_mode="updates"):
        _print_stream_event(event)

    # Read the final persisted state
    final_state = compiled_graph.get_state(config)
    last_message = final_state.values["messages"][-1]
    print(f"\n  FINAL RESPONSE:\n  {last_message.content[:500]}")
    print(f"\n  State persisted in: {_DB_PATH}")


def demo_human_in_the_loop() -> None:
    """
    Full human-in-the-loop scenario:
      RUN 1 → classify() sets category="human" → interrupt fires
      HUMAN AGENT injects reply via update_state()
      RUN 2 → graph resumes → human node completes → END
    """
    thread_id = "demo_human_escalation_001"
    config = {"configurable": {"thread_id": thread_id}}
    user_msg = "I want to speak to a manager immediately. This situation is completely unacceptable!"

    print(f"\n{'═' * 65}")
    print("  SCENARIO 4: Human-in-the-Loop Escalation")
    print(f"  Thread : {thread_id}")
    print(f"  User   : \"{user_msg}\"")
    print(f"{'─' * 65}")

    # ── RUN 1: stream until the interrupt ─────────────────────────────────
    print("\n  [RUN 1] Streaming — graph will pause at 'human' node…")
    for event in compiled_graph.stream(
        {"messages": [HumanMessage(content=user_msg)]},
        config,
        stream_mode="updates",
    ):
        _print_stream_event(event)

    # ── Inspect the interrupted state ─────────────────────────────────────
    snapshot = compiled_graph.get_state(config)
    print(f"\n  [HARNESS] Execution interrupted.")
    print(f"  [HARNESS] Pending node(s): {snapshot.next}")
    print(f"  [HARNESS] Category in state: \"{snapshot.values.get('category')}\"")
    print("\n  → Human agent is now reviewing the conversation…\n")

    # ── Human agent writes their reply ────────────────────────────────────
    human_reply = (
        "Hello! I'm the Customer Success Manager and I've personally reviewed your case. "
        "I sincerely apologize for the experience you've had — this is not the standard "
        "we hold ourselves to. I'm processing a full refund for your account right now, "
        "which will appear within 3–5 business days. You'll receive a confirmation email "
        "within the hour. Is there anything else I can do to make this right for you?"
    )
    print(f"  [HUMAN AGENT] Injecting reply into thread '{thread_id}':")
    print(f"  \"{human_reply[:100]}…\"\n")

    # update_state with as_node="human" appends the message (via add_messages reducer)
    # and marks the 'human' node as having executed — the graph advances past it.
    compiled_graph.update_state(
        config,
        {"messages": [AIMessage(content=human_reply)]},
        as_node="human",
    )
    print("  [HARNESS] State updated. Human node marked as complete.")

    # ── RUN 2: resume from after the human node → END ─────────────────────
    print("\n  [RUN 2] Resuming graph after human intervention…")
    events_seen = 0
    for event in compiled_graph.stream(None, config, stream_mode="updates"):
        _print_stream_event(event)
        events_seen += 1
    if events_seen == 0:
        print("    (No further nodes — graph reached END.)")

    # ── Print the full persisted conversation log ──────────────────────────
    final_state = compiled_graph.get_state(config)
    print("\n  FULL CONVERSATION LOG (from support.db):")
    for i, msg in enumerate(final_state.values["messages"], 1):
        role = type(msg).__name__.replace("Message", "")
        print(f"    [{i}] {role}: {msg.content[:200]}")

    print(f"\n  Thread '{thread_id}' complete. State persisted in: {_DB_PATH}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("\nERROR: GROQ_API_KEY environment variable is not set.")
        print("Get a free key at groq.com → export GROQ_API_KEY='gsk_...'")
        raise SystemExit(1)

    print("=" * 65)
    print("  Albert School — Unit 4: Customer Support Agent")
    print("  LangGraph Harness: SqliteSaver + interrupt_before + streaming")
    print("=" * 65)
    print(f"  Persistence: {_DB_PATH}\n")

    # Scenario 1: billing path
    demo_automated(
        thread_id="demo_billing_001",
        user_message="I was charged twice this month for my subscription. Can you refund the duplicate?",
        scenario_label="SCENARIO 1: Billing Query (Automated Path)",
    )

    # Scenario 2: technical path
    demo_automated(
        thread_id="demo_technical_001",
        user_message="My API key returns a 401 Unauthorized error even after regenerating it. What's wrong?",
        scenario_label="SCENARIO 2: Technical Support (Automated Path)",
    )

    # Scenario 3: general path
    demo_automated(
        thread_id="demo_general_001",
        user_message="What is the difference between the Starter and Professional plans?",
        scenario_label="SCENARIO 3: General Query (Automated Path)",
    )

    # Scenario 4: human escalation with interrupt/resume
    demo_human_in_the_loop()

    print("\n" + "=" * 65)
    print("  All scenarios complete.")
    print(f"  Full conversation history stored in: {_DB_PATH}")
    print("=" * 65)
