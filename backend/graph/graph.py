from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.graph.state import AgentState
from backend.graph.nodes import (
    router_node,
    planner_node,
    knowledge_node,
    developer_node,
    qa_node,
    qa_router,
    route_after_router,
    route_after_planner,
    route_after_knowledge,
    revision_node,
    git_prepare_node,
    policy_node,
    route_after_policy,
    approval_node,
    route_after_approval,
    git_commit_node,
    cleanup_node,
)


builder = StateGraph(AgentState)

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "knowledge",
    knowledge_node
)

builder.add_node(
    "developer",
    developer_node
)

builder.add_node(
    "qa",
    qa_node
)

builder.add_node(
    "revision",
    revision_node
)
builder.add_node(
    "git_prepare",
    git_prepare_node
)
builder.add_node(
    "policy",
    policy_node
)
builder.add_node(
    "approval",
    approval_node
)
builder.add_node(
    "git_commit",
    git_commit_node
)
builder.add_node(
    "cleanup",
    cleanup_node
)

builder.add_edge(
    START,
    "router"
)

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "planner": "planner",
        "knowledge": "knowledge",
        "developer": "developer",
        "end": END,
    },
)

builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge": "knowledge",
        "developer": "developer",
    },
)

builder.add_conditional_edges(
    "knowledge",
    route_after_knowledge,
    {
        "developer": "developer",
        "end": END,
    },
)

builder.add_edge(
    "developer",
    "qa"
)

builder.add_conditional_edges(
    "qa",
    qa_router,
    {
        "pass": "git_prepare",
        "fail": "revision",
        "max_retries": END,
    },
)

builder.add_edge(
    "git_prepare",
    "policy"
)

builder.add_conditional_edges(
    "policy",
    route_after_policy,
    {
        "approval": "approval",
        "cleanup": "cleanup",
    },
)

builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "git_commit": "git_commit",
        "cleanup": "cleanup",
    },
)

builder.add_edge(
    "git_commit",
    END
)

builder.add_edge(
    "cleanup",
    END
)

builder.add_edge(
    "revision",
    "developer"
)


checkpointer = MemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)