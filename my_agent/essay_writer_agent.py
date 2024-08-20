from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Literal
from my_agent.utils.nodes import should_continue_er, plan_node, generation_node, reflection_node, research_plan_node, research_critique_node
from my_agent.utils.state import AgentState_er
import operator

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai", "gemini"]

builder = StateGraph(AgentState_er, config_schema=GraphConfig)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", 
    should_continue_er, 
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile()

