from functools import lru_cache
from urllib import response
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# from my_agent.utils.tools import tools
# from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from my_agent.utils.state import AgentState_er
from typing import List
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    try:
        if model_name == "openai":
            model = ChatOpenAI(temperature=0, model_name="gpt-4o") 
        elif model_name == "anthropic":
            model =  ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
        elif model_name == "gemini":
            model =  ChatGoogleGenerativeAI(model_name="gemini-pro", temperature=0)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise e
    # model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    #response = agent.run(input=messages)  # Use agent.run for tool usage
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
# tool_node = ToolNode(tools)

## Prompts for essay writer ##

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

class Queries(BaseModel):
    queries: List[str]

def plan_node(state: AgentState_er, config: dict):
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    # Create a list of messages to send to the model. The first message is a
    # system message with the prompt for the plan node. The second message is a
    # human message with the user's task.
    #
    # The model will be invoked with this list of messages, and the response will
    # be the plan output by the model.
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    print("Exiting plan node with:", response.content)
    return {"plan": response.content}


def research_plan_node(state: AgentState_er, config: dict):
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        print(f"Query: {q}")
        # response = ToolNode([TavilySearchResults(max_results=3)]).invoke([{"query": q }])
        response = None # stubbing out
        print(f"Response: {response}")
        #for r in response['results']:
        #    content.append(r['content'])
    print("Exiting research plan node with:", content)
    return {"content": content}

def generation_node(state: AgentState_er, config: dict):
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    try : 
        if state["revision_number"] is None:
            state["revision_number"] = 0
        current_revision_number = state["revision_number"]
        print(f"Current revision number: {current_revision_number}")
    except Exception as e:
        print(f"Error: {e}")
    print("Generation node invoked with messages:", messages)
    response = model.invoke(messages)
    current_revision_number += 1
    print("Exiting generation node with:", response.content)
    return {
        "draft": response.content, 
        "revision_number": current_revision_number
    }

def reflection_node(state: AgentState_er, config: dict):
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    print("Exiting reflection node with:", response.content)
    return {"critique": response.content}

def research_critique_node(state: AgentState_er, config: dict):
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        # response = ToolNode([TavilySearchResults(max_results=3)]).invoke([q])
        response = None # stubbing out
        # for r in response['results']:
        #     content.append(r['content'])
    print("Exiting research critique node with:", content)
    return {"content": content}

def should_continue_er(state):
    if "max_revisions" not in state or state["max_revisions"] is None :
        state["max_revisions"] = 0
    if state["max_revisions"] == 0 :
        state["max_revisions"] = 3 # default number of revisions
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

