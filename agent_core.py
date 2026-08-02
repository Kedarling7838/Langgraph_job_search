import json
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from cv_parser import parse_cv       # our CV extraction + Pydantic parsing
from job_tool  import search_jobs    # our JSearch API tool
from scorer    import score_jobs     # our local-embedding scoring

load_dotenv()


# ── State ─────────────────────────────────────────────────────────────────────
# State is the shared memory that all nodes can read and write.
# Every field here travels through the whole graph.
class State(TypedDict):
    messages:    Annotated[list[AnyMessage], add_messages]  # message history (appends)
    cv_text:     str        # the raw text extracted from the PDF
    cv_profile:  dict       # the structured profile (name, skills, etc.)
    scored_jobs: list       # the final ranked job list


# ── LLM and tools ─────────────────────────────────────────────────────────────
llm            = init_chat_model("groq:llama-3.3-70b-versatile")
tools          = [search_jobs]
llm_with_tools = llm.bind_tools(tools)   # tell the LLM it can use search_jobs


# ── Node 1: parse_cv_node ─────────────────────────────────────────────────────
# Reads the raw CV text from state, asks the LLM to extract the profile.
def parse_cv_node(state):
    profile = parse_cv(state["cv_text"])          # returns a CVProfile object
    profile_dict = profile.model_dump()            # convert Pydantic object to plain dict
    return {"cv_profile": profile_dict}


# ── Node 2: agent_node ────────────────────────────────────────────────────────
# Same idea as Class 1 agent node, but we inject the CV profile as a system message
# so the LLM knows WHO it is searching for.
def agent_node(state):
    profile = state["cv_profile"]

    # SystemMessage = background instructions given to the LLM before the conversation
    profile_text = json.dumps(profile, indent=2)   # convert dict to readable JSON string
    system_message = SystemMessage(content=(
        "You are a job search assistant.\n"
        "Candidate profile:\n" + profile_text + "\n\n"
        "Make 2-3 targeted job searches using the search_jobs tool. "
        "Vary your queries to cover different angles of their skills and title.\n"
        "IMPORTANT: for the location argument use 'remote' or a MAJOR tech hub "
        "such as Bangalore, Hyderabad, Pune or Mumbai. Do NOT use the candidate's "
        "small home town — it usually has zero listings and returns no results."
    ))

    # Combine system message + conversation history
    all_messages = [system_message] + state["messages"]

    # Ask the LLM: what should we do next? (call a tool, or give a final answer?)
    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}


# ── Routing function ──────────────────────────────────────────────────────────
# Called after agent_node. Decides which node to go to next.
def should_continue(state):
    last_message = state["messages"][-1]   # look at the most recent message
    if last_message.tool_calls:
        return "tools"   # LLM wants to call search_jobs → go to tools node
    else:
        return "score"   # LLM is done searching → go to score node


# ── Node 3: score_node ────────────────────────────────────────────────────────
# Reads all search results from the message history.
# Uses local embeddings (fastembed) to rank them by how well they match the CV.
def score_node(state):
    scored = score_jobs(state["cv_profile"], state["messages"])
    return {"scored_jobs": scored}


# ── Node 4: format_node ───────────────────────────────────────────────────────
# Reads the top-ranked jobs and asks the LLM to write a nice summary.
def format_node(state):
    jobs    = state["scored_jobs"]
    profile = state["cv_profile"]

    # Build a text block listing each job
    job_lines = []
    for i, job in enumerate(jobs):
        number      = str(i + 1)
        title       = job.get("job_title", "")
        employer    = job.get("employer_name", "")
        score       = str(job.get("match_score", ""))
        description = job.get("job_description", "")[:300]
        apply_link  = job.get("job_apply_link", "N/A")

        line = number + ". " + title + " at " + employer + " [match: " + score + "%]\n"
        line = line + "   " + description + "\n"
        line = line + "   Apply: " + apply_link
        job_lines.append(line)

    jobs_text = "\n\n".join(job_lines)   # join all job blocks with a blank line between

    # Build the skills string safely
    skills_list = profile.get("skills", [])
    skills_text = ", ".join(skills_list)

    # Ask the LLM to write an encouraging summary
    prompt = (
        "You are helping " + profile.get("name", "the candidate") + " find a job.\n"
        "Their title: " + profile.get("current_title", "") + "\n"
        "Skills: " + skills_text + "\n\n"
        "Here are the top matched jobs:\n" + jobs_text + "\n\n"
        "Write a short, encouraging summary of these results and why each is a good fit."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(State)

    # Add all nodes
    graph.add_node("parse_cv", parse_cv_node)
    graph.add_node("agent",    agent_node)
    graph.add_node("tools",    ToolNode(tools))   # prebuilt — runs any tool automatically
    graph.add_node("score",    score_node)
    graph.add_node("format",   format_node)

    # Add edges (connections between nodes)
    graph.add_edge(START,       "parse_cv")   # start → parse CV
    graph.add_edge("parse_cv",  "agent")      # parse done → start searching
    graph.add_conditional_edges("agent", should_continue)  # loop or move to scoring
    graph.add_edge("tools",     "agent")      # tool result → back to agent (the loop)
    graph.add_edge("score",     "format")     # scoring done → write summary
    graph.add_edge("format",    END)          # summary written → done

    return graph.compile()


# ── Helper: run the agent and return results ──────────────────────────────────
def run(cv_text):
    app = build_graph()

    # The starting state — we only set cv_text, everything else starts empty
    starting_state = {
        "cv_text":     cv_text,
        "messages":    [],
        "cv_profile":  {},
        "scored_jobs": [],
    }

    result      = app.invoke(starting_state)
    final_text  = result["messages"][-1].content   # the LLM's summary
    scored_jobs = result.get("scored_jobs", [])

    return final_text, scored_jobs


# ── Helper: stream steps one by one ──────────────────────────────────────────
def stream_steps(cv_text):
    app = build_graph()

    starting_state = {
        "cv_text":     cv_text,
        "messages":    [],
        "cv_profile":  {},
        "scored_jobs": [],
    }

    for step in app.stream(starting_state):
        yield step