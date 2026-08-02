#   streamlit run streamlit_app.py

import json
import tempfile
import streamlit as st
from langchain_core.messages import ToolMessage

from cv_parser  import extract_text, parse_cv
from agent_core import build_graph
from opik.integrations.langchain import OpikTracer   # traces every LLM call to Opik

# ── Page title and description ────────────────────────────────────────────────
st.title("LangGraph Job Search Agent")
st.write("Upload your CV and the agent searches real job listings personalised to your profile.")

# ── CV Upload ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

# These variables hold the CV data — start empty
cv_text    = ""
cv_profile = None

if uploaded_file:
    # Save the uploaded file to a temporary file on disk so PyMuPDF can read it
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())   # write uploaded bytes to the temp file
    temp_file.close()
    temp_path = temp_file.name              # remember the file path

    with st.spinner("Reading your CV..."):
        cv_text    = extract_text(temp_path)   # read raw text from the PDF
        cv_profile = parse_cv(cv_text)         # LLM extracts structured profile

    st.success("CV parsed — " + cv_profile.name + ", " + cv_profile.current_title)

    # Show the extracted profile in a collapsible section
    with st.expander("Extracted Profile"):
        skills_text = ", ".join(cv_profile.skills)
        st.write("**Skills:** " + skills_text)
        st.write("**Experience:** " + str(cv_profile.years_experience) + " years")
        st.write("**Education:** " + cv_profile.education)
        st.write("**Preferred location:** " + cv_profile.preferred_location)
        st.write("**Summary:** " + cv_profile.summary)

# ── Run the agent ─────────────────────────────────────────────────────────────
# Only show the button if a CV has been uploaded
if cv_text and st.button("Find Matching Jobs"):

    app        = build_graph()
    steps_log  = []   # we will collect every step here
    sources    = []   # we will collect every job link here
    debug_raw  = []   # raw JSON the tool returned on each call (for debugging)
    api_errors = []   # any API error objects the tool reported

    # Opik records every LLM call (tokens, cost, latency) to the dashboard.
    # It reads OPIK_API_KEY from .env automatically. We pass it to stream() as a
    # callback so LangGraph reports each node and each LLM call as it runs.
    tracer      = OpikTracer()
    opik_config = {"callbacks": [tracer]}

    with st.status("Agent is searching for jobs...", expanded=True) as status:

        for step in app.stream({
            "cv_text":     cv_text,
            "messages":    [],
            "cv_profile":  {},
            "scored_jobs": [],
        }, config=opik_config):
            # Each step is a dict with exactly one key = the node name
            # e.g. {"agent": {"messages": [...]}}
            node_name   = list(step.keys())[0]     # get the node name
            node_output = list(step.values())[0]   # get the node output

            steps_log.append((node_name, node_output))   # save for later

            # ── Show progress for each node ────────────────────────────────────
            if node_name == "parse_cv":
                extracted_profile = node_output.get("cv_profile", {})
                candidate_name    = extracted_profile.get("name", "candidate")
                st.write("**parse_cv** → profile extracted for " + candidate_name)

            elif node_name == "agent":
                last_message = node_output["messages"][-1]
                if last_message.tool_calls:
                    # The LLM decided to search — show what it is searching for
                    for tool_call in last_message.tool_calls:
                        query    = tool_call["args"].get("query", "")
                        location = tool_call["args"].get("location", "")
                        st.write("**agent** → searching: `" + query + "` in `" + location + "`")
                else:
                    st.write("**agent** → searches complete, scoring results")

            elif node_name == "tools":
                # The tool ran — its result is either a list of jobs OR an error
                # object ({"error": ..., "status_code": ...}). Handle both.
                for message in node_output["messages"]:
                    # isinstance checks: "is this a ToolMessage (tool result)?"
                    if isinstance(message, ToolMessage):
                        debug_raw.append(message.content)   # keep raw for debug panel
                        try:
                            payload = json.loads(message.content)
                        except Exception:
                            payload = None

                        if isinstance(payload, dict) and payload.get("error"):
                            # The API failed — show the real reason immediately
                            api_errors.append(payload)
                            st.error(
                                "JSearch API error — status "
                                + str(payload.get("status_code"))
                                + ": " + str(payload.get("error"))
                            )
                            if payload.get("body"):
                                st.caption("Response body: " + str(payload.get("body"))[:300])
                        elif isinstance(payload, list):
                            if len(payload) == 0:
                                st.warning("JSearch returned 0 jobs for this query.")
                            for job in payload:
                                sources.append({
                                    "title": job.get("job_title", "Job"),
                                    "url":   job.get("job_apply_link", "#"),
                                })
                st.write("**tools** → retrieved " + str(len(sources)) + " listing(s) so far")

            elif node_name == "score":
                scored = node_output.get("scored_jobs", [])
                st.write("**score** → embeddings ranked top " + str(len(scored)) + " matches")

            elif node_name == "format":
                st.write("**format** → writing personalised summary")

        status.update(label="Done — " + str(len(steps_log)) + " steps", state="complete")

    # Make sure every trace is sent to Opik, then link to the dashboard so you
    # can inspect tokens, cost and latency for this run.
    tracer.flush()
    st.link_button("🔍 View traces in Opik", "https://www.comet.com/opik")
    st.caption("Every LLM call in this run (parse_cv, agent, format) is traced in Opik.")

    # ── Extract final results from steps_log ──────────────────────────────────
    # The final message is the LLM's summary (last message in the last step)
    last_node_name, last_node_output = steps_log[-1]
    final_answer = last_node_output["messages"][-1].content

    # Find the scored_jobs — they live in the "score" node's output
    scored_jobs = []
    for node_name, node_output in steps_log:
        if "scored_jobs" in node_output:
            scored_jobs = node_output["scored_jobs"]
            break   # we found them, no need to keep looping

    # ── Display top matched jobs ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Top Matched Jobs")

    # If the live API returned nothing, the tool falls back to bundled sample
    # listings — make that unmistakable so nobody mistakes them for live jobs.
    is_sample = any(job.get("_source") == "sample" for job in scored_jobs)
    is_cache  = any(job.get("_source") == "cache" for job in scored_jobs)
    if is_sample:
        st.warning(
            "⚠️ Showing **SAMPLE** jobs — the live JSearch API returned no results "
            "(quota / endpoint issue). The Apply links below are realistic examples, "
            "not live postings."
        )
    elif is_cache:
        st.info("Showing **cached** results from a previous search (no new API request made).")

    if not scored_jobs:
        if api_errors:
            first = api_errors[0]
            st.error(
                "No jobs because the JSearch API failed — status "
                + str(first.get("status_code")) + ". "
                "A 429 means you hit the rate limit or monthly quota (200/month); "
                "403 means the endpoint isn't in your plan; a timeout means the API was slow."
            )
        else:
            st.info("The API returned 0 jobs for these queries — try running again.")

    # ── Debug: raw API responses ───────────────────────────────────────────────
    # Expand this to see exactly what JSearch returned on each tool call.
    with st.expander("Debug — raw API responses"):
        if debug_raw:
            for i, raw in enumerate(debug_raw):
                st.markdown("**Tool call " + str(i + 1) + ":**")
                st.code(raw[:1500], language="json")
        else:
            st.caption("No tool calls were made.")

    for i, job in enumerate(scored_jobs):
        # Coerce every field to a safe string — the API can return None
        job_title   = job.get("job_title")   or "Unknown role"
        employer    = job.get("employer_name") or "Unknown employer"
        match_score = job.get("match_score", 0)
        description = (job.get("job_description") or "")[:500]
        apply_link  = job.get("job_apply_link") or ""
        job_type    = job.get("job_employment_type") or "N/A"
        country     = job.get("job_country") or "N/A"

        # Card header shows rank, role, employer and the match score
        card_title = (
            str(i + 1) + ". " + job_title + " — " + employer
            + "  ·  " + str(match_score) + "% match"
        )

        with st.expander(card_title, expanded=(i == 0)):
            # Match score as a prominent metric
            st.metric(label="Match score", value=str(match_score) + "%")

            # Job meta shown side by side
            col_type, col_country = st.columns(2)
            col_type.write("**Type:** " + job_type)
            col_country.write("**Country:** " + country)

            if description:
                st.write(description + "...")

            # Apply link rendered as a real button (falls back to a note)
            if apply_link:
                st.link_button("Apply for this job  →", apply_link)
            else:
                st.caption("No apply link provided for this listing.")

    # ── Agent summary ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Agent Summary")
    st.markdown(final_answer)

    # ── All sources found ──────────────────────────────────────────────────────
    if sources:
        st.markdown("---")
        st.markdown("### All Listings Found")
        st.caption("Every job the agent pulled from JSearch, with a direct link:")
        for source in sources:
            title = source.get("title") or "Job"
            url   = source.get("url") or "#"
            st.markdown("- [" + title + "](" + url + ")")