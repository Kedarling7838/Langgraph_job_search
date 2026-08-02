import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool   # @tool turns a function into an agent tool

load_dotenv()   # load RAPIDAPI_KEY from .env

# Read our RapidAPI key from the .env file
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
# /search-v2 is the only endpoint on this subscription (/search returns 404).
BASE_URL      = "https://jsearch.p.rapidapi.com/search-v2"

# Flip to False to run fully offline (no API calls) — the tool then serves cached
# or sample jobs only. Handy for a class demo so you never burn the 200/month quota.
USE_LIVE_API  = True

# Files kept next to this module so paths work no matter where we run from
_HERE        = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE   = os.path.join(_HERE, "jobs_cache.json")     # query -> list of jobs
SAMPLE_FILE  = os.path.join(_HERE, "sample_jobs.json")    # fallback listings


# ── Cache + sample helpers ────────────────────────────────────────────────────
def _load_cache():
    # Read the on-disk cache (query string -> list of jobs). Empty dict if none.
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(key, jobs):
    # Store a successful result so future identical searches cost zero requests
    cache = _load_cache()
    cache[key] = jobs
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass   # caching is best-effort — never let it break a search


def _load_sample():
    # Bundled realistic listings used when the live API gives us nothing
    try:
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _tag(jobs, source):
    # Mark where each job came from so the UI can label it (live/cache/sample)
    tagged = []
    for job in jobs:
        one = dict(job)
        one["_source"] = source
        tagged.append(one)
    return tagged


# ── Live API call ─────────────────────────────────────────────────────────────
def _fetch_live(params):
    """Call JSearch and return a list of slim jobs, or None if it failed/empty."""
    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

    # JSearch can be slow, so use a generous timeout and retry once on failure.
    response  = None
    for attempt in range(2):
        try:
            response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
            print("Status code:", response.status_code)
            if response.status_code == 200:
                break
            # 4xx/5xx won't be fixed by retrying — log and stop
            print("JSearch non-200:", response.status_code, response.text[:300])
            return None
        except requests.exceptions.RequestException as err:
            print("JSearch request failed (attempt " + str(attempt + 1) + "):", err)
            response = None

    if response is None or response.status_code != 200:
        return None

    # /search returns data as a list; /search-v2 nests it under data["jobs"].
    data     = response.json().get("data", [])
    all_jobs = data.get("jobs", []) if isinstance(data, dict) else data

    if not all_jobs:
        print("JSearch returned 200 but 0 jobs for query:", params["query"])
        return None

    # Keep the first 5 and slim them to just the fields we use (saves tokens)
    slim_jobs = []
    for job in all_jobs[:5]:
        slim_jobs.append({
            "job_title":           job.get("job_title", ""),
            "employer_name":       job.get("employer_name", ""),
            "job_description":     (job.get("job_description") or "")[:600],
            "job_apply_link":      job.get("job_apply_link", ""),
            "job_employment_type": job.get("job_employment_type", ""),
            "job_country":         job.get("job_country", ""),
        })
    return slim_jobs


# ── What is @tool? ────────────────────────────────────────────────────────────
# The @tool decorator registers this function as a tool the LLM can call.
# The LLM reads the docstring to understand WHEN to use it.
# The type hints (query: str) become the JSON schema the LLM uses to call it.
@tool
def search_jobs(query, location="remote"):
    """Search for real job openings matching a query and location.

    Args:
        query:    Job title or skills, e.g. 'Python data engineer'
        location: City, country, or 'remote'

    Returns:
        JSON string of up to 5 job listings.
    """
    # Build the request the way JSearch's docs want it:
    #  - remote  → use the dedicated work_from_home flag (far more results)
    #  - a place → put the location INSIDE the query ("role jobs in city")
    loc = (location or "").strip()
    if loc == "" or loc.lower() in ("remote", "anywhere", "work from home"):
        params = {
            "query":          query + " jobs",
            "page":           "1",
            "num_pages":      "1",
            "work_from_home": "true",
        }
    else:
        params = {
            "query":     query + " jobs in " + loc,
            "page":      "1",
            "num_pages": "1",
        }
    cache_key = params["query"].strip().lower()

    # 1. Try the live API (unless we're in offline demo mode)
    if USE_LIVE_API:
        live_jobs = _fetch_live(params)
        if live_jobs:
            _save_cache(cache_key, live_jobs)        # remember for next time
            return json.dumps(_tag(live_jobs, "live"))

    # 2. Live gave us nothing → reuse a previously cached result for this query
    cached = _load_cache().get(cache_key)
    if cached:
        print("Serving cached jobs for query:", cache_key)
        return json.dumps(_tag(cached, "cache"))

    # 3. Still nothing → fall back to bundled sample listings (labelled in the UI)
    print("No live/cached jobs — serving sample listings.")
    return json.dumps(_tag(_load_sample(), "sample"))