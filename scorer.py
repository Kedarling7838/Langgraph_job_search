import json
import numpy as np
from fastembed import TextEmbedding
from langchain_core.messages import ToolMessage

# We score jobs with EMBEDDINGS: turn the CV and each job into a list of numbers
# that captures its meaning, then measure how aligned they are (cosine similarity).
# fastembed runs a small model locally — free, no API key, no PyTorch crash.
model = TextEmbedding("BAAI/bge-small-en-v1.5")   # first run downloads ~130 MB, then cached


def score_jobs(cv_profile, messages):
    # 1. Collect every job the agent found (each search result is a ToolMessage).
    jobs = []
    for message in messages:
        if isinstance(message, ToolMessage):
            jobs += json.loads(message.content)
    if not jobs:
        return []

    # 2. One line of text for the CV, and one for each job.
    cv_text   = cv_profile["current_title"] + " " + " ".join(cv_profile["skills"])
    job_texts = [job["job_title"] + " " + job["job_description"] for job in jobs]

    # 3. Turn every text into an embedding (a list of numbers = its meaning).
    vectors     = list(model.embed([cv_text] + job_texts))
    cv_vector   = vectors[0]
    job_vectors = vectors[1:]

    # 4. Score each job by how close its meaning is to the CV (cosine similarity, 0-100%).
    for job, vector in zip(jobs, job_vectors):
        similarity = np.dot(cv_vector, vector) / (np.linalg.norm(cv_vector) * np.linalg.norm(vector))
        job["match_score"] = round(float(similarity) * 100, 1)

    # 5. Best matches first — keep the top 5.
    jobs.sort(key=lambda job: job["match_score"], reverse=True)
    return jobs[:5]