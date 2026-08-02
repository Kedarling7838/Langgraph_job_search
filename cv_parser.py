import fitz                               # pip install pymupdf  — reads PDF files
from dotenv import load_dotenv            # reads our .env file
from pydantic import BaseModel, field_validator   # define a typed data structure
from langchain.chat_models import init_chat_model   # creates a Groq LLM

load_dotenv()   # load GROQ_API_KEY from .env

# ── What is a Pydantic BaseModel? ─────────────────────────────────────────────
# It is a class where every field has a type.
# The LLM will fill these fields automatically when we call parse_cv().
class CVProfile(BaseModel):
    name:               str            # candidate full name
    current_title:      str            # e.g. "Senior Data Engineer"
    skills:             list[str]      # e.g. ["Python", "Kafka", "AWS"]
    # The LLM sometimes returns this as 6 (int) and sometimes as "6" (string).
    # Accept both so Groq's schema check passes, then normalise to an int below.
    years_experience:   int | str      # e.g. 6
    preferred_location: str            # e.g. "London" or "Remote"
    education:          str            # e.g. "BSc Computer Science"
    summary:            str            # short 1-2 sentence summary for job matching

    @field_validator("years_experience", mode="before")
    @classmethod
    def _coerce_years(cls, value):
        # Turn "1", "1 year", 1, etc. into a plain int; default to 0 if unknown
        if isinstance(value, int):
            return value
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0


# Create the LLM once so we don't create it again and again
llm = init_chat_model("groq:llama-3.3-70b-versatile")


def extract_text(pdf_path):
    # Open the PDF file and read text from every page
    all_text = ""
    pdf_file = fitz.open(pdf_path)        # open the PDF
    for page in pdf_file:                  # loop over every page
        page_text = page.get_text()        # read text from this page
        all_text = all_text + page_text    # add it to our big string
    pdf_file.close()
    return all_text


def parse_cv(cv_text):
    # Ask the LLM to read the raw CV text and fill in the CVProfile fields
    prompt = "Extract the candidate profile from this CV text:\n\n" + cv_text

    # with_structured_output tells the LLM: reply ONLY with JSON that matches CVProfile
    parser = llm.with_structured_output(CVProfile)
    profile = parser.invoke(prompt)        # send prompt, get back a CVProfile object
    return profile