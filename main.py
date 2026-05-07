import io
import re
import pdfplumber
import spacy
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- THE IMPORT
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Initialize FastAPI with professional metadata
app = FastAPI(
    title="Semantic Resume Screening AI", 
    description="Context-aware NLP engine for matching resumes to job descriptions.",
    version="1.0.0"
)

# --- CORS VIP PASS ---
# Notice how we are using CORSMiddleware right here. 
# This will make the "not accessed" warning disappear immediately!
# --- CORS VIP PASS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ai-recruiter-frontend-nwgs.vercel.app" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------

# ---------------------------------------------------------
# AI MODEL INITIALIZATION (Singleton Pattern)
# ---------------------------------------------------------
print("Loading AI Models into memory... This may take a minute.")
# Initialize FastAPI with professional metadata
app = FastAPI(
    title="Semantic Resume Screening AI", 
    description="Context-aware NLP engine for matching resumes to job descriptions.",
    version="1.0.0"
)

# ---------------------------------------------------------
# AI MODEL INITIALIZATION (Singleton Pattern)
# ---------------------------------------------------------
print("Loading AI Models into memory... This may take a minute.")
nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("Models loaded successfully. Server ready.")


# ---------------------------------------------------------
# PYDANTIC SCHEMAS (Strict Type Validation)
# ---------------------------------------------------------
class ScreeningResult(BaseModel):
    filename: str
    match_score_percentage: float
    extracted_email: str | None
    extracted_phone: str | None
    key_entities: list[str]


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from PDF bytes safely."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable PDF: {str(e)}")
    
    return text.strip()

def extract_contact_info(text: str) -> dict:
    """Uses Regex to find standard contact identifiers."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None
    }



# API ENDPOINTS

@app.post("/api/v1/screen-resume", response_model=ScreeningResult)
async def screen_resume(
    resume_pdf: UploadFile = File(...), 
    job_description: str = Form(...)
):
  
    # 1. Read and Extract Text
    file_bytes = await resume_pdf.read()
    resume_text = extract_text_from_pdf(file_bytes)
    
    if not resume_text:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

    # 2. Named Entity Recognition (NER) via spaCy
    doc = nlp(resume_text)
    entities = list(set([ent.text.strip() for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]))
    contact_info = extract_contact_info(resume_text)

    # 3. Deep Semantic Matching via HuggingFace Transformers
    resume_embedding = encoder.encode(resume_text, convert_to_tensor=True)
    jd_embedding = encoder.encode(job_description, convert_to_tensor=True)

    # Calculate Cosine Similarity between the vectors
    cosine_scores = util.cos_sim(resume_embedding, jd_embedding)
    match_score = float(cosine_scores[0][0]) * 100
    match_score = max(0.0, match_score)

    # 4. Return structured JSON
    return ScreeningResult(
        filename=resume_pdf.filename,
        match_score_percentage=round(match_score, 2),
        extracted_email=contact_info["email"],
        extracted_phone=contact_info["phone"],
        key_entities=entities[:15] 
    )