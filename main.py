# main.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from youtube_bot_llm import answer_youtube_question
import uvicorn

# ------------------ Configure logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ FastAPI app ------------------
app = FastAPI(title="YouTube RAG QA API")

# Add CORS middleware for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------ Request model ------------------
class QARequest(BaseModel):
    youtube_url: HttpUrl
    question: str

# ------------------ CORS preflight handler ------------------
@app.options("/ask")
async def options_ask():
    return {"message": "OK"}

# ------------------ POST /ask endpoint ------------------
@app.post("/ask")
async def ask_question(payload: QARequest):
    youtube_url = str(payload.youtube_url)

    # validate only YouTube
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        logger.warning("Invalid YouTube URL received: %s", youtube_url)
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported")

    try:
        answer = answer_youtube_question(youtube_url, payload.question)
    except Exception as e:
        logger.error("Error processing question for URL %s: %s", youtube_url, e)
        raise HTTPException(status_code=500, detail="Internal server error")

    if not answer or "No transcript available" in answer:
        logger.info("Transcript not available for video: %s", youtube_url)
        raise HTTPException(status_code=404, detail="Transcript not available for this video")

    return {"answer": answer}

# ------------------ Health check ------------------
@app.get("/")
async def root():
    return {"message": "YouTube AI Summarizer API is running!"}

# ------------------ Run server ------------------
