# rag_youtube.py
import os
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ------------------ Load environment ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ------------------ Configure logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Initialize models ------------------
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=GOOGLE_API_KEY)
    llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=GOOGLE_API_KEY, temperature=0.2)
except Exception as e:
    logger.error("Failed to initialize AI models: %s", e)
    embeddings_model = None
    llm_model = None

# ------------------ Utility functions ------------------
def extract_video_id(url: str):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        if parsed_url.hostname == "youtu.be":
            return parsed_url.path.lstrip("/")
        return None
    except Exception as e:
        logger.error("Error extracting video ID from URL '%s': %s", url, e)
        return None

def split_text(transcript: str):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(transcript)
    except Exception as e:
        logger.error("Error splitting transcript: %s", e)
        return [transcript]

def transcript_from_youtube(video_url: str = "") -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.warning("No video ID extracted from URL: %s", video_url)
        return "No transcript available for this video"

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=['ne', 'hi', 'en'])
        transcript_text = " ".join(chunk.text for chunk in transcript.snippets)
        return transcript_text.replace("\xa0", " ").strip()
    except TranscriptsDisabled:
        logger.warning("Transcripts are disabled for video: %s", video_id)
        return "No transcript available for this video"
    except NoTranscriptFound:
        logger.warning("No transcript found for video: %s", video_id)
        return "No transcript available for this video"
    except Exception as e:
        logger.error("Unexpected error fetching transcript for video %s: %s", video_id, e)
        return "No transcript available for this video"

def embed_chunks(chunks):
    if not embeddings_model:
        logger.error("Embedding model not initialized")
        return None
    try:
        # Create in-memory vector store (per video)
        docs = [Document(page_content=chunk, metadata={"source": "youtube"}) for chunk in chunks]
        db = Chroma.from_documents(docs, embeddings_model, persist_directory=None)
        return db
    except Exception as e:
        logger.error("Error creating vector embeddings: %s", e)
        return None

def format_context(documents):
    try:
        return "".join(doc.page_content for doc in documents)
    except Exception as e:
        logger.error("Error formatting context: %s", e)
        return ""

# ------------------ Core function ------------------
def answer_youtube_question(video_url: str, question: str) -> str:
    if not llm_model or not embeddings_model:
        return "AI models are not properly initialized."

    # ------------------ Fetch transcript ------------------
    transcript = transcript_from_youtube(video_url)
    if not transcript or "No transcript available" in transcript:
        return "No transcript available for this video"

    # ------------------ Split transcript into chunks ------------------
    chunks = split_text(transcript)

    # ------------------ Create fresh vector store per video ------------------
    vector_store = embed_chunks(chunks)
    if not vector_store:
        return "Failed to process transcript embeddings"

    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

        # ------------------ Prompt definition ------------------
        prompt = PromptTemplate(
            template='''
You are a specialized Question Answering AI. Answer questions based solely on the provided transcript. Follow these rules:

1. Context-Grounded: Base your answers on the transcript. You may use external knowledge only to clarify or explain prerequisite concepts.
2. Language: Write answers in English only, regardless of transcript language (Hindi, Nepali, English).
3. Direct Answers: Always answer directly. Do not include phrases like "According to the transcript" or "Based on the text."
4. Insufficient Data: If the transcript does not provide enough information, respond naturally, indicating the answer is not fully clear from the video, without saying "I don't know."
5. No Extra Content: Avoid personal opinions, apologies, filler, or general summaries unless explicitly asked.
6. Summaries: If asked to summarize, provide a detailed, comprehensive summary strictly based on the transcript.

Transcript Context: {context}
Question: {question}

            ''',
            input_variables=['context', 'question']
        )

        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_context),
                "question": RunnablePassthrough()
            }
        )

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm_model | parser

        # ------------------ Invoke LLM ------------------
        return main_chain.invoke(question)

    except Exception as e:
        logger.error("Error answering question: %s", e)
        return "An error occurred while generating the answer"
