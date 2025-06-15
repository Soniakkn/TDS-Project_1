from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pytesseract
from PIL import Image
import base64
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import subprocess
import os
import io

# === Configure Tesseract ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Initialize FastAPI app ===
app = FastAPI()

# === Load FAISS index and chunks ===
index = faiss.read_index("tds_index.faiss")
with open("tds_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# === Load embedding model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === API request schema ===
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image string
    link: Optional[str] = None   # optional link

# === Helper: Run LLM using Ollama (mistral) ===
def query_llm(prompt: str) -> str:
    try:
        result = subprocess.run(
            [r"C:\Users\Sigma\AppData\Local\Programs\Ollama\ollama.exe", "run", "mistral"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25
        )
        output = result.stdout.decode()
        return output.split("Answer:")[-1].strip()
    except Exception as e:
        return f"LLM error: {str(e)}"

# === Helper: Extract text from base64 image ===
def extract_text_from_base64(base64_image: str) -> str:
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""

# === API endpoint ===
@app.post("/api/")
async def get_answer(req: QuestionRequest):
    # Step 1: Extract OCR text (if image provided)
    ocr_text = extract_text_from_base64(req.image) if req.image else ""

    # Step 2: Combine image text and question
    full_question = req.question.strip()
    if ocr_text:
        full_question += "\n" + ocr_text

    # Step 3: Embed question and retrieve similar content
    query_vec = embedding_model.encode([full_question])
    D, I = index.search(query_vec, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]

    # Step 4: Build prompt
    context = "\n---\n".join(relevant_chunks)
    prompt = f"""You are a helpful Virtual Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras.

Relevant Information:
{context}
"""
    if req.link:
        prompt += f"\nRelated Discussion Link:\n{req.link}"

    prompt += f"""

Question:
{full_question}

Answer:"""

    # Step 5: Get answer from LLM
    answer = query_llm(prompt)

    # Step 6: Extract links from answer
    links = []
    for line in answer.splitlines():
        if "http" in line:
            url_part = line.split("http", 1)[-1].split()[0].strip(").\"")
            url = "http" + url_part
            label = line.split("http")[0].strip("-•: ") or "Relevant link"
            links.append({"url": url, "text": label})

    # Step 7: Return structured response
    return {
        "answer": answer,
        "links": links
    }
