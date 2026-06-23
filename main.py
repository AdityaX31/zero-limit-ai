from fastapi import FastAPI, UploadFile, File
import os
import httpx
import asyncio
from dotenv import load_dotenv
from pypdf import PdfReader
import io

load_dotenv()

app = FastAPI(title="Design Reviewer AI PRO", version="1.0")

# =========================
# ENV
# =========================
OPENAI_KEY = os.getenv("OPENAI_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")
REMOVE_BG_KEY = os.getenv("REMOVE_BG_API_KEY")

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    return {
        "openai": bool(OPENAI_KEY),
        "groq": bool(GROQ_KEY),
        "gemini": bool(GEMINI_KEY),
        "mistral": bool(MISTRAL_KEY),
        "hf": bool(HF_KEY),
        "remove_bg": bool(REMOVE_BG_KEY)
    }

# =========================
# AI CALLS (GROQ ONLY SIMPLE CORE)
# =========================
async def groq(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return "Groq error"

# fallback simple (biar stabil dulu)
async def ai(prompt):
    try:
        return await groq(prompt)
    except:
        return "AI failed"

# =========================
# 1. TYPO TEXT
# =========================
@app.post("/typo-text")
async def typo_text(data: dict):
    prompt = data.get("text", "")
    result = await ai(f"Check grammar & typo and fix this text:\n{prompt}")
    return {"result": result}

# =========================
# 2. REVIEW IMAGE (SIMPLIFIED VISION PROMPT)
# =========================
@app.post("/review-image")
async def review_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    prompt = """
You are a design expert.
Analyze this image:
- typo
- layout
- color
- readability
Give score 0-100 and suggestions.
Return clean structured answer.
"""

    result = await ai(prompt + "\n(image attached not processed in this simple version)")
    return {
        "result": result,
        "note": "vision placeholder (upgrade next)"
    }

# =========================
# 3. BULK REVIEW
# =========================
@app.post("/review-bulk")
async def review_bulk(data: dict):
    items = data.get("items", [])
    result = []

    for i in items:
        r = await ai(f"Review this design text:\n{i}")
        result.append(r)

    return {"results": result}

# =========================
# 4. PDF REVIEW (TEXT ONLY)
# =========================
@app.post("/pdf-review")
async def pdf_review(file: UploadFile = File(...)):
    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    result = await ai(f"Check typo and improve this PDF text:\n{text[:3000]}")

    return {"result": result}

# =========================
# 5. REMOVE BACKGROUND
# =========================
@app.post("/remove-background")
async def remove_bg(file: UploadFile = File(...)):
    image = await file.read()

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.remove.bg/v1.0/removebg",
            headers={"X-Api-Key": REMOVE_BG_KEY},
            files={"image_file": image},
            data={"size": "auto"}
        )

        if r.status_code == 200:
            return {"image_bytes": r.content.hex()}

        return {"error": r.text}