from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import io
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="ZERO LIMIT AI", version="2.0")

# =========================
# CORS FIX (WAJIB UNTUK BOLT.NEW)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV
# =========================
GROQ_KEY = os.getenv("GROQ_API_KEY")
REMOVE_BG_KEY = os.getenv("REMOVE_BG_API_KEY")

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/debug")
def debug():
    return {
        "groq": bool(GROQ_KEY),
        "remove_bg": bool(REMOVE_BG_KEY)
    }

# =========================
# AI CORE (GROQ ONLY STABLE)
# =========================
async def ai(prompt: str):
    async with httpx.AsyncClient(timeout=60) as client:
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

        return f"AI_ERROR: {r.text}"

# =========================
# 1. TYPO TEXT
# =========================
@app.post("/typo-text")
async def typo_text(data: dict):
    text = data.get("text", "")
    result = await ai(f"Fix grammar and typos:\n{text}")
    return {"result": result}

# =========================
# 2. REVIEW IMAGE (TEXT ONLY VERSION)
# =========================
@app.post("/review-image")
async def review_image(file: UploadFile = File(...)):
    _ = await file.read()

    result = await ai(
        "You are a design expert. "
        "Analyze design quality, layout, readability, and typo. "
        "Give score 0-100 + suggestions."
    )

    return {"result": result}

# =========================
# 3. BULK REVIEW
# =========================
@app.post("/review-bulk")
async def review_bulk(data: dict):
    items = data.get("items", [])

    results = []
    for i in items:
        r = await ai(f"Review this design text:\n{i}")
        results.append(r)

    return {"results": results}

# =========================
# 4. PDF REVIEW
# =========================
@app.post("/pdf-review")
async def pdf_review(file: UploadFile = File(...)):
    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    result = await ai(f"Fix typos and improve this text:\n{text[:3000]}")

    return {"result": result}

# =========================
# 5. REMOVE BACKGROUND
# =========================
@app.post("/remove-background")
async def remove_bg(file: UploadFile = File(...)):
    image = await file.read()

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.remove.bg/v1.0/removebg",
            headers={"X-Api-Key": REMOVE_BG_KEY},
            files={"image_file": image},
            data={"size": "auto"}
        )

        if r.status_code == 200:
            return {"image_base64": r.content.hex()}

        return {"error": r.text}