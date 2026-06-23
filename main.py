from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import io
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ZERO LIMIT AI STABLE", version="3.0")

# =========================
# CORS FIX (BOLT READY)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

# =========================
# AI CORE (ANTI TOKEN OVERFLOW)
# =========================
async def ai(prompt: str):
    prompt = prompt[:2000]  # 🔥 LIMIT GLOBAL ANTI ERROR

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 500   # 🔥 penting biar gak overflow
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
    text = (data.get("text", ""))[:2000]

    result = await ai(
        "Fix grammar & typos. Be short.\n"
        "Return: corrected text + bullet mistakes.\n\n"
        f"{text}"
    )

    return {"result": result}

# =========================
# 2. REVIEW IMAGE (TEXT MODE)
# =========================
@app.post("/review-image")
async def review_image(file: UploadFile = File(...)):
    _ = await file.read()

    result = await ai(
        "Analyze design. Be very short.\n"
        "Return:\n- score (0-100)\n- 3 issues\n- 3 suggestions"
    )

    return {"result": result}

# =========================
# 3. BULK REVIEW (LIMITED)
# =========================
@app.post("/review-bulk")
async def review_bulk(data: dict):
    items = data.get("items", [])[:5]  # 🔥 LIMIT BIAR GA OVERFLOW

    results = []

    for i in items:
        i = str(i)[:500]

        r = await ai(
            "Review design text briefly:\n"
            "Give score + issue + suggestion.\n\n"
            f"{i}"
        )

        results.append(r)

    return {"results": results}

# =========================
# 4. PDF REVIEW (SAFE CHUNK)
# =========================
@app.post("/pdf-review")
async def pdf_review(file: UploadFile = File(...)):
    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf.pages[:5]:  # 🔥 max 5 pages
        text += (page.extract_text() or "")

    text = text[:2000]

    result = await ai(
        "Fix grammar & improve text. Be short.\n\n" + text
    )

    return {"result": result}

# =========================
# 5. REMOVE BG
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
            return {
                "image_base64": r.content.hex()
            }

        return {"error": r.text}