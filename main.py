from fastapi import FastAPI
import httpx
import asyncio
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# =========================
# API KEYS (ENV SAFE)
# =========================
OPENAI_KEY = os.getenv("OPENAI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# =========================
# CACHE MEMORY
# =========================
cache = {}

def hash_text(text: str):
    return hashlib.md5(text.encode()).hexdigest()


# =========================
# TASK ROUTER
# =========================
def classify(prompt: str):
    p = prompt.lower()

    if any(x in p for x in ["gambar", "image", "foto", "poster"]):
        return "vision"

    if any(x in p for x in ["review", "bagus", "design", "warna", "layout"]):
        return "reasoning"

    if any(x in p for x in ["typo", "ejaan", "grammar"]):
        return "light"

    return "general"


# =========================
# PROVIDERS
# =========================

async def openai(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception("OpenAI failed")


async def groq(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception("Groq failed")


async def gemini(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise Exception("Gemini failed")


async def mistral(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "mistral-small",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception("Mistral failed")


async def huggingface(prompt):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": prompt}
        )
        if r.status_code == 200:
            return r.json()[0]["generated_text"]
        raise Exception("HF failed")


# =========================
# FASTEST PARALLEL RUNNER
# =========================
async def fastest(prompt, providers):
    tasks = [asyncio.create_task(p(prompt)) for p in providers]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for d in done:
        try:
            result = d.result()
            for p in pending:
                p.cancel()
            return result
        except:
            continue

    return None


# =========================
# ROUTER CORE
# =========================
async def router(prompt: str):

    key = hash_text(prompt)

    # CACHE HIT
    if key in cache:
        return cache[key]

    task = classify(prompt)

    # ROUTING STRATEGY
    if task == "vision":
        providers = [gemini, openai, groq]

    elif task == "reasoning":
        providers = [openai, groq, mistral]

    elif task == "light":
        providers = [groq, openai]

    else:
        providers = [groq, openai, mistral, huggingface]

    # FAST PATH
    try:
        result = await fastest(prompt, providers)
        if result:
            cache[key] = result
            return result
    except:
        pass

    # FALLBACK PATH
    for p in [openai, groq, gemini, mistral, huggingface]:
        try:
            result = await p(prompt)
            cache[key] = result
            return result
        except:
            continue

    return "System degraded: all AI providers failed"


# =========================
# API ENDPOINT
# =========================
@app.post("/analyze")
async def analyze(data: dict):

    prompt = data.get("prompt", "")

    result = await router(prompt)

    return {
        "result": result,
        "system": "ZERO_LIMIT_AI_FINAL"
    }


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "running"}