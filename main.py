from fastapi import FastAPI
import httpx
import asyncio
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ZERO LIMIT AI", version="2.0")

# =========================
# ENV
# =========================
OPENAI_KEY = os.getenv("OPENAI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

cache = {}
ACTIVE_PROVIDERS = []

def hash_text(text: str):
    return hashlib.md5(text.encode()).hexdigest()

# =========================
# INIT CHECK PROVIDERS
# =========================
async def check_provider(name, func):
    try:
        await func("hi")
        ACTIVE_PROVIDERS.append(func)
        print(f"{name} ACTIVE")
    except:
        print(f"{name} DEAD")

@app.on_event("startup")
async def startup():
    global ACTIVE_PROVIDERS
    ACTIVE_PROVIDERS = []

    await check_provider("groq", groq_provider)
    await check_provider("openai", openai_provider)
    await check_provider("gemini", gemini_provider)
    await check_provider("mistral", mistral_provider)
    await check_provider("hf", hf_provider)

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "running", "system": "ZERO_LIMIT_AI_V2"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    return {
        "openai": bool(OPENAI_KEY),
        "groq": bool(GROQ_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "mistral": bool(MISTRAL_API_KEY),
        "hf": bool(HUGGINGFACE_API_KEY),
        "active_providers": len(ACTIVE_PROVIDERS)
    }

# =========================
# CLASSIFY
# =========================
def classify(prompt: str):
    p = prompt.lower()

    if any(x in p for x in ["gambar", "image", "foto", "poster"]):
        return "vision"

    if any(x in p for x in ["review", "design", "layout", "warna"]):
        return "reasoning"

    if any(x in p for x in ["typo", "ejaan", "grammar"]):
        return "light"

    return "general"

# =========================
# PROVIDERS
# =========================
async def openai_provider(prompt):
    if not OPENAI_KEY:
        raise Exception("NO_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
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
        raise Exception(r.text)

async def groq_provider(prompt):
    if not GROQ_API_KEY:
        raise Exception("NO_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
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
        raise Exception(r.text)

async def gemini_provider(prompt):
    if not GEMINI_API_KEY:
        raise Exception("NO_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise Exception(r.text)

async def mistral_provider(prompt):
    if not MISTRAL_API_KEY:
        raise Exception("NO_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception(r.text)

async def hf_provider(prompt):
    if not HUGGINGFACE_API_KEY:
        raise Exception("NO_KEY")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": prompt}
        )
        if r.status_code == 200:
            data = r.json()
            return data[0].get("generated_text", "")
        raise Exception(r.text)

# =========================
# FASTEST ENGINE
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
            pass

    return None

# =========================
# ROUTER
# =========================
async def router(prompt: str):

    key = hash_text(prompt)

    if key in cache:
        return cache[key]

    task = classify(prompt)

    providers_map = {
        "vision": [gemini_provider, openai_provider, groq_provider],
        "reasoning": [openai_provider, groq_provider, mistral_provider],
        "light": [groq_provider, openai_provider],
        "general": ACTIVE_PROVIDERS if ACTIVE_PROVIDERS else [groq_provider]
    }

    providers = providers_map.get(task, providers_map["general"])

    try:
        result = await fastest(prompt, providers)
        if result:
            cache[key] = result
            return result
    except:
        pass

    for p in [openai_provider, groq_provider, gemini_provider, mistral_provider, hf_provider]:
        try:
            result = await p(prompt)
            cache[key] = result
            return result
        except:
            continue

    return {
        "error": "ALL_PROVIDERS_FAILED"
    }

# =========================
# API
# =========================
@app.post("/analyze")
async def analyze(data: dict):
    prompt = data.get("prompt", "")
    result = await router(prompt)

    return {
        "result": result,
        "system": "ZERO_LIMIT_AI_V2"
    }