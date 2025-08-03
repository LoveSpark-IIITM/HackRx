from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import os, fitz, io, urllib.parse, docx
from dotenv import load_dotenv
from typing import List, Optional, Union
from email import policy
from email.parser import BytesParser
import httpx
import asyncio
import diskcache

load_dotenv()

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

# Disk cache for storing document content
cache = diskcache.Cache("/mnt/data/cache")

# ---------------------- Request Schema ----------------------
class RunRequest(BaseModel):
    documents: Optional[Union[str, List[str]]] = None
    questions: List[str]
    email_file: Optional[str] = None

# ---------------------- Prompt Template ----------------------
PROMPT_TEMPLATE = """
You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

Context:
{context}

Question:
{query}

Instructions:
1. Provide a clear and direct answer based ONLY on the context.
2. Do not specify clause numbers or descriptions.
3. If the answer is "Yes" or "No," include a short explanation based on the clause.
4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
5. Do NOT invent or assume any information outside the given context.
6. Limit each answer to a maximum of one paragraph.
7. If the context is too long, summarize it to focus on relevant parts.

Answer:
"""

# ---------------------- Async Mistral Call ----------------------
async def call_mistral_async(context: str, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, query=question)

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.3,
        "top_p": 1,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

# ---------------------- Async Document Downloader ----------------------
async def download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

# ---------------------- Extractors from Bytes ----------------------
def extract_text_from_pdf_bytes(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join([page.get_text("text") for page in doc])

def extract_text_from_docx_bytes(data: bytes) -> str:
    file_like = io.BytesIO(data)
    document = docx.Document(file_like)
    return "\n".join([p.text for p in document.paragraphs if p.text.strip()])

def extract_text_from_txt_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")

def extract_text_from_eml_bytes(data: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(data)
    email_text = f"Subject: {msg['subject']}\n\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                email_text += part.get_content()
    else:
        email_text += msg.get_content()
    return email_text.strip()

# ---------------------- Health Check ----------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

# ---------------------- Main Endpoint ----------------------
@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        combined_text = ""
        doc_urls = [request.documents] if isinstance(request.documents, str) else (request.documents or [])

        # Concurrently fetch documents
        download_tasks = [download_bytes(url) for url in doc_urls]
        file_datas = await asyncio.gather(*download_tasks)

        for url, data in zip(doc_urls, file_datas):
            cached = cache.get(url)
            if cached:
                combined_text += "\n" + cached
                continue

            parsed_url = urllib.parse.urlparse(url)
            file_name = os.path.basename(parsed_url.path).lower()

            if file_name.endswith(".pdf"):
                extracted = extract_text_from_pdf_bytes(data)
            elif file_name.endswith(".docx"):
                extracted = extract_text_from_docx_bytes(data)
            elif file_name.endswith(".txt"):
                extracted = extract_text_from_txt_bytes(data)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported document format: {url}")

            combined_text += "\n" + extracted
            cache.set(url, extracted)

        # Optional email file
        if request.email_file:
            email_cached = cache.get(request.email_file)
            if email_cached:
                combined_text += "\n" + email_cached
            else:
                email_bytes = await download_bytes(request.email_file)
                email_text = extract_text_from_eml_bytes(email_bytes)
                combined_text += "\n" + email_text
                cache.set(request.email_file, email_text)

        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

        context = combined_text.strip()

        # Concurrent LLM calls
        tasks = [call_mistral_async(context, q) for q in request.questions]
        split_answers = await asyncio.gather(*tasks)

        return {"answers": split_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
