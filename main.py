from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import os, requests, fitz
import tempfile
from dotenv import load_dotenv

# LangChain / Embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")


# Request schema
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# Prompt
PROMPT_TEMPLATE = """
You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

Context:
{context}

Question:
{query}

Instructions:
1. Provide a clear and direct answer based ONLY on the context.
2. Donot specify the clause number or clause description.
3. If the answer is "Yes" or "No," include a short explanation based on the clause.
4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
5. Do NOT invent or assume any information outside the given context.
6. Limit the answer to maximum upto one paragraph.
7. If the context is too long, summarize it to focus on the relevant parts.

Answer:
"""

# Mistral API
def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# PDF text extraction
def extract_text_from_url(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    text = ""
    doc = fitz.open(tmp_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    os.remove(tmp_path)
    return text.strip()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}


# FastAPI endpoint
@app.post("/api/v1/hackrx/run")
def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Extract document text
        text = extract_text_from_url(request.documents)

        # # Split the text if long
        # splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        # docs = splitter.split_text(text)
        # context = "\n\n".join(docs[:3])  # limit context size

        # For each question, query LLM
        answers = []
        for question in request.questions:
            prompt = PROMPT_TEMPLATE.format(context=text, query=question)
            answer = call_mistral(prompt)
            answers.append(answer.strip())

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
