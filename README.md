# Cognitive Retrieval — AI-Powered Document Intelligence

> Upload any PDF or image and ask questions in natural language. Powered by a multi-agent RAG pipeline with vision OCR, knowledge graphs, and LLM fallback.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ✨ Features

- 📄 **Multi-document support** — upload multiple PDFs, switch between them with a click
- 🔍 **Vision OCR** — YOLO + EasyOCR extracts text from scanned PDFs and images
- 🧠 **Multi-agent RAG** — dense retrieval, knowledge graph, hierarchical PageIndex, hybrid fusion
- 🤖 **LLM fallback chain** — Gemini 2.5 Flash → Together AI (Llama 3.3 70B) → structured context
- 📊 **Knowledge Graph** — entity/relation extraction for multi-hop reasoning
- 💬 **Auto-summary** — instant document summary on upload
- 🔒 **Document isolation** — each PDF gets its own Qdrant filter scope

---

## 🏗️ Architecture

```
frontend (Next.js 14)
    ↓ NEXT_PUBLIC_API_URL
backend (FastAPI)
    ├── Vision Service (YOLO + EasyOCR + pdfplumber)
    ├── RAG Service (sentence-transformers + Qdrant)
    ├── PageIndex Service (Gemini / Together AI)
    ├── Knowledge Graph Service (entity extraction)
    ├── Retrieval Orchestrator (dense + graph + hybrid)
    └── LLM Answer Generation (Gemini → Together AI)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- Node.js 20+
- API keys: [Gemini](https://aistudio.google.com/), [Together AI](https://together.ai/), [HuggingFace](https://huggingface.co/settings/tokens)

### 1. Clone & configure
```bash
git clone https://github.com/Priyesh29sri/OCR---Cognitive-Retrieval-.git
cd OCR---Cognitive-Retrieval-

cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with the start script
```bash
# Creates venv, installs deps, clears locks, starts both servers
bash start.sh
```

Open **http://localhost:3000**

### 3. Or run with Docker Compose
```bash
docker compose up --build
```

---

## ☁️ Deploy on Render (One-click)

1. Fork this repo
2. Go to [render.com](https://render.com) → **New → Blueprint**
3. Connect your forked repo
4. Render reads `render.yaml` and creates both services automatically
5. Set the environment variables in the Render dashboard:
   - `GEMINI_API_KEY`
   - `TOGETHER_API_KEY`
   - `HUGGING_FACE_HUB_TOKEN`
6. After the backend deploys, copy its URL (e.g. `https://cognitive-retrieval-backend.onrender.com`) and update the `NEXT_PUBLIC_API_URL` build arg in the frontend service

> **Note:** Free tier Render services spin down after 15 minutes of inactivity and take ~30s to cold-start.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `TOGETHER_API_KEY` | Yes | Together AI key (Llama 3.3 70B fallback) |
| `HUGGING_FACE_HUB_TOKEN` | Yes | HuggingFace token (sentence-transformers) |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend URL (default: `http://127.0.0.1:8000`) |

---

## 📁 Project Structure

```
.
├── app/                        # FastAPI backend
│   ├── main.py                 # API endpoints
│   ├── models/                 # Pydantic schemas
│   ├── repositories/           # Qdrant vector store
│   └── services/               # All AI services
├── frontend/                   # Next.js frontend
│   ├── app/page.tsx            # Main UI
│   ├── components/ui/          # Shadcn + custom components
│   └── lib/api.ts              # Backend API client
├── scripts/                    # Benchmarking & evaluation
├── Dockerfile                  # Backend Docker image
├── frontend/Dockerfile         # Frontend Docker image
├── docker-compose.yml          # Local Docker setup
├── render.yaml                 # Render deployment blueprint
├── requirements.txt            # Python dependencies
└── start.sh                    # Local start script
```

---

## 🧪 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload PDF or image |
| POST | `/query` | Ask a question |

### Upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "document_id": "1"}'
```

---

## 📜 License

MIT © 2026 [Priyesh Srivastava](https://github.com/Priyesh29sri)
