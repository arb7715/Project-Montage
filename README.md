# Project Montage — AI Animated Short Film Pipeline

Course project (CS-4015 Agentic AI): LangGraph orchestration from screenplay to lip-synced scene MP4s.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements_phase2.txt
```

Paste your Colab tunnel URL into `config/colab_api.txt` (see `config/colab_api.example.txt`). Run the Colab cell stack from `colab_sd_api.txt`.

## Run

```powershell
python -m src.smoke_test_colab
python -m src.main
python -m src.main_phase2
streamlit run src/ui/app.py
```

See `PROJECT_CONTEXT.md` for full architecture, MCP constraints, and handover notes.
