# Clinical Trial Eligibility (CTE) Agent

## How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Setup and Run

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

4. **Access the UI:**
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Testing

Run the smoke tests to verify the server is working:
```bash
pytest tests/test_smoke.py -v
```

---

## Architecture

*To be completed in Phase 0 summary.*

## Design Decisions

*To be completed as phases are implemented.*

## What's Next with More Time

*To be completed in final summary.*
