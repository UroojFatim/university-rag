backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

frontend
pip install -r frontend/requirements.txt
streamlit run frontend/app.py

