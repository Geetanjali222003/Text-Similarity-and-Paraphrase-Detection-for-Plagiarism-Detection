from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import google.generativeai as genai
import os

# -----------------------
# Load TF-IDF model files
# -----------------------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("doc_names.pkl", "rb") as f:
    doc_names = pickle.load(f)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

class TextInput(BaseModel):
    text: str

# -----------------------
# Plagiarism Check Route
# -----------------------
@app.post("/plagiarism-check")
def plagiarism_check(data: TextInput):
    query_vec = vectorizer.transform([data.text])
    cosine_sim = np.dot(query_vec, tfidf_matrix.T).toarray()[0]

    best_idx = int(np.argmax(cosine_sim))
    best_score = float(cosine_sim[best_idx])

    return {
        "most_similar_doc": doc_names[best_idx],
        "similarity_score": round(best_score * 100, 2)
    }

# -----------------------
# AI Detection Route
# -----------------------
@app.post("/ai-detection")
def ai_detection(data: TextInput):
    try:
        # Configure Gemini API
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel("gemini-pro")

        prompt = f"Detect if this text was written by AI or a human. Return ONLY a percentage (0-100) of AI-likeness.\n\nText:\n{data.text}"
        response = model.generate_content(prompt)

        return {
            "ai_likeness": response.text.strip()
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Root
# -----------------------
@app.get("/")
def root():
    return {"message": "Plagiarism & AI Detection API running!"}
