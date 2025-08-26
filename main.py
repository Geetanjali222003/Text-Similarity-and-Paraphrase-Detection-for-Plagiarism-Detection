from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
from docx import Document
import re

app = FastAPI()

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from different file types"""
    try:
        if filename.lower().endswith('.txt'):
            return file_content.decode('utf-8', errors='ignore')
        
        elif filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif filename.lower().endswith(('.docx', '.doc')):
            doc = Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            # Try to decode as text
            return file_content.decode('utf-8', errors='ignore')
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file {filename}: {str(e)}")

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    # Clean and preprocess text
    text1 = re.sub(r'\s+', ' ', text1.strip())
    text2 = re.sub(r'\s+', ' ', text2.strip())
    
    # Use TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI deployed to Render!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Service is running"}

@app.post("/compare-files")
async def compare_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Read file contents
        content1 = await file1.read()
        content2 = await file2.read()
        
        # Extract text from files
        text1 = extract_text_from_file(content1, file1.filename or "file1")
        text2 = extract_text_from_file(content2, file2.filename or "file2")
        
        # Calculate similarity
        similarity_score = calculate_similarity(text1, text2)
        plagiarism_percentage = similarity_score * 100
        
        return {
            "similarity_score": similarity_score,
            "plagiarism_percentage": plagiarism_percentage,
            "algorithm_used": "cosine_tfidf",
            "is_plagiarized": plagiarism_percentage > 50,
            "matching_segments": [],
            "status": "success",
            "message": "Comparison completed successfully"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error", "message": "Comparison failed"}
        )

@app.post("/compare")
async def compare_texts(request: dict):
    try:
        text1 = request.get("text1", "")
        text2 = request.get("text2", "")
        
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 are required")
        
        similarity_score = calculate_similarity(text1, text2)
        plagiarism_percentage = similarity_score * 100
        
        return {
            "similarity_score": similarity_score,
            "plagiarism_percentage": plagiarism_percentage,
            "algorithm_used": "cosine_tfidf",
            "is_plagiarized": plagiarism_percentage > 50,
            "matching_segments": [],
            "status": "success",
            "message": "Comparison completed successfully"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error", "message": "Text comparison failed"}
        )