import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import firebase_admin
from firebase_admin import credentials, firestore
import re
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)
print('bot-says-hello-world')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

# Load the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Firebase Admin SDK (check if already initialized)
if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccountKey.json') 
    # Replace with the correct path to your service account key
    firebase_admin.initialize_app(cred)

# Create a Firestore client
db = firestore.client()

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    """Splits the text into smaller chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to find the most relevant chunk for the question using Sentence Transformers
def find_relevant_chunk(question, chunks):
    """Finds the most relevant chunk of text for a given question using Sentence Transformers."""
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    chunk_embeddings = sentence_model.encode(chunks, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    best_chunk_index = np.argmax(cosine_similarities)  # Get index of the most similar chunk
    return chunks[best_chunk_index]  # Return the most relevant chunk

# Function to answer questions based on the relevant context
def answer_question(question, chunks):
    """Answers the question based on the most relevant chunk."""
    relevant_chunk = find_relevant_chunk(question, chunks)  # Get the relevant chunk
    result = qa_pipeline(question=question, context=relevant_chunk)
    return result['answer'], relevant_chunk

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    collection_name = 'onix_data'  # Replace with your collection name
    # Query to get the most recent document based on the timestamp
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        return doc.to_dict()  # Return the data of the most recent document

# API route to handle questions and return answers
@app.route('/ask', methods=['POST'])
def ask_question():
    # Parse the incoming request
    request_data = request.json
    question = request_data.get('question')

    # Read the most recent data from Firestore
    recent_data = read_recent_uploaded_data()

    # Get the combined text from the Firestore document
    combined_text = recent_data.get('combined_text', '')

    # Chunk the combined text
    chunks = chunk_text(combined_text)

    # Get the answer and the relevant chunk
    answer, relevant_chunk = answer_question(question, chunks)

    # Return the result as a JSON response
    response = {
        'question': question,
        'answer': answer,
        'relevant_chunk': relevant_chunk
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0')
