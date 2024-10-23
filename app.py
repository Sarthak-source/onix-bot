import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import firebase_admin
from firebase_admin import credentials, firestore
import re
from flask import Flask, request, jsonify
import os
import logging
#import json

# Initialize Flask app
app = Flask(__name__)
print('bot-says-hello-world')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

# Load the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

type_ = os.getenv("TYPE")
project_id = os.getenv("PROJECT_ID")
private_key_id = os.getenv("PRIVATE_KEY_ID")
private_key = os.getenv("PRIVATE_KEY")
client_email = os.getenv("CLIENT_EMAIL")
client_id = os.getenv("CLIENT_ID")
auth_uri = os.getenv("AUTH_URI")
token_uri = os.getenv("TOKEN_URI")
auth_provider_x509_cert_url = os.getenv("AUTH_PROVIDER_X509_CERT_URL")
client_x509_cert_url = os.getenv("CLIENT_X509_CERT_URL")
universe_domain = os.getenv("UNIVERSE_DOMAIN")

# Initialize Firebase Admin SDK (check if already initialized)
if not firebase_admin._apps:
    cred_initialize = credentials.Certificate({
        "type": type_,
        "project_id": project_id,
        "private_key_id": private_key_id,
        "private_key": private_key.replace("\\n", "\n"),  # Replace escaped newlines with real newlines
        "client_email": client_email,
        "client_id": client_id,
        "auth_uri": auth_uri,
        "token_uri": token_uri,
        "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
        "client_x509_cert_url": client_x509_cert_url
    })
    firebase_admin.initialize_app(cred_initialize)

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

# Set up logging
#logging.basicConfig(level=logging.INFO)

@app.route('/ask', methods=['POST'])
def ask_question():
    # Parse the incoming request
    request_data = request.json
    question = request_data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Read the most recent data from Firestore
        recent_data = read_recent_uploaded_data()

        # Check if recent_data is None
        if not recent_data:
            return jsonify({'error': 'No recent data found'}), 404

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

    except Exception as e:
        logging.error('Error processing question: %s', str(e))
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0')
