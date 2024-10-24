import os
import logging
import gc
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print('bot-says-hello-world')

# Initialize Firebase Admin SDK
firebase_credentials = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL")
}

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(firebase_credentials))

# Create a Firestore client
db = firestore.client()

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    collection_name = 'onix_data'
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        data = doc.to_dict()
        logging.info(f"Recent data length: {len(data)}")
        return data
    return None

# Initialize Google Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')
recent_data = read_recent_uploaded_data()

# Configure the Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model, you can choose another one

# Function to chunk text into semantically relevant pieces
def chunk_text(text, chunk_size=500, similarity_threshold=0.8):
    sentences = text.split('.')  # Split text into sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + '.'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '.'

            # If we reach the chunk_size, save the chunk
            if len(current_chunk) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""

    # Don't forget to add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Compute embeddings for each chunk
    embeddings = embedding_model.encode(chunks)

    # Filter chunks based on cosine similarity
    filtered_chunks = []
    for i in range(len(embeddings)):
        if len(filtered_chunks) == 0:
            filtered_chunks.append(chunks[i])
        else:
            similarity = cosine_similarity([embeddings[i]], [embeddings[len(filtered_chunks)-1]])
            if similarity < similarity_threshold:
                filtered_chunks.append(chunks[i])

    return filtered_chunks

# Function to answer questions using Google Gemini
def answer_question(question, text):
    chunks = chunk_text(text)  # Use the advanced chunk_text function

    for chunk in chunks:
        try:
            # Call the Gemini model
            response = model.generate_content([question, chunk])
            answer = response.text
            if answer:
                return answer, chunk
        except Exception as e:
            logging.error('Error during text generation: %s', str(e))
    return "No answer found", ""

# Set up logging
logging.basicConfig(level=logging.INFO)

# API route to handle questions and return answers
@app.route('/ask', methods=['POST'])
def ask_question_api():
    request_data = request.json
    logging.info('Received request: %s', request_data)

    question = request_data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        if not recent_data:
            return jsonify({'error': 'No recent data found'}), 404

        combined_text = recent_data.get('combined_text', '')
        answer, relevant_chunk = answer_question(question, combined_text)

        response = {
            'question': question,
            'answer': answer,
            'relevant_chunk': relevant_chunk
        }
        return jsonify(response)

    except Exception as e:
        logging.error('Error processing question: %s', str(e), exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

    finally:
        gc.collect()

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0')
