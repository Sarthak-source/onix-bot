import numpy as np
import sys
import os
import logging
import gc
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
from pinferencia import Server
from pinferencia.task import task  # Ensure the correct import

# Initialize Flask app
app = Flask(__name__)
print('bot-says-hello-world')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the Question Answering Model class for Pinferencia
class QuestionAnsweringModel:
    def __init__(self):
        # Load the question-answering pipeline using Hugging Face's transformers
        self.qa_pipeline = pipeline("question-answering", model='distilbert-base-uncased')

    @task
    def predict(self, data):
        # Extract the question and context from the input data
        question = data['question']
        context = data['context']
        # Use the QA pipeline to generate an answer
        return self.qa_pipeline(question=question, context=context)

# Define the Sentence Embedding Model class for Pinferencia
class SentenceEmbeddingModel:
    def __init__(self):
        # Load the Sentence Transformer model for embedding
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    @task
    def predict(self, data):
        # Encode the input text and return the sentence embeddings
        return self.sentence_model.encode(data['text'], convert_to_tensor=True)

# Initialize the Pinferencia server
server = Server()

# Register the models to different routes on the server
server.register(QuestionAnsweringModel, "/qa")
server.register(SentenceEmbeddingModel, "/sentence")

# Firebase credentials
firebase_credentials = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL")
}

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(firebase_credentials))

# Create a Firestore client
db = firestore.client()

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    """Yields smaller chunks of text."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# Function to find the most relevant chunk for the question using Sentence Transformers
def find_relevant_chunk(question, chunk):
    """Finds the most relevant chunk of text for a given question using Sentence Transformers."""
    # Get embeddings for the question and the chunk of text
    question_embedding = server.model("/sentence").predict({"text": question})
    chunk_embedding = server.model("/sentence").predict({"text": chunk})

    # Compute cosine similarity between the question and chunk
    cosine_similarity = util.pytorch_cos_sim(question_embedding, chunk_embedding)
    return cosine_similarity.item()  # Return the similarity score

# Function to answer questions based on the relevant context
def answer_question(question, text):
    """Answers the question based on the most relevant chunk of context."""
    for chunk in chunk_text(text):
        # Check if the chunk is relevant based on similarity
        similarity = find_relevant_chunk(question, chunk)
        if similarity > 0.5:  # Use a threshold to determine relevance
            # Get the answer from the QA model
            result = server.model("/qa").predict({"question": question, "context": chunk})
            if result['answer']:  # Return the answer if found
                return result['answer'], chunk
    return "No answer found", ""

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    """Reads the most recent document from Firestore."""
    collection_name = 'onix_data'  # Replace with your Firestore collection name
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        data = doc.to_dict()  # Get the data of the most recent document
        logging.info(f"Recent data length: {len(data)}")  # Log the length of recent data
        return data
    return None  # Return None if no documents are found

# API route to handle questions and return answers
@app.route('/ask', methods=['POST'])
def ask_question_api():
    """Handles incoming POST requests to answer questions."""
    # Parse the incoming request JSON data
    request_data = request.json
    logging.info('Received request: %s', request_data)

    question = request_data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Read the most recent data from Firestore
        recent_data = read_recent_uploaded_data()
        if not recent_data:
            return jsonify({'error': 'No recent data found'}), 404

        # Get the combined text from the Firestore document
        combined_text = recent_data.get('combined_text', '')

        # Get the answer and the relevant chunk
        answer, relevant_chunk = answer_question(question, combined_text)

        # Return the result as a JSON response
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
        # Run garbage collection to free memory
        gc.collect()

# Run the Pinferencia server and Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0')
