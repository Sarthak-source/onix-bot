import os
import logging
import gc
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)
print('bot-says-hello-world')

# Initialize Firebase Admin SDK
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

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(firebase_credentials))

# Create a Firestore client
db = firestore.client()

# Configure the Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# Function to answer questions using Google Gemini
def answer_question(question, text):
    for chunk in chunk_text(text):
        # Call the Gemini model
        response = genai.generate_text(
            model='gemini-pro',  # Update with the correct model name
            prompt=f"Q: {question}\nA:",
            context=chunk
        )
        answer = response.result.get('text')
        if answer:  # Return the answer if found
            return answer, chunk
    return "No answer found", ""

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    collection_name = 'onix_data'
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        data = doc.to_dict()
        logging.info(f"Recent data length: {len(data)}")
        return data
    return None  

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
        recent_data = read_recent_uploaded_data()
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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0')
