import os
import logging
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
#from dotenv import load_dotenv

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

# Configure the Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
# Initialize Google Gemini model
#model = genai.GenerativeModel('gemini-1.5-flash')

# Function to chunk text into semantically relevant pieces
# def chunk_text(text):
#     return [text.strip()]

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with the allow_dangerous_deserialization parameter set to True
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    return response


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

        #combined_text = recent_data.get('combined_text', '')

        # Call user_input to get the answer
        user_input(question)

        # You may need to modify user_input to return a meaningful response
        # If user_input directly prints, you might want to adjust that logic.
        # Assuming user_input is modified to return the answer instead of printing it.
        answer_response = user_input(question)

        response = {
            'question': question,
            'answer': answer_response  # Adjust based on the return value of user_input
        }
        return jsonify(response)

    except Exception as e:
        logging.error('Error processing question: %s', str(e), exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0')
