import os
import logging
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print('bot-says-hello-world')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

collection_name = 'onix_data'
prompt_template = """
    Answer the question as thoroughly as possible using the information provided in the context. If the exact answer is not available, do not guess. Instead, provide a list of related terms or concepts from the context that could be useful. If there are no relevant matches at all, explicitly state, "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer (or related terms if answer is not available):
    """

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    logger.info("Fetching the most recent uploaded data from Firestore.")
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        data = doc.to_dict()
        logger.info(f"Recent data fetched: {data}")
        return data
    logger.warning("No recent data found in Firestore.")
    return None

def get_text_chunks(text):
    logger.info(f"Splitting text into chunks. Original text length: {len(text)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    logger.info(f"Number of chunks created: {len(chunks)}")
    return chunks

def get_vector_store(text_chunks):
    logger.info(f"Creating vector store from {len(text_chunks)} text chunks.")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logger.info("Vector store saved successfully.")

# Load initial data and create vector store
raw_text = read_recent_uploaded_data()
if raw_text and 'combined_text' in raw_text:
    text_chunks = get_text_chunks(raw_text['combined_text'])
    get_vector_store(text_chunks)
else:
    logger.error("Combined text is missing or empty.")
    raise ValueError("Combined text is missing or empty.")

def get_conversational_chain():
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    logger.info("Conversational chain created successfully.")
    return chain

chain = get_conversational_chain()

def user_input(user_question):
    logger.info(f"User question received: {user_question}")
    docs = new_db.similarity_search(user_question)
    logger.info(f"Documents found: {len(docs)}")

    if not docs:
        logger.warning("No relevant documents found for the question.")
        return {"output_text": "Answer is not available in the context."}

    # Load FAISS index with the allow_dangerous_deserialization parameter set to True
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    logger.info(f"Response generated: {response}")
    return response

# API route to handle questions and return answers
@app.route('/ask', methods=['POST'])
def ask_question_api():
    request_data = request.json
    logger.info('Received request: %s', request_data)

    question = request_data.get('question')
    if not question:
        logger.error('No question provided in the request.')
        return jsonify({'error': 'No question provided'}), 400

    try:
        recent_data = read_recent_uploaded_data()
        if not recent_data:
            return jsonify({'error': 'No recent data found'}), 404

        answer_response = user_input(question)
        response = {
            'question': question,
            'answer': answer_response  # Adjust based on the return value of user_input
        }
        logger.info(f"Response sent: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error('Error processing question: %s', str(e), exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0')