import os
import logging
import uuid  # Import uuid for generating session IDs
from flask import Flask, request, jsonify, session  # Add session to import
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
#from pydantic import BaseModel, Field
import json
import re


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management
CORS(app, origins=["https://sarthak-source.github.io","http://localhost:50374"])

print('bot-says-hello-world')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
#print(os.getenv("PRIVATE_KEY"))
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

#print(firebase_credentials)

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(firebase_credentials))

# Create a Firestore client
db = firestore.client()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
generation_config = {
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", generation_config=generation_config)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

collection_name = 'onix_data'
prompt_template = """
    Use the **context** provided to answer the **question** in a friendly, conversational tone. Keep the response lively and engaging. If the answer requires more detail, break it into clear, easy-to-read points with relevant **emojis**, limiting to a maximum of 5 points. For simpler questions, provide a brief and concise response—easy to read at a single glance.

    **Context**:
    {context}

    **Question**:
    {question}

    **Answer**:
    - Based on the **context**, answer the **question** in a natural, engaging way.
    - If detailed elaboration is needed, provide key insights, supporting ideas, and useful terms with appropriate **emojis** to highlight them, and limit to 5 points.
    - For simpler answers, keep it concise and to the point.
    - Ensure the answer is easy to digest at a single glance and feels conversational, as if you're chatting with a friend.
    - Aim for a response length between 50 and 100 words.
    - If the **question** isn't specific, engage in friendly, casual conversation to keep the interaction warm and approachable.

    Try to generate the response dynamically by understanding the level of detail required and make sure it's readable at a glance. No reference needs to be made to the documents or any source material
"""

intent_prompt = """
Given a user question, determine the intent and, if it is related to opening a specific screen, return both "open_screen_command" as the intent and the corresponding route.

Available routes:
- "open logs screen" or "show logs" → route: "/logs"
- "show customer orders" or "open orders screen" → route: "/orders"
- "view dashboard" or "open dashboard" → route: "/dashboard"
- "open settings" or "show settings screen" → route: "/settings"

If the question does not relate to a screen command, classify it as either "command" or "question" and do not return a route.

Question: {question}
Respond with a JSON object containing the intent and, if applicable, the route, structured as follows.
"""

import json
import re

def process_command(question):
    # Create the prompt template
    prompt_for_command = PromptTemplate(template=intent_prompt, input_variables=["question"])

    # Create the LLMChain for processing the prompt
    runnable = prompt_for_command | model

    # Pass the question
    result = runnable.invoke({"question": question})

    # Retrieve and clean the content of the result
    result_content = result.content.strip()

    print("=======================================")
    print("Result Content:", result_content, result)
    print("=======================================")

    # Clean up the response by removing ```json and ``` markers if they exist
    cleaned_content = re.sub(r"^```json|```$", "", result_content.strip(), flags=re.MULTILINE).strip()

    try:
        # First, try parsing the outer layer of JSON content (if any)
        result_dict = json.loads(cleaned_content)
        
        # Check if the "intent" field contains a string that looks like JSON
        if isinstance(result_dict, dict) and 'intent' in result_dict:
            intent_content = result_dict['intent']
            
            # If the intent is a string that looks like JSON, we need to parse it
            if isinstance(intent_content, str):
                try:
                    # Attempt to parse the inner JSON string
                    intent_dict = json.loads(intent_content)
                    result_dict['intent'] = intent_dict  # Replace the raw string with the parsed JSON
                except json.JSONDecodeError:
                    pass  # If the inner content is not JSON, leave it as-is

        return result_dict  # Return the cleaned and parsed dictionary
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse result content as JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Function to read the most recent uploaded data from Firestore
def read_recent_uploaded_data():
    logger.info("Fetching the most recent uploaded data from Firestore.")
    recent_doc = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in recent_doc:
        data = doc.to_dict()
        #logger.info(f"Recent data fetched: {data}")
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

# Function to initialize session
def initialize_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())  # Generate a random session ID
        session['conversation_history'] = []  # Initialize conversation history
        logger.info(f"Session initialized with ID: {session['session_id']}")

# API route to handle questions and return answers
@app.route('/ask', methods=['POST'])
def ask_question_api():
    initialize_session()  # Ensure the session is initialized
    request_data = request.json
    logger.info('Received request: %s', request_data)

    question = request_data.get('question')
    if not question:
        logger.error('No question provided in the request.')
        return jsonify({'error': 'No question provided'}), 400

    try:
        if not raw_text:
            return jsonify({'error': 'No recent data found'}), 404

        # Retrieve the previous context from the session
        previous_context = "\n".join(session['conversation_history'])  # Join previous questions and answers
        
        # Add the new question to the history
        session['conversation_history'].append(f"Q: {question}")  # Store the user's question

        # Check and limit conversation history length
        MAX_HISTORY_LENGTH = 10  # Set your desired maximum length
        if len(session['conversation_history']) > MAX_HISTORY_LENGTH:
            session['conversation_history'].pop(0)  # Remove the oldest entry if limit is exceeded

        # Combine previous context and the new question for generating the response
        combined_context = previous_context + "\n" + f"Q: {question}\n"  # Prepare context for model

        # Get the intent from the question
        intent_log = process_command(question)
        print("Result intent_log:", intent_log)

        # Check if the intent is 'open_screen_command' and handle accordingly
        if intent_log['intent'] == 'open_screen_command':
            # Skip further processing if intent is 'open_screen_command'
            return jsonify({
                'session_id': session['session_id'],  # Include the session ID in the response
                'question': question,
                'intent': intent_log,
                'answer': 'Screen open command processed successfully'
            })

        # If the intent is not 'open_screen_command', continue with normal processing
        answer_response = user_input(question)

        # Store the answer in the conversation history
        session['conversation_history'].append(f"A: {answer_response['output_text']}")  # Store the generated answer

        # Check and limit conversation history length again
        if len(session['conversation_history']) > MAX_HISTORY_LENGTH:
            session['conversation_history'].pop(0)  # Remove the oldest entry if limit is exceeded again

        # Include combined context when responding
        response = {
            'session_id': session['session_id'],  # Include the session ID in the response
            'question': question,
            'answer': answer_response,
            'conversation_history': session['conversation_history'],  # Include the conversation history
            'combined_context': combined_context,  # Include the combined context for debugging
            'intent': intent_log
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
