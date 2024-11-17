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
CORS(app, origins=["https://sarthak-source.github.io","http://localhost:56279"])

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

main_url='https://onix-ix.firebaseapp.com'

collection_name = 'onix_data'
prompt_template = """
    Use the **context** provided to answer the **question** in a friendly, conversational tone. Keep the response lively and engaging. If the answer requires more detail, break it into clear, easy-to-read points with relevant **emojis**, limiting to a maximum of 5 points. For simpler questions, provide a brief and concise response—easy to read at a single glance.

    **Context**:
    {context}

    **Question**:
    {question}

    **Answer**:
    - Answer as if you are software that is talking about itself and what are is given in **context**, responds in the first person, speaking directly as the Onyx system.
    - Based on the **context**, answer the **question** in a natural, engaging way.
    - If detailed elaboration is needed, provide key insights, supporting ideas, and useful terms with appropriate **emojis** to highlight them, and limit to 5 points.
    - For simpler answers, keep it concise and to the point.
    - Ensure the answer is easy to digest at a single glance and feels conversational, as if you're chatting with a friend.
    - Aim for a response length between 50 and 100 words.
    - If the **question** isn't specific, engage in friendly, casual conversation to keep the interaction warm and approachable.
    - "end conversation" or "thank you" is in **question** something similar say "You’re welcome! If you need more assistance, feel free to reach out. Have a great day!" or something matching this sentence

    Try to generate the response dynamically by understanding the level of detail required and make sure it's readable at a glance. No reference needs to be made to the documents or any source material
"""



intent_prompt = """
Given a user question, determine the intent and, if it is related to opening a specific screen or performing an action, return the intent along with any necessary details such as the route or action type. Respond in a friendly tone and use appropriate emojis.
If the question does not relate to these commands, classify it as either "command" or "question" and do not return a route or action return intent.

**Available intents and routes/actions:**
- **General Prompt for Selection:**
  If the **command** doesn’t include "Previous" " Open" offer the user selection options, 
  Give a list of string of either "Previous **command**" " Open **command**" options next to option string put an emoji  in option field relevant to question ask, for example:- ['Access Settings ⚙️'], 
  Give a message field → intent: `"select_intent_command"`
  Default option is customer orders if nothing is mentioned after
  
- **Specific Commands:**
  - "get details for order [order number]" → intent: `"view_order_details"`, action: `"lookup_order"`, order_number: `[order number], 'option' is new field with *Print customer order* or *Open customer order* in list of strings only, use emoji for options visual appeal if Open send 'https://onix-ix.firebaseapp.com', if Print send print [order number] give ask do you need anything else `
  - "update order status to [status number]" → intent: `"update_order_status"`, action: `"update_status"`, status: `[status number]`
  - "show previous orders" or "display last orders" → intent: `"view_previous_orders"`, action: `"list_recent_orders"`, limit: `5`

**Special Instructions:**
- If a command to "update" or "open" lacks a specific number (e.g., order or status number), ask the user for this information in a friendly tone.
- When a number is provided, confirm the action is completed.
- After completing an action, provide a link to open the order page directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
- And next to it in bold ask should I print
- If none of the above mentioned in message please say I can't help with this, please ask something else

- **Customers Orders:**
  - "open all customer orders" → intent: `"open_screen_command"`, route: `"$tab/allCustomersOrders"`, and a message that provides a link to open the customer orders directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new customer order" → intent: `"open_screen_command"`, route: `"$tab/newCustomerOrder"`, and a message that provides a link to open the new customer order screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous customer order" → intent: `"open_screen_command"`, route: `"$tab/previousCustomerOrder"`, and a message that provides a link to open the previous customer order screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Sales Invoices:**
  - "open all previous sales invoices" → intent: `"open_screen_command"`, route: `"$tab/allPreviousSalesInvoices"`, and a message that provides a link to open the previous sales invoices screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new sales invoice" → intent: `"open_screen_command"`, route: `"$tab/newSaleInvoice"`, and a message that provides a link to open the new sales invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous sales invoice" → intent: `"open_screen_command"`, route: `"$tab/previousSaleInvoice"`, and a message that provides a link to open the previous sales invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Sales Return Invoice:**
  - "open all sales return invoices" → intent: `"open_screen_command"`, route: `"$tab/allSalesReturnInvoice"`, and a message that provides a link to open the all sales return invoices screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new sales return invoice" → intent: `"open_screen_command"`, route: `"$tab/newSaleReturnInvoice"`, and a message that provides a link to open the new sales return invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous sales return invoice" → intent: `"open_screen_command"`, route: `"$tab/previousSaleReturnInvoice"`, and a message that provides a link to open the previous sales return invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Quotation:**
  - "open all previous quotations" → intent: `"open_screen_command"`, route: `"$tab/allPreviousQuotations"`, and a message that provides a link to open the previous quotations screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new quotation" → intent: `"open_screen_command"`, route: `"$tab/newQuotation"`, and a message that provides a link to open the new quotation screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous quotation" → intent: `"open_screen_command"`, route: `"$tab/previousQuotation"`, and a message that provides a link to open the previous quotation screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Sales Return Request:**
  - "open all sales return requests" → intent: `"open_screen_command"`, route: `"$tab/allSalesReturnRequest"`, and a message that provides a link to open the all sales return requests screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new sales return request" → intent: `"open_screen_command"`, route: `"$tab/newSaleReturnRequest"`, and a message that provides a link to open the new sales return request screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous sales return request" → intent: `"open_screen_command"`, route: `"$tab/previousSaleReturnRequest"`, and a message that provides a link to open the previous sales return request screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Customers Notices:**
  - "open all customers notices" → intent: `"open_screen_command"`, route: `"$tab/allCustomersNotices"`, and a message that provides a link to open the customers notices screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new customer notice" → intent: `"open_screen_command"`, route: `"$tab/newCustomerNotice"`, and a message that provides a link to open the new customer notice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous customer notice" → intent: `"open_screen_command"`, route: `"$tab/previousCustomerNotice"`, and a message that provides a link to open the previous customer notice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Sales Invoice Issuance Permission:**
  - "open all sales invoice issuance permissions" → intent: `"open_screen_command"`, route: `"$tab/allSalesInvoiceIssuancePermission"`, and a message that provides a link to open the all sales invoice issuance permissions screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open new sales invoice issuance permission" → intent: `"open_screen_command"`, route: `"$tab/newSaleInvoiceIssuancePermission"`, and a message that provides a link to open the new sales invoice issuance permission screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous sales invoice issuance permission" → intent: `"open_screen_command"`, route: `"$tab/previousSaleInvoiceIssuancePermission"`, and a message that provides a link to open the previous sales invoice issuance permission screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Direct Sales Invoice:**
  - "open new direct selling invoice" → intent: `"open_screen_command"`, route: `"$tab/newDirectSellingInvoiceScreen"`, and a message that provides a link to open the new direct selling invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open all direct selling invoices" → intent: `"open_screen_command"`, route: `"$tab/allDirectSellingInvoiceScreen"`, and a message that provides a link to open the all direct selling invoices screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)
  - "open previous direct selling invoice" → intent: `"open_screen_command"`, route: `"$tab/previousDirectSellingInvoiceScreen"`, and a message that provides a link to open the previous direct selling invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Pay In Cash for Sales Return:**
  - "open new pay in cash for sales return" → intent: `"open_screen_command"`, route: `"$tab/newPayInCashForSalesReturnScreen"`, and a message that provides a link to open the pay in cash for sales return screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Stocktaking of Direct Sales Return:**
  - "open new stocktaking of direct sales return" → intent: `"open_screen_command"`, route: `"$tab/NewStocktakingOfDirectSalesReturn"`, and a message that provides a link to open the stocktaking of direct sales return screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Direct Selling Return Invoice Screen:**
  - "open direct selling return invoice screen" → intent: `"open_screen_command"`, route: `"$tab/directSellingReturnInvoiceScreen"`, and a message that provides a link to open the direct selling return invoice screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Clearing Cashiers Sales:**
  - "open clearing cashiers sales" → intent: `"open_screen_command"`, route: `"$tab/clearingCashiersSales"`, and a message that provides a link to open the clearing cashiers sales screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

- **Post to Financial System:**
  - "open new post to financial system" → intent: `"open_screen_command"`, route: `"$tab/newPostToFinancialSystem"`, and a message that provides a link to open the post to financial system screen directly. (Link prefix: `'https://onix-ix.firebaseapp.com'`)

**Question:** {question}

**Response Format:**
Return a JSON object containing the intent, and, if applicable, the route or action in the following structure.

"""


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
        
def get_order_details(order_number):
    # Simulate fetching order details based on order number
    if order_number != 'unknown':
        # You can replace this with actual logic to fetch order details from a database or API
        return f"Here are the details for Order {order_number}: \n•Customer Name: Acme Corporation \n•Order Status: Processing \n•Order Date: November 10, 2024 \n•Items: 100 units of Product X, \n•50 units of Product Y \n•Total Amount: $3,000 \n•Estimated Delivery: November 20, 2024 \n•Assigned Representative: Sarah L."
    else:
        return ""  # Return empty string if no order details are found or order_number is unknown
def update_order_status(status):
    if status:
        # Simulate updating the order status
        # Replace this with actual logic for updating order status in a system
        return f"Order status has been updated to: {status}"
    else:
        return "No status provided, order status not updated."
    
def view_previous_orders():
    # Placeholder data for the last five orders
    orders = [
        {"order_number": "12345", "date": "2024-11-12", "status": "Delivered", "total": "$150.00"},
        {"order_number": "12344", "date": "2024-11-10", "status": "Processing", "total": "$320.50"},
        {"order_number": "12343", "date": "2024-11-08", "status": "Shipped", "total": "$225.99"},
        {"order_number": "12342", "date": "2024-11-05", "status": "Cancelled", "total": "$90.00"},
        {"order_number": "12341", "date": "2024-11-02", "status": "Delivered", "total": "$180.75"},
    ]

    # Display the list of orders
    orders_display = "Here are your last 5 orders:\n"
    for order in orders:
        orders_display += (
            f"•Order Number: #{order['order_number']}, "
            f"•Date: {order['date']}, "
            f"•Status: {order['status']}, "
            f"•Total: {order['total']}\n\n"
        )
        
    return f"{orders_display}"

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
        # Retrieve previous context from the session
        previous_context = "\n".join(session['conversation_history'])

        # Add the new question to the history
        session['conversation_history'].append(f"Q: {question}")

        # Limit conversation history length
        MAX_HISTORY_LENGTH = 10
        if len(session['conversation_history']) > MAX_HISTORY_LENGTH:
            session['conversation_history'].pop(0)

        # Combine previous context and the new question for generating the response
        combined_context = previous_context + "\n" + f"Q: {question}\n"

        # Get the intent from the question
        intent_log = process_command(question)
        print("Result intent_log:", intent_log)

        # Handle different intents
        if intent_log['intent'] == 'open_screen_command':
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                #'answer': intent_log['message']
            })

        elif intent_log['intent'] == 'view_order_details':
            order_number = intent_log.get('order_number', 'unknown')
            order_details = get_order_details(order_number)
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                'answer': f"{order_details}\n\n{intent_log['message']}"
            })

        elif intent_log['intent'] == 'update_order_status':
            status = intent_log.get('order_number', 'not specified')
            update_status_result = update_order_status(status)
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                'answer': f"{update_status_result}\n\n{intent_log['message']}"
            })
            
        elif intent_log['intent'] == 'view_previous_orders':
            view_previous_order=view_previous_orders()
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                'answer': f"Here are last 5 order {view_previous_order}"
            })

        elif intent_log['intent'] == 'end_conversation':
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                'answer': "Thank you for chatting! If you have more questions, feel free to ask."
            })
            
        elif intent_log['intent'] == 'select_intent_command':
            return jsonify({
                'session_id': session['session_id'],
                'question': question,
                'intent': intent_log,
                'answer': intent_log['message'],
            })

        else:
            # Default handling for questions that don’t match any specific intent
            intent_log['intent'] = 'question'  # Explicitly set the intent to 'question'
            answer_response = user_input(question)  # Generate a general answer
            session['conversation_history'].append(f"A: {answer_response['output_text']}")

            # Limit conversation history length again
            if len(session['conversation_history']) > MAX_HISTORY_LENGTH:
                session['conversation_history'].pop(0)

            response = {
                'session_id': session['session_id'],
                'question': question,
                'answer': answer_response,
                'conversation_history': session['conversation_history'],
                'combined_context': combined_context,
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
