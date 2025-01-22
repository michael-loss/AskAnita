import boto3
import botocore
import os
import json
import logging
import os

def format_chunks_for_prompt(chunks):
    combined_text = "\n\n".join(
        chunk['content']['text']
        for chunk in chunks
        if isinstance(chunk, dict) and 'content' in chunk and 'text' in chunk['content']
    )
    return combined_text

def deduplicate_chunks(chunks):
    # Check if chunks contain "retrievalResults" and extract it
    if isinstance(chunks, dict) and "retrievalResults" in chunks:
        chunks = chunks["retrievalResults"]

    seen = set()
    deduplicated = []

    for chunk in chunks:
        # Ensure the chunk is a dictionary and contains the 'content' and 'text' keys
        if isinstance(chunk, dict) and "content" in chunk and "text" in chunk["content"]:
            text = chunk["content"]["text"]
            if text not in seen:
                deduplicated.append(chunk)
                seen.add(text)
        else:
            print(f"Skipping invalid chunk: {chunk}")  # Debugging invalid chunks

    return deduplicated

def get_chat(fbedrock_agent_runtime_client,foundation_model, kb_id_hierarchical,query, region='us-east-1'):
    response = fbedrock_agent_runtime_client.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': kb_id_hierarchical,
                "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region, foundation_model),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults":5#,
                        #'overrideSearchType': 'SEMANTIC'
                    } 
                },
                'orchestrationConfiguration': {
                    'queryTransformationConfiguration': {
                        'type': 'QUERY_DECOMPOSITION'
                    }
                }
            }
        }
    )

    return response

def filter_chunks_by_score(chunks, threshold=0.7):
    return [chunk for chunk in chunks if chunk["score"] >= threshold]


def get_context(fbedrock_agent_runtime_client, foundation_model, kb_id_hierarchical, query, region='us-east-1'):
    """
    Retrieve relevant documents using bedrock_agent_runtime_client.retrieve(),
    then use those results as context in a prompt for the model to generate the response.
    """
    try:

        context = fbedrock_agent_runtime_client.retrieve(
            knowledgeBaseId=kb_id_hierarchical, 
            nextToken='string',
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults":15,
                    'overrideSearchType': 'HYBRID'
                } 
            },
            retrievalQuery={
                'text': query
            }
                )
        # Access the retrieval results from the context
        
        
        retrieval_results = context.get("retrievalResults", [])
        sorted_results = sorted(retrieval_results, key=lambda x: x.get("score", 0), reverse=True)
        
        #deduplicated_chunks = deduplicate_chunks(sorted_results)

        #retrieval_results = deduplicated_chunks.get("retrievalResults", [])
        #sorted_results = sorted(retrieval_results, key=lambda x: x.get("score", 0), reverse=True)
        #formatted_context = format_chunks_for_prompt(deduplicated_chunks)
        # Sort the retrieval results by the 'score' key in descending order
        #sorted_results = sorted(formatted_context, key=lambda x: x.get("score", 0), reverse=True)

    except Exception as e:
        return f"An error occurred: {str(e)}"
    return sorted_results

def get_response(fbedrock_client, foundation_model, query, region='us-east-1'):
    system = [{
    "text": "You are a helpful AI assistant."
    }]

    messages = [{
        "role": "user",
        "content": [{"text": query}]
    }]

    inference_config = {
        "maxTokens": 2000,
        "temperature": 0.0,
        "topP": 1.0,
        "topK": 50
    }

        # Construct the request body
    request_body = {
        "messages": messages,
        "system": system,
        "inferenceConfig": inference_config
    }

        # Call the model
    response = fbedrock_client.invoke_model(
            modelId=foundation_model,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )
        
        # Parse the response
    response_body = json.loads(response['body'].read())
        
        # Extract the generated text
    output_text = response_body['output']['message']['content'][0]['text']

    return output_text

import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import streamlit as st
import toml
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import time
#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from requests_aws4auth import AWS4Auth
#from chatter import *
from langdetect import detect

kb_id=""

class ChatHandler:
    def __init__(self):
        self.memory = ChatMessageHistory()

    def add_message(self, role, content):
        if role == "human":
            self.memory.add_user_message(content)
        elif role == "ai":
            self.memory.add_ai_message(content)

    def get_chat_history(self):
        return self.memory.messages

    def get_conversation_string(self):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in self.memory.messages])
    
    def save_message(self, user_input, ai_response):
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)

def get_awsauth(region, service):
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

def get_model_info(response, requested_model_id):
    response_body = json.loads(response.get('body').read())
    metadata = response.get('ResponseMetadata', {})
    headers = metadata.get('HTTPHeaders', {})
    
    # Try different ways to get model info
    model_info = {
        'requested_model': requested_model_id,
        'actual_model': None,
        'provider': None
    }
    
    # Try to get from response body (some models include this)
    if 'modelId' in response_body:
        model_info['actual_model'] = response_body['modelId']
    
    # Try to get from headers
    model_header = headers.get('x-amzn-bedrock-model-id')
    if model_header:
        model_info['actual_model'] = model_header
    
    # Extract provider from model ID
    if requested_model_id:
        provider = requested_model_id.split('.')[0]  # e.g., 'meta' from 'meta.llama3-70b-instruct-v1'
        model_info['provider'] = provider

    return model_info

def load_dotStreat_sl():
    """
    Load environment variables from either Streamlit Cloud secrets or local .streamlit/secrets.toml
    """
    try:
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None
        
        if is_streamlit_cloud:
            for key, value in st.secrets.items():
                if not key.startswith('_'):
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            full_key = f"{key}_{sub_key}".upper()
                            os.environ[full_key] = str(sub_value)
                    else:
                        os.environ[key.upper()] = str(value)
            return True
            
        else:
            secrets_path = Path('.streamlit/secrets.toml')
            
            if not secrets_path.exists():
                print(f"Warning: {secrets_path} not found")
                return False
                
            secrets = toml.load(secrets_path)
            
            for key, value in secrets.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        full_key = f"{key}_{sub_key}".upper()
                        os.environ[full_key] = str(sub_value)
                else:
                    os.environ[key.upper()] = str(value)
            
            return True
            
    except Exception as e:
        print(f"Error loading secrets: {str(e)}")
        return False

# Initialize AWS session and clients
load_dotStreat_sl()

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

bedrock = session.client('bedrock-runtime', 'us-east-1', 
                        endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')

#bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime') 
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime') 

# OpenSearch setup
opensearch = boto3.client("opensearchserverless")
host = os.getenv('opensearch_host')
region = 'us-east-1'
service = 'aoss'
credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)


client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

accept = 'application/json'
contentType = 'application/json'

def get_embedding(text):
    """
    Get embeddings using Amazon Titan embedding model
    """
    # Create the Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    
    # Prepare the request
    request_body = {
        "inputText": text
    }
    
    # Invoke the model
    #amazon.titan-embed-text-v1
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    # Process the response
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    
    return embedding

def answer_query_nova_kb(user_input, chat_handler):
    
    language_map = {
    "en": "English",
    "pl": "Polish",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    # Add more languages as needed
}
    model_id = st.session_state["model_id"]
    userQuery = user_input
    chat_history = chat_handler.get_conversation_string()  
    kb_id = st.session_state["kb_id"]
    mode = st.session_state["mode"] 
    
    detected_language_code = detect(userQuery)    

    detected_language_name = language_map.get(detected_language_code, "Unknown")
    context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery)

    prompt_data = f"""
        Assistant: You are an AI assistant designed to provide factual and accurate answers to user questions based on the Context provided.
        Language Consistency: The user's question is in {detected_language_name}. Respond in {detected_language_name}.
        

        Conversation History (for reference to clarify intent, NOT as a source for answers):
        {chat_history}

        Context (primary and authoritative source for answers):
        {context}

        Question:
        {userQuery}

        Instructions:
        1. Always use the Context as the primary and authoritative source for your answers.
        2. Use the Conversation History ONLY to:
        - Clarify the user's intent (e.g., identify which position statement or topic they are referring to).
        - Maintain continuity in the conversation.
        3. Do NOT generate answers based on the Conversation History alone. If the required information is not in the Context, respond with: "I don't know."
        4. Be concise and professional in your responses.
        5. Include specific details from the Context in your answer when applicable.
        6. If the user references a previous answer from the Conversation History, verify its accuracy against the Context before including it in your response.
        7. Please mention the sources by referring to specific ENA documents, policy briefs, and webpage URLs. 
        8. Sources URLs may be derived from information outside of the context. In the case of URLs create links directly to the sources on ENA's webite whenever possible.
        
        Example Behavior:
        - If the user asks, "Summarize it in 3 bullet points," ensure your summary comes exclusively from the Context provided.
        - If the user asks a follow-up question like, "Who is the author?" and it’s unclear which document they mean, use the Conversation History to infer the user's intent (e.g., "Are you referring to the document titled 'Access to Quality Healthcare'?").

        Answer:
        """

    output_text = get_response(bedrock, model_id, prompt_data, region='us-east-1')

    # Save interaction to chat memory
    chat_handler.add_message("human", userQuery)
    chat_handler.add_message("ai", output_text)
   
    output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\n"
    return output_text

import streamlit as st


# Define your main function
def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        # Simulate clear button click on first run
        #clear_all()
    #clear cache on start
        st.session_state.chat_handler = ChatHandler()
        st.rerun()
        st.cache_data.clear()
        st.cache_resource.clear()

    # Create a sidebar for the left panel
    with st.sidebar:

        st.image("AnitaMDorr.jpg", width=300, use_container_width=True)  # Increased width and using full column width
        
        # Add title below the image
        st.title("Hello! I'm ANITA - v2")

        # Add radio button group for "ENA Focus"
        # enafocus = st.radio(
        #     "ENA Focus",
        #     #("Position Statements", "Education"),
        #     ("Position Statements", "HR"),
        #     index=0,  # Default to "Position Statements"
        #     help="Select the ENA focus area"
        # )

        # First, store the radio button in a variable using session_state
        if 'previous_enafocus' not in st.session_state:
            st.session_state.previous_enafocus = None

        def on_enafocus_change():
            st.session_state.chat_handler = ChatHandler()
            st.cache_data.clear()
            st.cache_resource.clear()

        # Modify your radio button to include the callback
        enafocus = st.radio(
            "ENA Focus",
            #("Position Statements", "Education"),
            ("Position Statements", "HR"),
            index=0,  # Default to "Position Statements"
            help="Select the ENA focus area",
            key="enafocus",
            on_change=on_enafocus_change
        )

        # Add radio button group for "LLM Model"
        llm_model = st.radio(
            "LLM Model",
            ("Nova"),
            index=0,  # Default "
            help="Select the LLM model"
        )

        # Add clear chat button in the sidebar
        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()
            st.cache_data.clear()
            st.cache_resource.clear()

    # Set the prompt based on ENA Focus selection
    if enafocus == "Position Statements":
        chat_input_prompt = "Ask me anything about ENA's position statements!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_postions_id"]
        st.session_state["mode"] = "position statements"
    elif enafocus == "HR":
        chat_input_prompt = "Ask me anything about ENA's HR Documents!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_hr_id"]
        st.session_state["mode"] = "human resources documents"
        #mode

    # Set the response function based on LLM Model selection
    if llm_model == "Claude":
        #st.session_state["model_id"] = st.secrets["model_id_3"]
        response_function = answer_query_nova_kb
    elif llm_model == "Nova Lite":
        #st.session_state["model_id"] = st.secrets["model_id_2"]
        response_function = answer_query_nova_kb
    elif llm_model == "Nova":
        st.session_state["model_id"] = st.secrets["model_id_1"]
        #response_function = answer_query_nova
        st.cache_resource.clear()
        st.cache_data.clear()
        response_function = answer_query_nova_kb

    # Create a container for the header with subtitle (this will be the main content area)
    header_container = st.container()

    # Add custom CSS to style the button and layout
    st.markdown("""
        <style>
        .stButton>button {
            background-color: transparent;
            border: none;
            color: #4F8BF9;
            margin-top: 0px;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #f0f2f6;
            color: #4F8BF9;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize chat handler in session state if not already initialized
    if 'chat_handler' not in st.session_state:
        st.session_state.chat_handler = ChatHandler()

    # Display chat history
    for message in st.session_state.chat_handler.get_chat_history():
        with st.chat_message(message.type):
            st.write(message.content)

    # Get user input
    prompt = st.chat_input(chat_input_prompt)
    if prompt:
        # Display user message
        with st.chat_message("human"):
            st.write(prompt)

        # Get and display AI response based on the selected model
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = response_function(prompt, st.session_state.chat_handler)
                st.write(response)

# Call the main function to run the app
if __name__ == "__main__":
  main()
 
  
