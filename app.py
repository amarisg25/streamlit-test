import logging 

# Setup Logging
logging.basicConfig(
    level=logging.INFO,  # Set global logging level to INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logging.getLogger("watchdog").setLevel(logging.WARNING)  # Suppress watchdog DEBUG logs
logging.getLogger("chromadb").setLevel(logging.WARNING)  # Optionally suppress chromadb logs


import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import autogen
import chromadb
 # Import logging
import json


# CONFIGURATION
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
env_api_key = os.getenv('OPENAI_API_KEY')

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Streamlit Sidebar for Configuration
with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox(
        "Model", 
        ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini'], 
        index=1,
        key="model_select"  # Optional: Add a unique key
    )
    user_api_key = st.text_input(
        "API Key", 
        type="password", 
        key="api_key_input"  # Optional: Add a unique key
    )

# Determine which API key to use
api_key = user_api_key 

if not api_key:
    st.warning('Please provide a valid OpenAI API key in the sidebar.', icon="⚠️")
    st.stop()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function description for LLM
llm_config = {
    "temperature": 0,
    "timeout": 300,
    "cache_seed": 43,
    "config_list": [{
        "model": selected_model,
        "api_key": api_key  # Use the API key entered by the user
    }]
}

# FUNCTION TO CHECK TERMINATION
def check_termination(x):
    """
    Checks if the message content ends with "TERMINATE" to determine if the conversation should end.

    Parameters:
    x (dict): A dictionary containing the message content

    Returns:
    bool: True if the message ends with "TERMINATE", False otherwise.
    """
    return x.get("content", "").rstrip().endswith("TERMINATE")

# class TrackableGroupChatManager(autogen.GroupChatManager):
#     def _process_received_message(self, message, sender, silent):
#         # Ensure message is a string
#         if isinstance(message, dict) and 'content' in message:
#             message_content = message['content']
#         elif isinstance(message, str):
#             message_content = message
#         else:
#             message_content = str(message)
        
#         # Append the message to Streamlit chat history
#         st.session_state.chat_history.append({"role": sender.name, "content": message_content})
        
#         # Also append to LangChain memory
#         if sender.name == "user":
#             memory.chat_memory.add_user_message(message_content)
#         else:
#             memory.chat_memory.add_ai_message(message_content)
        
#         # Display the message in Streamlit
#         with st.chat_message(sender.name):
#             st.markdown(message_content)
        
#         # Log the current chat history for debugging
#         logging.debug(f"Chat History: {json.dumps(st.session_state.chat_history, indent=2)}")
        
#         return super()._process_received_message(message_content, sender, silent)
class TrackableGroupChatManager(autogen.GroupChatManager):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message(message, sender, silent)

# Load documents from a URL
loader = WebBaseLoader("https://github.com/amarisg25/counselling-chatbot/blob/main/FastAPI/embeddings/HIV_PrEP_knowledge_embedding.json")
data = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)
logging.debug(f"Number of splits: {len(all_splits)}")

# Store splits in the vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))

# Initialize the LLM with the selected model
llm = ChatOpenAI(model_name=selected_model, temperature=0, openai_api_key=api_key)

# Define a prompt template that includes conversation history
prompt = PromptTemplate(
    template="""
    The following is a conversation between a patient and an HIV PrEP counselor. Use the conversation history to provide relevant and context-aware responses.

    {chat_history}

    Patient: {question}
    Counselor:""",
    input_variables=["chat_history", "question"]
)

# Initialize the LLMChain with the memory
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Patient (Chatbot-user)
patient = autogen.UserProxyAgent(
    name="patient",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=llm_config
)

# Initialize RetrievalQA
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": hub.pull("rlm/rag-prompt")}
)

def answer_question(question: str) -> str:
    """
    Answer a question based on HIV PrEP knowledge base, utilizing conversation history.

    :param question: The question to answer.
    :return: The answer as a string.
    """
    response = llm_chain.run(question=question)
    logging.debug(f"Answer Question Response: {response} (type: {type(response)})")
    if isinstance(response, str):
        return response
    else:
        return "I'm sorry, I couldn't process your request."

# Main counselor - answers general questions 
counselor = autogen.UserProxyAgent(
    name="counselor",
    system_message="You are an HIV PrEP counselor. Use the conversation history to answer user's questions concisely.",
    is_termination_msg=lambda x: check_termination(x),
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=llm_config
)
FAQ_agent = autogen.AssistantAgent(
    name="suggests_retrieve_function",
    is_termination_msg=lambda x: check_termination(x),
    system_message="Suggests function to use to answer HIV/PrEP counselling questions.",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=llm_config
)
# Initialize the group chat
group_chat = autogen.GroupChat(
    agents=[counselor, FAQ_agent, patient],
    messages=[],
)

manager = TrackableGroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    system_message="When asked a question about HIV/PREP, always call the FAQ agent before to help the counselor answer. Then have the counselor answer the question concisely using the retrieved information."
)

# Streamlit user input for chatbot interaction
st.title("HIV PrEP Counseling Chatbot")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])

# User input field
user_input = st.text_input("You: ", "")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    logging.debug(f"User Input: {user_input}")

    # Prepare to resume the chat and send the new message from the user
    last_agent, = manager.resume(messages=st.session_state.chat_history)

    # NEW MESSAGE FROM USER
    last_message = user_input

    # Resume the chat using the last agent and message
    async def initiate_chat():
        try:
            await last_agent.initiate_chat(recipient=manager, message=last_message, clear_history=False)
        except Exception as e:
            logging.error(f"Error initiating chat: {e}")
            st.session_state.chat_history.append({"role": "error", "content": "An error occurred while processing your request."})

    # Schedule the coroutine to run
    asyncio.create_task(initiate_chat())

    # Display the updated chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])