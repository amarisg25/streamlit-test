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
import autogen
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()


# Load environment variables
load_dotenv()
env_api_key = os.getenv('OPENAI_API_KEY')

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
        # value=env_api_key if env_api_key else "",
        key="api_key_input"  # Optional: Add a unique key
    )

# Determine which API key to use
api_key = user_api_key 


if not api_key:
    st.warning('Please provide a valid OpenAI API key in the sidebar.', icon="⚠️")
    st.stop()

# if user_api_key else env_api_key
config_list = {
    "model": "gpt-4o-mini", 
    "api_key": api_key 
}

@st.cache_resource
def initialize_vectorstore(api_key):
    # Load documents from a URL
    loader = WebBaseLoader("https://raw.githubusercontent.com/amarisg25/counselling-chatbot/930a7b8deabab8ad286856536d499164968df7a1/embeddings/HIV_PrEP_knowledge_embedding.json")
    data = loader.load()
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    
    print(api_key)
    # Initialize the vector store
    return Chroma.from_documents(
        documents=all_splits, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
    )

# Initialize the vector store
vectorstore = initialize_vectorstore(api_key)

# Initialize the LLM with the selected model
llm = ChatOpenAI(model_name=selected_model, temperature=0, openai_api_key=api_key)

# Initialize RetrievalQA
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm, 
    retriever=retriever, 
    chain_type_kwargs={"prompt": prompt}
)

def answer_question(question: str) -> str:
    """
    Answer a question based on HIV PrEP knowledge base.

    :param question: The question to answer.
    :return: The answer as a string.
    """
    result = qa_chain.invoke({"query": question})
    return result.get("result", "I'm sorry, I couldn't find an answer to that question.")

class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

class TrackableGroupChatManager(autogen.GroupChatManager):
    def _process_received_message(self, message, sender, silent):
        # Log received message
        print(f"Received message from {sender.name}: {message}")

        # Your existing code...
        self.messages.append((sender.name, message))
        with st.chat_message(sender.name):
            st.markdown(message)
        
        # Call the parent class's method
        return super()._process_received_message(message, sender, silent)


# Streamlit Container for Chat
with st.container():
    user_input = st.chat_input("Type something...")
    if user_input:
        if not user_api_key or not selected_model:
            st.warning(
                'You must provide a valid OpenAI API key and choose a preferred model.', 
                icon="⚠️"
            )
            st.stop()

        llm_config = {
            "timeout": 600,
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": api_key
                }
            ]
        }

       

        # Create an AssistantAgent instance named "counselor"
        counselor = TrackableUserProxyAgent(
            name="counselor", 
            system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions.", 
            llm_config=llm_config,
            code_execution_config={"work_dir": "coding", "use_docker": False}
        )

        # Create a UserProxyAgent instance named "patient"
        patient = TrackableUserProxyAgent(
            name="patient", 
            human_input_mode="ALWAYS", 
            llm_config=llm_config,
            code_execution_config={"work_dir": "coding", "use_docker": False}
        )

        FAQ_agent = TrackableAssistantAgent(
            name="suggests_retrieve_function",
            system_message="Suggests function to use to answer HIV/PrEP counselling questions",
            human_input_mode="NEVER",
            code_execution_config={"work_dir":"coding", "use_docker":False},
            llm_config=llm_config
        )

        # Register the wrapper function with autogen
        autogen.agentchat.register_function(
            answer_question,
            caller=FAQ_agent,
            executor=counselor,
            name="answer_question",
            description="Retrieves embedding data content to answer user's question.",
        )

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        agents = [counselor, FAQ_agent, patient]

        group_chat = autogen.GroupChat(
            agents= agents, 
            messages=[], 
            )
        
        manager = TrackableGroupChatManager(
            groupchat=group_chat, 
            llm_config=config_list, 
            system_message="When asked a question about HIV/PREP, always call the FAQ agent before to help the counselor answer. Then have the counselor answer the question concisely using the retrieved information."
        )
        manager._process_received_message(user_input, patient, silent=False)
        # Define an asynchronous function
        async def initiate_chat():
            await patient.a_initiate_chat(
                manager,
                message=user_input,
            )

        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
