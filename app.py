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
import chromadb

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
    bool: True if the message ends with "TERMINATE", False otherwise
    """
    return x.get("content", "").rstrip().endswith("TERMINATE")

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
print(f"Number of splits: {len(all_splits)}")

# Store splits in the vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))

# Initialize the LLM with the selected model
llm = ChatOpenAI(model_name=selected_model, temperature=0, openai_api_key=api_key)

# Patient (Chatbot-user)
patient = autogen.UserProxyAgent(
    name="patient",
    human_input_mode="NEVER",
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
    Answer a question based on HIV PrEP knowledge base.

    :param question: The question to answer.
    :return: The answer as a string.
    """
    result = qa_chain.invoke({"query": question})
    return result.get("result", "I'm sorry, I couldn't find an answer to that question.")

# Main counselor - answers general questions 
counselor = autogen.UserProxyAgent(
    name="counselor",
    system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions.",
    is_termination_msg=lambda x: check_termination(x),
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=llm_config
)

# FAQ agent - provides context for the counselor
FAQ_agent = autogen.AssistantAgent(
    name="suggests_retrieve_function",
    is_termination_msg=lambda x: check_termination(x),
    system_message="Suggests function to use to answer HIV/PrEP counselling questions",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=llm_config
)

autogen.agentchat.register_function(
    answer_question,
    caller=FAQ_agent,
    executor=counselor,
    name="answer_question",
    description="Retrieves embedding data content to answer user's question.",
)


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# INITIALIZE THE GROUP CHAT
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

    # Process the message
    manager._process_received_message(user_input, patient, silent=False)

    # Async chat initiation
    async def initiate_chat():
        await patient.a_initiate_chat(manager, message=user_input)

    # Call the function to initiate chat
    loop.run_until_complete(initiate_chat())

    # Display the updated chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])
