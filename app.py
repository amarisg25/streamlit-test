# import streamlit as st
# import asyncio
# from autogen import AssistantAgent, UserProxyAgent
import os
import streamlit as st
from dotenv import load_dotenv
# import json
# from langchain_community.document_loaders import DirectoryLoader, JSONLoader, WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain import hub
# import autogen
# from langchain.tools import BaseTool, StructuredTool, Tool, tool
# from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
# import asyncio


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
st.write("# AutoGen Chat Agents")

# class TrackableAssistantAgent(AssistantAgent):
#     def _process_received_message(self, message, sender, silent):
#         with st.chat_message(sender.name):
#             st.markdown(message)
#         return super()._process_received_message(message, sender, silent)

# class TrackableUserProxyAgent(UserProxyAgent):
#     def _process_received_message(self, message, sender, silent):
#         with st.chat_message(sender.name):
#             st.markdown(message)
#         return super()._process_received_message(message, sender, silent)
    


# selected_model = None
# selected_key = None

# with st.sidebar:
#     st.header("OpenAI Configuration")
#     selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini'], index=1)
#     selected_key = st.text_input("API Key", type="password")


# #  Load documents from a URL
# # loader = JSONLoader('embeddings/HIV_PrEP_knowledge_embedding.json', jq_schema='.quiz', text_content=False)
# loader = WebBaseLoader("https://github.com/amarisg25/counselling-chatbot/blob/930a7b8deabab8ad286856536d499164968df7a1/embeddings/HIV_PrEP_knowledge_embedding.json")
# data = loader.load()

# # Split documents into manageable chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# all_splits = text_splitter.split_documents(data)
# print(f"Number of splits: {len(all_splits)}")
# # Check the contents of all_splits
# # for i, split in enumerate(all_splits):
# #     print(f"Split Document {i}: {split}")
# #
# # Store splits in the vector store
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))

# # Initialize the LLM with the correct model
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

 
# # Initialize RetrievalQA
# prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
# retriever = vectorstore.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(
#     llm, retriever= retriever, chain_type_kwargs={"prompt": prompt}
# )

# def answer_question(question: str) -> str:
#     """
#     Answer a question based on HIV PrEP knowledge base.

#     :param question: The question to answer.
#     :return: The answer as a string.
#     """
#     result = qa_chain.invoke({"query": question})
#     return result.get("result", "I'm sorry, I couldn't find an answer to that question.")


# with st.container():
#     # for message in st.session_state["messages"]:
#     #    st.markdown(message)

#     user_input = st.chat_input("Type something...")
#     if user_input:
#         if not selected_key or not selected_model:
#             st.warning(
#                 'You must provide valid OpenAI API key and choose preferred model', icon="⚠️")
#             st.stop()

#         llm_config = {
#             "timeout": 600,
#             "config_list": [
#                 {
#                     "model": selected_model,
#                     "api_key": selected_key
#                 }
#             ]
#         }

#         # create an AssistantAgent instance named "assistant"
#         counselor = TrackableUserProxyAgent(
#             name="counselor", system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions. ", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})

#         # create a UserProxyAgent instance named "user"
#         patient = TrackableUserProxyAgent(
#             name="patient", human_input_mode="ALWAYS", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})


#         FAQ_agent = TrackableAssistantAgent(
#             name="suggests_retrieve_function",
#             system_message="Suggests function to use to answer HIV/PrEPcounselling questions",
#             human_input_mode="NEVER",
#             code_execution_config={"work_dir":"coding", "use_docker":False},
#             llm_config=llm_config
#         )

#         # Register the wrapper function with autogen
#         autogen.agentchat.register_function(
#             answer_question,
#             caller=FAQ_agent,
#             executor=counselor,
#             name="answer_question",
#             description="Retrieves embedding data content to answer user's question.",
#         )

#         # Create an event loop
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# # Define an asynchronous function
#         async def initiate_chat():
#             await patient.a_initiate_chat(
#                 FAQ_agent,
#                 message=user_input,
#             )

#         # Run the asynchronous function within the event loop
#         loop.run_until_complete(initiate_chat())


import asyncio
from autogen import AssistantAgent, UserProxyAgent

st.write("# AutoGen Chat Agents")

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
    


# with st.sidebar:
#     st.header("OpenAI Configuration")
#     selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini'], index=1)
#     selected_key = st.text_input("API Key", type="password")

with st.container():
    # for message in st.session_state["messages"]:
    #    st.markdown(message)

    user_input = st.chat_input("Type something...")
    if user_input:
        selected_model = 'gpt-4o-mini'
        selected_key = api_key

        # if not selected_key or not selected_model:
        #     st.warning(
        #         'You must provide valid OpenAI API key and choose preferred model', icon="⚠️")
        #     st.stop()

        llm_config = {
            "timeout": 600,
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": selected_key
                }
            ]
        }

        # create an AssistantAgent instance named "assistant"
        counselor = TrackableAssistantAgent(
            name="counselor", system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions. ", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})

        # create a UserProxyAgent instance named "user"
        patient = TrackableUserProxyAgent(
            name="patient", human_input_mode="ALWAYS", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Define an asynchronous function
        async def initiate_chat():
            await patient.a_initiate_chat(
                counselor,
                message=user_input,
            )

        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())