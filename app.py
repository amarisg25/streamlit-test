# import streamlit as st
# import asyncio
# from autogen import AssistantAgent, UserProxyAgent

# st.write("# AutoGen Chat Agents")

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
#         counselor = TrackableAssistantAgent(
#             name="counselor", system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions. ", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})

#         # create a UserProxyAgent instance named "user"
#         patient = TrackableUserProxyAgent(
#             name="patient", human_input_mode="ALWAYS", llm_config=llm_config,code_execution_config={"work_dir": "coding", "use_docker": False})

#         # Create an event loop
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# # Define an asynchronous function
#         async def initiate_chat():
#             await patient.a_initiate_chat(
#                 counselor,
#                 message=user_input,
#             )

#         # Run the asynchronous function within the event loop
#         loop.run_until_complete(initiate_chat())

import os
import streamlit as st
import asyncio
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager
)

# Disable Docker usage globally for AutoGen
os.environ["AUTOGEN_USE_DOCKER"] = "0"

st.set_page_config(page_title="AutoGen Group Chat Agents", page_icon="💬", layout="wide")
st.write("# AutoGen Group Chat Agents")

# Initialize Session State for GroupChat and GroupChatManager
if "group_chat" not in st.session_state:
    st.session_state.group_chat = None

if "group_chat_manager" not in st.session_state:
    st.session_state.group_chat_manager = None

# Define Agent Classes with Streamlit Integration
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

# Sidebar for OpenAI Configuration
selected_model = None
selected_key = None

with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini'], index=1)
    selected_key = st.text_input("API Key", type="password")

# Main Container for Chat Input and Display
with st.container():
    user_input = st.chat_input("Type something...")
    if user_input:
        if not selected_key or not selected_model:
            st.warning(
                'You must provide a valid OpenAI API key and choose a preferred model', icon="⚠️")
            st.stop()

        # Configure LLM Settings
        llm_config = {
            "timeout": 600,  # Updated parameter
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": selected_key
                }
            ]
        }

        # Initialize Agents and GroupChat if not already done
        if st.session_state.group_chat is None:
            # Create Agent Instances
            counselor = TrackableAssistantAgent(
                name="counselor",
                system_message="You are an HIV PrEP counselor. Call the function provided to answer user's questions.",
                llm_config=llm_config,
                code_execution_config={"work_dir": "coding", "use_docker": False}
            )

            patient = TrackableUserProxyAgent(
                name="patient",
                human_input_mode="ALWAYS",
                llm_config=llm_config,
                code_execution_config={"work_dir": "coding", "use_docker": False}
            )

            # Initialize GroupChat with Agents
            group_chat = GroupChat(
                agents=[counselor, patient],
                messages=[]
            )
            st.session_state.group_chat = group_chat

            # Initialize GroupChatManager with System Message
            system_message = (
                "When asked a question about HIV/PREP, always call the FAQ agent before "
                "to help the counselor answer. Then have the counselor answer the question "
                "concisely using the retrieved information."
            )
            group_chat_manager = GroupChatManager(
                groupchat=st.session_state.group_chat,
                llm_config=llm_config,
                system_message=system_message
            )
            st.session_state.group_chat_manager = group_chat_manager

        # Reference to GroupChatManager
        manager = st.session_state.group_chat_manager

        # Append User Message to GroupChat
        st.session_state.group_chat.messages.append({
            "sender": "patient",
            "message": user_input
        })

        # Create an Event Loop if not already present
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Define an Asynchronous Function to Handle Chat
        async def initiate_group_chat():
            await manager.handle_message("patient", user_input)

        # Run the Asynchronous Function
        loop.run_until_complete(initiate_group_chat())
