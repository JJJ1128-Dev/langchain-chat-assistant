import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="LangChain Chat Assistant", page_icon="🤖")

st.title("🤖 LangChain 对话助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for message in st.session_state.messages:
    st.write(f"**{message['role']}:** {message['content']}")

def send_message():
    prompt = st.session_state.input_text
    if prompt.strip() == "":
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/memory",
            json={"messages": st.session_state.messages}
        )
        
        if response.status_code == 200:
            full_response = response.json().get("response", "暂无响应")
        else:
            full_response = f"服务器错误: {response.status_code}"
    except Exception as e:
        full_response = f"连接错误: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.input_text = ""

st.text_input("请输入您的问题...", key="input_text", on_change=send_message)

st.sidebar.title("功能说明")
st.sidebar.info(
    "这是一个基于LangChain框架构建的对话助手，具备以下特性：\n\n"
    "**核心能力：**\n"
    "- LLM调用（OpenAI GPT-3.5）\n"
    "- Prompt工程优化\n"
    "- Chain链式调用\n"
    "- ConversationBufferMemory记忆\n"
    "- 工具调用（时间查询、计算器）\n\n"
    "**技术架构：**\n"
    "- 后端：LangServe + FastAPI\n"
    "- 前端：Streamlit\n"
    "- 监控：LangSmith\n\n"
    "**支持的工具：**\n"
    "- ⏰ 获取当前时间\n"
    "- 🧮 数学计算"
)
