import streamlit as st
import requests
import uuid

st.set_page_config(page_title="LangChain Chat Assistant", page_icon="🤖", layout="wide")

# 后端地址（部署时修改）
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .chat-message { padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .assistant-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.title("⚙️ 配置")
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.success("✅ 后端服务正常 (LangServe)")
        else:
            st.error("❌ 后端服务异常")
    except:
        st.error("❌ 无法连接后端")
        st.info(f"请确保后端已部署，并修改 BACKEND_URL")
    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages.clear()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.markdown("""
    **核心功能（满足作业要求）**
    1. LLM调用 (DeepSeek)
    2. Prompt工程
    3. Chain链式调用
    4. Memory记忆
    5. Tool工具使用
       - ⏰ 当前时间
       - 🧮 数学计算
       - ☀️ 天气查询
    6. LangServe 部署
    """)

st.markdown('<h1 class="main-header">🤖 LangChain 对话助手</h1>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("请输入你的问题...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat/agent",
                    json={"message": user_input, "session_id": st.session_state.session_id},
                    timeout=30
                )
                if response.status_code == 200:
                    reply = response.json()["response"]
                else:
                    reply = f"API 错误: {response.status_code}"
            except Exception as e:
                reply = f"连接错误: {e}"
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
