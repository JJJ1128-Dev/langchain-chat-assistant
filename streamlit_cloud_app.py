"""
LangChain Chat Assistant - Streamlit Cloud 部署版本
兼容 LangChain 1.x 新 API
满足作业要求：LLM调用、Prompt工程、Chain链式调用、Memory记忆、Tool工具使用
"""

import streamlit as st
import os
from datetime import datetime
import pytz
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ============ 新版 LangChain 导入（全部正确） ============
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# ============ 页面配置 ============
st.set_page_config(page_title="LangChain 对话助手", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .assistant-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
</style>
""", unsafe_allow_html=True)

# ============ LangSmith 追踪（可选） ============
class SimpleTracer:
    def __init__(self):
        self.enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.api_key = os.getenv("LANGSMITH_API_KEY")
    def start_trace(self, name, inputs): pass
    def end_trace(self, outputs, error=None): pass

def get_tracer():
    return SimpleTracer()

# ============ 工具函数 ============
def get_current_time(location: str = "UTC") -> str:
    try:
        tz = pytz.timezone(location)
        return f"当前{location}时间: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "max": max, "min": min, "sum": sum, "pow": pow, "round": round})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

def search_knowledge(query: str) -> str:
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

# ============ LangChain 组件 ============
@st.cache_resource
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    )

@st.cache_resource
def get_tools():
    return [
        Tool(name="get_current_time", func=get_current_time, description="获取指定时区时间"),
        Tool(name="calculate", func=calculate, description="计算数学表达式"),
        Tool(name="search_knowledge", func=search_knowledge, description="搜索知识库")
    ]

# ============ Prompt 工程 ============
SYSTEM_PROMPT = """你是一个友好的AI助手。你有以下工具：
- get_current_time: 获取时间
- calculate: 计算数学表达式
- search_knowledge: 搜索知识

用户需要时请主动使用工具。用中文回复。"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ============ 会话状态 ============
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "use_agent" not in st.session_state:
        st.session_state.use_agent = False
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

# ============ 创建 Chain 和 Agent ============
def create_chain(memory):
    llm = get_llm()
    if not llm:
        return None
    # 简化版 Chain：直接使用 LLMChain（新版仍然可用）
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    return LLMChain(llm=llm, prompt=prompt, memory=memory)

def create_agent_executor():
    llm = get_llm()
    if not llm:
        return None
    tools = get_tools()
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=False,
        handle_parsing_errors=True
    )

# ============ 主界面 ============
def main():
    init_state()
    st.markdown('<h1 class="main-header">🤖 LangChain 对话助手</h1>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.title("⚙️ 配置")
        llm = get_llm()
        if llm:
            st.success("✅ DeepSeek API 正常")
        else:
            st.error("❌ 未设置 DEEPSEEK_API_KEY")
            st.info("请在 Streamlit Cloud Secrets 中配置")
        st.divider()
        st.session_state.use_agent = st.toggle("🔧 使用工具模式 (Agent)", value=st.session_state.use_agent)
        if st.button("🗑️ 清空对话"):
            st.session_state.messages.clear()
            st.session_state.memory.clear()
            st.session_state.chain = None
            st.session_state.agent_executor = None
            st.rerun()
        st.divider()
        st.markdown("**核心功能**\n- LLM调用 (DeepSeek)\n- Prompt工程\n- Chain链式调用\n- Memory记忆\n- Tool工具使用")

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 输入
    user_input = st.chat_input("请输入你的问题...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    tracer = get_tracer()
                    if st.session_state.use_agent:
                        tracer.start_trace("agent", {"input": user_input})
                        if st.session_state.agent_executor is None:
                            st.session_state.agent_executor = create_agent_executor()
                        executor = st.session_state.agent_executor
                        if executor:
                            response = executor.invoke({"input": user_input})["output"]
                        else:
                            response = "❌ Agent 初始化失败，请检查 API Key"
                        tracer.end_trace({"response": response})
                    else:
                        tracer.start_trace("chain", {"input": user_input})
                        if st.session_state.chain is None:
                            st.session_state.chain = create_chain(st.session_state.memory)
                        if st.session_state.chain:
                            response = st.session_state.chain.predict(input=user_input)
                        else:
                            response = "❌ Chain 初始化失败，请检查 API Key"
                        tracer.end_trace({"response": response})

                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"发生错误: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"错误: {e}"})

if __name__ == "__main__":
    main()
