"""
LangChain Chat Assistant - Streamlit Cloud 部署版本
合并前后端，用于云端部署展示

功能特性：
- LLM调用 (DeepSeek)
- Prompt工程
- Chain链式调用
- ConversationBufferMemory记忆
- Tool工具使用
- LangSmith监控
"""

import streamlit as st
import os
from datetime import datetime
import pytz
from typing import List, Dict, Optional

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# ============ LangChain 核心导入（新版API） ============
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# ============ 页面配置 ============
st.set_page_config(
    page_title="LangChain Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义CSS样式 ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# ============ LangSmith 监控类 ============
class LangSmithTracer:
    def __init__(self, api_key: str, project_name: str = "default"):
        self.api_key = api_key
        self.project_name = project_name
        self.run_id = None
    
    def start_trace(self, run_name: str, inputs: dict):
        if not self.api_key:
            return None
        try:
            import requests
            headers = {"x-api-key": self.api_key}
            data = {
                "name": run_name,
                "run_type": "chain",
                "inputs": inputs,
                "project_name": self.project_name
            }
            response = requests.post(
                "https://api.smith.langchain.com/runs",
                headers=headers,
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                self.run_id = response.json().get("id")
        except Exception:
            pass
        return None
    
    def end_trace(self, outputs: dict, error: str = None):
        if not self.run_id or not self.api_key:
            return
        try:
            import requests
            data = {"outputs": outputs}
            if error:
                data["error"] = error
            requests.patch(
                f"https://api.smith.langchain.com/runs/{self.run_id}",
                headers={"x-api-key": self.api_key},
                json=data,
                timeout=5
            )
        except Exception:
            pass

def get_langsmith_tracer() -> Optional[LangSmithTracer]:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key and os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        return LangSmithTracer(
            api_key=api_key,
            project_name=os.getenv("LANGSMITH_PROJECT", "langchain-chat-assistant")
        )
    return None

# ============ 工具函数定义 ============
def get_current_time(location: str = "UTC") -> str:
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"当前{location}时间: {current_time}"
    except Exception:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

def calculate(expression: str) -> str:
    try:
        allowed_names = {"__builtins__": {}}
        allowed_names.update({
            'abs': abs, 'max': max, 'min': min, 'sum': sum,
            'pow': pow, 'round': round
        })
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_knowledge(query: str) -> str:
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

# ============ LangChain 组件初始化 ============
@st.cache_resource
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
        streaming=False
    )

@st.cache_resource
def get_tools():
    return [
        Tool(name="get_current_time", func=get_current_time, description="获取指定时区的当前时间"),
        Tool(name="calculate", func=calculate, description="计算数学表达式"),
        Tool(name="search_knowledge", func=search_knowledge, description="搜索知识库获取信息")
    ]

# ============ Prompt工程 ============
SYSTEM_PROMPT = """你是一个聪明、友好、知识渊博的对话助手，能够回答各种问题、提供帮助。

你拥有以下工具可以使用：
1. get_current_time - 获取指定时区的当前时间
2. calculate - 计算复杂数学表达式
3. search_knowledge - 搜索知识库获取信息

**使用工具的规则：**
- 当用户询问时间相关问题时，使用 get_current_time 工具
- 当用户询问数学计算问题时，使用 calculate 工具
- 当用户询问知识性问题时，使用 search_knowledge 工具

请用中文进行友好、自然的回复。"""

# 创建Prompt模板（新版使用 from_messages 直接传元组列表）
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ============ Chain 链式调用 ============
def create_conversation_chain(memory):
    llm = get_llm()
    if not llm:
        return None
    # 使用 LLMChain（新版仍然支持）
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False
    )
    return chain

def create_agent_executor():
    """创建支持工具的 Agent（新版 API）"""
    llm = get_llm()
    if not llm:
        return None
    tools = get_tools()
    # 创建 tool calling agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    )
    return executor

# ============ 会话状态管理 ============
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "use_agent" not in st.session_state:
        st.session_state.use_agent = False
    if "tracer" not in st.session_state:
        st.session_state.tracer = get_langsmith_tracer()

# ============ 主应用 ============
def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">🤖 LangChain 对话助手</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title("⚙️ 配置")
        llm = get_llm()
        if llm:
            st.success("✅ API连接正常")
        else:
            st.error("❌ API未配置")
        
        langsmith_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        if langsmith_enabled:
            st.success("✅ LangSmith监控已启用")
        else:
            st.info("ℹ️ LangSmith监控未启用")
        
        st.divider()
        st.session_state.use_agent = st.toggle("使用Agent模式（支持工具调用）", value=st.session_state.use_agent)
        
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.chain = None
            st.session_state.agent_executor = None
            st.rerun()
        
        st.divider()
        st.title("📖 功能说明")
        st.markdown("""
        **核心特性：**  
        1. LLM调用 (DeepSeek)  
        2. Prompt工程 (ChatPromptTemplate)  
        3. Chain链式调用 (LLMChain)  
        4. Memory记忆 (ConversationBufferMemory)  
        5. Tool工具使用 (Agent + Tools)  
        6. LangSmith监控
        """)
    
    # 显示对话历史
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>👤 用户:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 助手:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    
    # 输入框
    user_input = st.chat_input("请输入您的问题...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-message user-message"><strong>👤 用户:</strong> {user_input}</div>', unsafe_allow_html=True)
        
        with st.spinner("🤔 思考中..."):
            try:
                tracer = st.session_state.tracer
                
                if st.session_state.use_agent:
                    if tracer:
                        tracer.start_trace("agent_chat", {"message": user_input})
                    
                    if st.session_state.agent_executor is None:
                        st.session_state.agent_executor = create_agent_executor()
                    
                    executor = st.session_state.agent_executor
                    if executor:
                        response = executor.invoke({"input": user_input})["output"]
                    else:
                        response = "❌ API未配置，无法使用Agent模式"
                    
                    if tracer:
                        tracer.end_trace({"response": response})
                else:
                    if tracer:
                        tracer.start_trace("chain_chat", {"message": user_input})
                    
                    if st.session_state.chain is None:
                        st.session_state.chain = create_conversation_chain(st.session_state.memory)
                    
                    if st.session_state.chain:
                        response = st.session_state.chain.predict(input=user_input)
                    else:
                        response = "❌ API未配置，无法生成回复"
                    
                    if tracer:
                        tracer.end_trace({"response": response})
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 助手:</strong> {response}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                error_msg = f"❌ 错误: {str(e)}"
                st.error(error_msg)
                if tracer:
                    tracer.end_trace({}, error=str(e))
        
        st.rerun()

if __name__ == "__main__":
    main()
