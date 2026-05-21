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

# ============ LangChain 核心导入 ============
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.schema import HumanMessage, AIMessage

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
    .feature-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
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
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============ LangSmith 监控类 ============
class LangSmithTracer:
    """简单的LangSmith追踪器实现"""
    
    def __init__(self, api_key: str, project_name: str = "default"):
        self.api_key = api_key
        self.project_name = project_name
        self.endpoint = "https://api.smith.langchain.com"
        self.run_id = None
    
    def start_trace(self, run_name: str, inputs: dict):
        """开始追踪一个运行"""
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
                f"{self.endpoint}/runs",
                headers=headers,
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                self.run_id = response.json().get("id")
                return self.run_id
        except Exception as e:
            st.sidebar.warning(f"LangSmith追踪启动失败: {e}")
        return None
    
    def end_trace(self, outputs: dict, error: str = None):
        """结束追踪"""
        if not self.run_id or not self.api_key:
            return
        try:
            import requests
            headers = {"x-api-key": self.api_key}
            data = {"outputs": outputs}
            if error:
                data["error"] = error
            requests.patch(
                f"{self.endpoint}/runs/{self.run_id}",
                headers=headers,
                json=data,
                timeout=5
            )
        except Exception as e:
            pass

def get_langsmith_tracer() -> Optional[LangSmithTracer]:
    """获取LangSmith追踪器实例"""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key and os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        return LangSmithTracer(
            api_key=api_key,
            project_name=os.getenv("LANGCHAIN_PROJECT", "langchain-chat-assistant")
        )
    return None

# ============ 工具函数定义 ============
def get_current_time(location: str = "UTC") -> str:
    """获取指定时区的当前时间"""
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"当前{location}时间: {current_time}"
    except Exception as e:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

def calculate(expression: str) -> str:
    """计算数学表达式"""
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
    """模拟知识库搜索"""
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

# ============ LangChain 组件初始化 ============

@st.cache_resource
def get_llm():
    """获取LLM模型实例（缓存）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("⚠️ DEEPSEEK_API_KEY 环境变量未设置")
        st.info("请在Streamlit Cloud的Secrets中设置 DEEPSEEK_API_KEY")
        return None
    
    return ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.7,
        streaming=False
    )

@st.cache_resource
def get_tools():
    """获取工具列表"""
    return [
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="获取指定时区的当前时间"
        ),
        Tool(
            name="calculate",
            func=calculate,
            description="计算数学表达式"
        ),
        Tool(
            name="search_knowledge",
            func=search_knowledge,
            description="搜索知识库获取信息"
        )
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

# 创建Prompt模板
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# ============ Chain 链式调用构建 ============

def create_conversation_chain(memory):
    """创建对话Chain - Chain链式调用"""
    llm = get_llm()
    if not llm:
        return None
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False
    )
    return chain

def create_agent_chain():
    """创建Agent Chain（支持工具调用）"""
    llm = get_llm()
    if not llm:
        return None
    
    tools = get_tools()
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    )
    return agent

# ============ 会话状态管理 ============

def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "use_agent" not in st.session_state:
        st.session_state.use_agent = False
    if "tracer" not in st.session_state:
        st.session_state.tracer = get_langsmith_tracer()

# ============ 主应用 ============

def main():
    # 初始化会话状态
    init_session_state()
    
    # 页面标题
    st.markdown('<h1 class="main-header">🤖 LangChain 对话助手</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title("⚙️ 配置")
        
        # API状态检查
        llm = get_llm()
        if llm:
            st.success("✅ API连接正常")
        else:
            st.error("❌ API未配置")
        
        # LangSmith状态
        langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if langsmith_enabled:
            st.success("✅ LangSmith监控已启用")
        else:
            st.info("ℹ️ LangSmith监控未启用")
        
        st.divider()
        
        # 功能开关
        st.session_state.use_agent = st.toggle(
            "使用Agent模式（支持工具调用）",
            value=st.session_state.use_agent
        )
        
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state.chain = None
            st.rerun()
        
        st.divider()
        
        # 功能说明
        st.title("📖 功能说明")
        st.markdown("""
        **核心特性：**
        
        1. **LLM调用**
           - 使用DeepSeek大模型
           - OpenAI兼容接口
        
        2. **Prompt工程**
           - SystemMessagePromptTemplate
           - MessagesPlaceholder
           - 结构化提示词
        
        3. **Chain链式调用**
           - LLMChain组合
           - 模块化设计
        
        4. **Memory记忆**
           - ConversationBufferMemory
           - 多轮对话上下文
        
        5. **Tool工具使用**
           - ⏰ 获取当前时间
           - 🧮 数学计算
           - 🔍 知识搜索
        
        6. **LangSmith监控**
           - 全流程追踪
           - 性能监控
        """)
    
    # 主界面
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 显示对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>👤 用户:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<strong>🤖 助手:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
    
    with col2:
        st.subheader("📊 统计")
        st.metric("对话轮数", len(st.session_state.messages) // 2)
        st.metric("当前模式", "Agent" if st.session_state.use_agent else "基础")
        
        # 显示工具列表
        st.subheader("🔧 可用工具")
        tools = get_tools()
        for tool in tools:
            with st.expander(f"📌 {tool.name}"):
                st.write(tool.description)
    
    # 输入框
    st.divider()
    user_input = st.text_input(
        "请输入您的问题...",
        key="user_input",
        placeholder="例如：现在几点了？或者 1+2*3等于多少？"
    )
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 显示用户消息
        with col1:
            st.markdown(
                f'<div class="chat-message user-message">'
                f'<strong>👤 用户:</strong> {user_input}</div>',
                unsafe_allow_html=True
            )
        
        # 生成回复
        with st.spinner("🤔 思考中..."):
            try:
                tracer = st.session_state.tracer
                
                if st.session_state.use_agent:
                    # Agent模式 - Tool工具使用
                    if tracer:
                        tracer.start_trace("agent_chat", {"message": user_input})
                    
                    agent = create_agent_chain()
                    if agent:
                        response = agent.run(input=user_input)
                    else:
                        response = "❌ API未配置，无法使用Agent模式"
                    
                    if tracer:
                        tracer.end_trace({"response": response})
                else:
                    # 基础Chain模式
                    if tracer:
                        tracer.start_trace("chain_chat", {"message": user_input})
                    
                    if not st.session_state.chain:
                        st.session_state.chain = create_conversation_chain(st.session_state.memory)
                    
                    if st.session_state.chain:
                        response = st.session_state.chain.predict(input=user_input)
                    else:
                        response = "❌ API未配置，无法生成回复"
                    
                    if tracer:
                        tracer.end_trace({"response": response})
                
                # 添加助手消息
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 显示助手消息
                with col1:
                    st.markdown(
                        f'<div class="chat-message assistant-message">'
                        f'<strong>🤖 助手:</strong> {response}</div>',
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                error_msg = f"❌ 错误: {str(e)}"
                st.error(error_msg)
                if tracer:
                    tracer.end_trace({}, error=str(e))
        
        # 清空输入框
        st.rerun()

if __name__ == "__main__":
    main()
