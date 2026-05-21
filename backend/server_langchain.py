"""
LangChain Chat Assistant - 使用真正的LangChain组件实现
包含: LLM调用、Prompt工程、Chain链式调用、Memory记忆、Tool工具使用、LangSmith监控
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import pytz
import json
import requests as http_requests

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# ============ LangChain 核心导入 ============
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

# ============ FastAPI 应用初始化 ============
app = FastAPI(
    title="LangChain Chat Assistant",
    description="基于LangChain的对话助手，集成LangSmith监控",
    version="2.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 数据模型 ============
class ChatMessage(BaseModel):
    role: str = Field(description="消息角色: user 或 assistant")
    content: str = Field(description="消息内容")

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field(description="对话历史")

class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")
    session_id: Optional[str] = Field(default="default", description="会话ID")

class ChatResponse(BaseModel):
    response: str = Field(description="助手回复")
    session_id: str = Field(description="会话ID")

# ============ LangSmith 监控类 ============
class LangSmithTracer:
    """简单的LangSmith追踪器实现"""
    
    def __init__(self, api_key: str, project_name: str = "default"):
        self.api_key = api_key
        self.project_name = project_name
        self.endpoint = "https://api.smith.langchain.com"
        self.session_id = None
        self.run_id = None
    
    def start_trace(self, run_name: str, inputs: dict):
        """开始追踪一个运行"""
        if not self.api_key:
            return None
        
        try:
            headers = {"x-api-key": self.api_key}
            data = {
                "name": run_name,
                "run_type": "chain",
                "inputs": inputs,
                "project_name": self.project_name
            }
            response = http_requests.post(
                f"{self.endpoint}/runs",
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                self.run_id = response.json().get("id")
                return self.run_id
        except Exception as e:
            print(f"LangSmith start_trace error: {e}")
        return None
    
    def end_trace(self, outputs: dict, error: str = None):
        """结束追踪"""
        if not self.run_id or not self.api_key:
            return
        
        try:
            headers = {"x-api-key": self.api_key}
            data = {"outputs": outputs}
            if error:
                data["error"] = error
            
            http_requests.patch(
                f"{self.endpoint}/runs/{self.run_id}",
                headers=headers,
                json=data
            )
        except Exception as e:
            print(f"LangSmith end_trace error: {e}")
    
    def log_event(self, event_type: str, data: dict):
        """记录事件"""
        if not self.api_key:
            return
        
        try:
            headers = {"x-api-key": self.api_key}
            event_data = {
                "run_id": self.run_id,
                "event_type": event_type,
                "data": data
            }
            http_requests.post(
                f"{self.endpoint}/events",
                headers=headers,
                json=event_data
            )
        except Exception as e:
            print(f"LangSmith log_event error: {e}")

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
        # 安全计算：只允许基本数学运算
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

# ============ LangChain 工具配置 ============
langchain_tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="获取指定时区的当前时间，参数location为时区名称如'Asia/Shanghai'、'UTC'等"
    ),
    Tool(
        name="calculate",
        func=calculate,
        description="计算数学表达式，参数expression为数学表达式字符串如'1+2*3'、'sqrt(16)'等"
    ),
    Tool(
        name="search_knowledge",
        func=search_knowledge,
        description="搜索知识库获取信息，参数query为搜索关键词"
    )
]

# ============ LangChain 组件初始化 ============

def get_llm():
    """获取LLM模型实例"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    
    # 使用DeepSeek API（通过OpenAI兼容接口）
    return ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.7,
        streaming=False
    )

# 存储会话记忆
session_memories: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str = "default") -> ConversationBufferMemory:
    """获取或创建会话记忆"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]

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

**直接回答的规则：**
- 对于一般性问题、闲聊、知识问答等，直接用自然语言回答
- 不要过度依赖工具，只有在确实需要时才调用
- 如果无法回答或不确定，可以礼貌地说明

请用中文进行友好、自然的回复。"""

# 创建Prompt模板 - Prompt工程
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# ============ Chain 链式调用构建 ============

def create_conversation_chain(session_id: str = "default", tracer: LangSmithTracer = None):
    """创建对话Chain - Chain链式调用"""
    llm = get_llm()
    memory = get_memory(session_id)
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True  # 开启详细日志
    )
    
    return chain, tracer

def create_agent_chain(tracer: LangSmithTracer = None):
    """创建Agent Chain（支持工具调用）- Tool工具使用"""
    llm = get_llm()
    
    agent = initialize_agent(
        langchain_tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    )
    
    return agent, tracer

# ============ API 路由 ============

@app.post("/chat/simple", response_model=ChatResponse)
async def chat_simple(request: ChatRequest):
    """
    简单对话接口（使用基础Chain）- LLM调用 + Prompt工程 + Chain链式调用
    """
    tracer = get_langsmith_tracer()
    
    try:
        # LangSmith追踪开始
        if tracer:
            tracer.start_trace("chat_simple", {"message": request.message, "session_id": request.session_id})
        
        chain, _ = create_conversation_chain(request.session_id, tracer)
        response = chain.predict(input=request.message)
        
        # LangSmith追踪结束
        if tracer:
            tracer.end_trace({"response": response})
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/agent", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    """
    Agent对话接口（支持工具调用）- Tool工具使用
    """
    tracer = get_langsmith_tracer()
    
    try:
        if tracer:
            tracer.start_trace("chat_agent", {"message": request.message})
        
        agent, _ = create_agent_chain(tracer)
        response = agent.run(input=request.message)
        
        if tracer:
            tracer.end_trace({"response": response})
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/memory")
async def chat_with_memory(chat_history: ChatHistory):
    """
    带记忆的对话接口（兼容旧版API）- Memory记忆
    """
    tracer = get_langsmith_tracer()
    
    try:
        if not chat_history.messages:
            return {"response": "请输入您的问题"}
        
        last_message = chat_history.messages[-1]
        if last_message.role != "user":
            return {"response": "最后一条消息必须是用户消息"}
        
        if tracer:
            tracer.start_trace("chat_memory", {
                "history_length": len(chat_history.messages),
                "last_message": last_message.content
            })
        
        # 创建新的会话记忆并加载历史消息 - ConversationBufferMemory
        session_id = "session_" + str(hash(str(chat_history.messages)))
        memory = get_memory(session_id)
        
        # 将历史消息加载到记忆中
        for i in range(0, len(chat_history.messages) - 1, 2):
            if i + 1 < len(chat_history.messages):
                user_msg = chat_history.messages[i]
                assistant_msg = chat_history.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    memory.chat_memory.add_user_message(user_msg.content)
                    memory.chat_memory.add_ai_message(assistant_msg.content)
        
        # 创建Chain并预测
        chain, _ = create_conversation_chain(session_id, tracer)
        response = chain.predict(input=last_message.content)
        
        if tracer:
            tracer.end_trace({"response": response})
        
        return {"response": response}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============ LangServe 风格的标准化路由 ============

@app.post("/langserve/invoke")
async def langserve_invoke(request: dict):
    """
    LangServe风格的invoke端点 - 支持高并发访问
    """
    tracer = get_langsmith_tracer()
    
    try:
        input_data = request.get("input", "")
        config = request.get("config", {})
        session_id = config.get("session_id", "default")
        
        if tracer:
            tracer.start_trace("langserve_invoke", {"input": input_data, "config": config})
        
        chain, _ = create_conversation_chain(session_id, tracer)
        response = chain.predict(input=input_data)
        
        if tracer:
            tracer.end_trace({"output": response})
        
        return {"output": response}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/langserve/batch")
async def langserve_batch(request: dict):
    """
    LangServe风格的batch端点 - 批量处理
    """
    tracer = get_langsmith_tracer()
    
    try:
        inputs = request.get("inputs", [])
        config = request.get("config", {})
        session_id = config.get("session_id", "default")
        
        if tracer:
            tracer.start_trace("langserve_batch", {"batch_size": len(inputs)})
        
        chain, _ = create_conversation_chain(session_id, tracer)
        outputs = []
        
        for inp in inputs:
            response = chain.predict(input=inp)
            outputs.append({"output": response})
        
        if tracer:
            tracer.end_trace({"outputs": outputs})
        
        return {"outputs": outputs}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============ 健康检查和根路由 ============

@app.get("/")
async def root():
    """根路由 - 服务信息"""
    return {
        "message": "LangChain Chat Assistant API",
        "version": "2.0",
        "features": [
            "1. LLM调用 (DeepSeek) - 使用ChatOpenAI模型",
            "2. Prompt工程 - SystemMessagePromptTemplate + MessagesPlaceholder",
            "3. Chain链式调用 - LLMChain组合LLM+Prompt+Memory",
            "4. ConversationBufferMemory记忆 - 保持对话上下文",
            "5. Tool工具使用 - get_current_time, calculate, search_knowledge",
            "6. LangSmith监控 - 全流程追踪",
            "7. LangServe部署 - /langserve/invoke, /langserve/batch"
        ],
        "endpoints": {
            "simple_chat": "POST /chat/simple",
            "agent_chat": "POST /chat/agent",
            "memory_chat": "POST /chat/memory",
            "langserve_invoke": "POST /langserve/invoke",
            "langserve_batch": "POST /langserve/batch",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    return {
        "status": "healthy",
        "langchain_version": "0.0.27",
        "langsmith_enabled": langsmith_enabled,
        "features": ["LLM", "Prompt", "Chain", "Memory", "Tool", "LangSmith"]
    }

@app.get("/tools")
async def list_tools():
    """列出可用工具"""
    return {
        "tools": [
            {"name": tool.name, "description": tool.description}
            for tool in langchain_tools
        ]
    }

# ============ 启动服务 ============
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("LangChain Chat Assistant 服务启动中...")
    print("=" * 60)
    print(f"LangSmith监控: {'已启用' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '未启用'}")
    print(f"API文档: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
