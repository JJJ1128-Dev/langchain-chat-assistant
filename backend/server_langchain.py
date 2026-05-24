"""
LangChain Chat Assistant - 使用真正的LangChain组件实现
包含: LLM调用、Prompt工程、Chain链式调用、Memory记忆、Tool工具使用、LangSmith监控、天气查询
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

# ============ LangChain 核心导入（使用新版API） ============
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langserve import add_routes

# ============ FastAPI 应用初始化 ============
app = FastAPI(
    title="LangChain Chat Assistant",
    description="基于LangChain的对话助手，集成LangSmith监控、工具调用（时间、计算、天气）",
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

# ============ LangSmith 监控类（保留原有功能） ============
class LangSmithTracer:
    def __init__(self, api_key: str, project_name: str = "default"):
        self.api_key = api_key
        self.project_name = project_name
        self.endpoint = "https://api.smith.langchain.com"
        self.run_id = None
    
    def start_trace(self, run_name: str, inputs: dict):
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
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                self.run_id = response.json().get("id")
                return self.run_id
        except Exception as e:
            print(f"LangSmith start_trace error: {e}")
        return None
    
    def end_trace(self, outputs: dict, error: str = None):
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
                json=data,
                timeout=5
            )
        except Exception as e:
            print(f"LangSmith end_trace error: {e}")

def get_langsmith_tracer() -> Optional[LangSmithTracer]:
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key and os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        return LangSmithTracer(
            api_key=api_key,
            project_name=os.getenv("LANGCHAIN_PROJECT", "langchain-chat-assistant")
        )
    return None

# ============ 工具函数定义（新增天气查询） ============
def get_current_time(location: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间"""
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"当前{location}时间: {current_time}"
    except Exception:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

def calculate(expression: str) -> str:
    """计算数学表达式（安全模式）"""
    try:
        allowed = {"abs": abs, "max": max, "min": min, "sum": sum, "pow": pow, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_knowledge(query: str) -> str:
    """模拟知识库搜索"""
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

def get_weather(city: str) -> str:
    """查询实时天气（使用 wttr.in，无需 API Key）"""
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        resp = http_requests.get(url, timeout=5)
        if resp.status_code == 200:
            weather = resp.text.strip()
            return f"{city}天气：{weather}"
        else:
            return f"无法获取{city}天气，请稍后再试。"
    except Exception as e:
        return f"天气查询失败：{str(e)}"

# ============ LangChain 工具配置 ============
langchain_tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="获取指定时区的当前时间。参数 location 可选，默认为 Asia/Shanghai，例如 'Asia/Shanghai'、'America/New_York'。"
    ),
    Tool(
        name="calculate",
        func=calculate,
        description="计算数学表达式。参数 expression 为字符串表达式，如 '2 + 3 * 4'。"
    ),
    Tool(
        name="search_knowledge",
        func=search_knowledge,
        description="搜索知识库获取信息。参数 query 为搜索关键词。"
    ),
    Tool(
        name="get_weather",
        func=get_weather,
        description="查询指定城市的实时天气。参数 city 为城市名，例如 '绍兴'、'北京'。"
    )
]

# ============ LangChain 组件初始化 ============
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    )

# 存储会话记忆
session_memories: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str = "default") -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]

# ============ Prompt工程 ============
SYSTEM_PROMPT = """你是一个智能助手，可以回答各种问题。当用户需要实时信息时，**必须使用工具**。

可用工具：
- get_current_time：获取当前时间（参数 location 可选，如 'Asia/Shanghai'）
- calculate：数学计算
- get_weather：查询天气（参数 city，如 '绍兴'）
- search_knowledge：搜索知识库

规则：
- 用户问“现在几点了”、“当前时间” → 调用 get_current_time，location 默认 'Asia/Shanghai'
- 用户问“XX天气” → 调用 get_weather，参数为城市名
- 用户问数学计算 → 调用 calculate
- 其他问题用自然语言回答

请用中文回复。"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ============ Agent 创建 ============
def create_agent_chain(session_id: str, tracer: LangSmithTracer = None):
    llm = get_llm()
    agent = create_tool_calling_agent(llm, langchain_tools, prompt_template)
    executor = AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        memory=get_memory(session_id),
        verbose=True,
        handle_parsing_errors=True
    )
    return executor, tracer

# ============ API 路由 ============

@app.post("/chat/simple", response_model=ChatResponse)
async def chat_simple(request: ChatRequest):
    """简单对话（不带工具）"""
    tracer = get_langsmith_tracer()
    try:
        if tracer:
            tracer.start_trace("chat_simple", {"message": request.message})
        llm = get_llm()
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的助手。用中文回答。"),
            ("human", "{input}")
        ])
        chain = simple_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": request.message})
        if tracer:
            tracer.end_trace({"response": response})
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/agent", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    """Agent对话（支持工具调用）"""
    tracer = get_langsmith_tracer()
    try:
        if tracer:
            tracer.start_trace("chat_agent", {"message": request.message})
        executor, _ = create_agent_chain(request.session_id, tracer)
        result = executor.invoke({"input": request.message})
        response = result["output"]
        if tracer:
            tracer.end_trace({"response": response})
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/memory")
async def chat_with_memory(chat_history: ChatHistory):
    """带记忆的对话（兼容旧版）"""
    tracer = get_langsmith_tracer()
    try:
        if not chat_history.messages:
            return {"response": "请输入您的问题"}
        last_message = chat_history.messages[-1]
        if last_message.role != "user":
            return {"response": "最后一条消息必须是用户消息"}
        if tracer:
            tracer.start_trace("chat_memory", {"last_message": last_message.content})
        session_id = "session_" + str(hash(str(chat_history.messages)))
        memory = get_memory(session_id)
        for i in range(0, len(chat_history.messages) - 1, 2):
            if i + 1 < len(chat_history.messages):
                user_msg = chat_history.messages[i]
                assistant_msg = chat_history.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    memory.chat_memory.add_user_message(user_msg.content)
                    memory.chat_memory.add_ai_message(assistant_msg.content)
        executor, _ = create_agent_chain(session_id, tracer)
        result = executor.invoke({"input": last_message.content})
        response = result["output"]
        if tracer:
            tracer.end_trace({"response": response})
        return {"response": response}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============ LangServe 风格端点 ============
@app.post("/langserve/invoke")
async def langserve_invoke(request: dict):
    tracer = get_langsmith_tracer()
    try:
        input_data = request.get("input", "")
        config = request.get("config", {})
        session_id = config.get("session_id", "default")
        if tracer:
            tracer.start_trace("langserve_invoke", {"input": input_data})
        executor, _ = create_agent_chain(session_id, tracer)
        result = executor.invoke({"input": input_data})
        if tracer:
            tracer.end_trace({"output": result["output"]})
        return {"output": result["output"]}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/langserve/batch")
async def langserve_batch(request: dict):
    tracer = get_langsmith_tracer()
    try:
        inputs = request.get("inputs", [])
        config = request.get("config", {})
        session_id = config.get("session_id", "default")
        if tracer:
            tracer.start_trace("langserve_batch", {"batch_size": len(inputs)})
        executor, _ = create_agent_chain(session_id, tracer)
        outputs = []
        for inp in inputs:
            result = executor.invoke({"input": inp})
            outputs.append({"output": result["output"]})
        if tracer:
            tracer.end_trace({"outputs": outputs})
        return {"outputs": outputs}
    except Exception as e:
        if tracer:
            tracer.end_trace({}, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============ 健康检查 ============
@app.get("/")
async def root():
    return {
        "message": "LangChain Chat Assistant API",
        "version": "2.0",
        "features": [
            "1. LLM调用 (DeepSeek)",
            "2. Prompt工程",
            "3. Chain链式调用 (Agent)",
            "4. Memory记忆 (ConversationBufferMemory)",
            "5. Tool工具使用 (时间、计算、搜索、天气)",
            "6. LangSmith监控",
            "7. LangServe部署"
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
    return {"status": "healthy"}

@app.get("/tools")
async def list_tools():
    return {"tools": [{"name": t.name, "description": t.description} for t in langchain_tools]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
