from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langserve import add_routes
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="LangChain Chat Assistant", version="1.0")

@tool
def get_current_time(location: str = "UTC") -> str:
    """获取当前时间"""
    from datetime import datetime
    import pytz
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"当前{location}时间: {current_time}"
    except:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和括号"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

class ChatHistory(BaseModel):
    messages: List[dict] = Field(description="对话历史，包含role和content字段")

def create_conversation_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    memory = ConversationBufferMemory(return_messages=True)
    
    tools = [get_current_time, calculate]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的对话助手，能够回答问题、提供帮助。使用工具时请用中文回复。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    return agent_executor

conversation_chain = create_conversation_chain()

add_routes(
    app,
    conversation_chain,
    path="/chat",
    enabled_endpoints=["invoke", "batch", "stream"],
)

@app.post("/chat/memory")
async def chat_with_memory(chat_history: ChatHistory):
    messages = []
    for msg in chat_history.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    conversation_chain.memory.chat_memory.messages = messages
    
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            response = conversation_chain.invoke({"input": last_message.content})
            return {"response": response["output"]}
    
    return {"response": "请输入您的问题"}

@app.get("/")
async def root():
    return {"message": "LangChain Chat Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)