from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
from datetime import datetime
import pytz
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="LangChain Chat Assistant", version="1.0")

class ChatMessage(BaseModel):
    role: str = Field(description="消息角色: user 或 assistant")
    content: str = Field(description="消息内容")

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field(description="对话历史")

def get_current_time(location: str = "UTC") -> str:
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"当前{location}时间: {current_time}"
    except Exception as e:
        return f"当前UTC时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

tools = {
    "get_current_time": get_current_time,
    "calculate": calculate
}

def build_prompt(messages: List[ChatMessage]) -> str:
    system_prompt = """你是一个聪明、友好、知识渊博的对话助手，能够回答各种问题、提供帮助。

你拥有以下工具可以使用：
1. get_current_time - 获取指定时区的当前时间
2. calculate - 计算复杂数学表达式

**使用工具的规则：**
- 当用户询问时间相关问题时，使用 get_current_time 工具
- 当用户询问数学计算问题时，使用 calculate 工具
- 工具调用格式：{"tool_call": {"name": "工具名称", "args": {"参数名": "参数值"}}}

**直接回答的规则：**
- 对于一般性问题、闲聊、知识问答等，直接用自然语言回答
- 不要过度依赖工具，只有在确实需要时才调用
- 如果无法回答或不确定，可以礼貌地说明

请用中文进行友好、自然的回复。"""
    
    prompt = system_prompt + "\n\n"
    
    for msg in messages:
        if msg.role == "user":
            prompt += f"用户: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"助手: {msg.content}\n"
        elif msg.role == "tool":
            prompt += f"工具结果: {msg.content}\n"
    
    prompt += "助手:"
    return prompt

def parse_tool_call(response_text: str) -> Optional[Dict]:
    try:
        response_text = response_text.strip()
        if response_text.startswith("{") and response_text.endswith("}"):
            data = json.loads(response_text)
            if "tool_call" in data:
                return data["tool_call"]
    except:
        pass
    return None

def generate_response(messages: List[ChatMessage], api_key: str) -> str:
    prompt = build_prompt(messages)
    
    import openai
    openai.api_key = api_key
    openai.api_base = "https://api.deepseek.com/v1"
    
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        tool_call = parse_tool_call(content)
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("args", {})
            
            if tool_name in tools:
                tool_result = tools[tool_name](**args)
                
                new_messages = messages.copy()
                new_messages.append(ChatMessage(role="assistant", content=content))
                new_messages.append(ChatMessage(role="tool", content=tool_result))
                
                return generate_response(new_messages, api_key)
        
        return content
    except Exception as e:
        return f"API调用错误: {str(e)}"

@app.post("/chat/memory")
async def chat_with_memory(chat_history: ChatHistory):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未设置")
    
    if not chat_history.messages:
        return {"response": "请输入您的问题"}
    
    last_message = chat_history.messages[-1]
    if last_message.role != "user":
        return {"response": "最后一条消息必须是用户消息"}
    
    response = generate_response(chat_history.messages, api_key)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "LangChain Chat Assistant API (DeepSeek)"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
