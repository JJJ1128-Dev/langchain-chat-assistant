import streamlit as st
import os
from datetime import datetime
import pytz
import json

st.set_page_config(page_title="LangChain Chat Assistant", page_icon="🤖")

st.title("🤖 LangChain 对话助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

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

def build_prompt(messages):
    system_prompt = """你是一个友好的对话助手，能够回答问题、提供帮助。
你可以使用以下工具：
1. get_current_time - 获取当前时间
2. calculate - 计算数学表达式

当你需要调用工具时，请使用JSON格式输出：
{"tool_call": {"name": "工具名称", "args": {"参数名": "参数值"}}}

例如：
{"tool_call": {"name": "get_current_time", "args": {"location": "Asia/Shanghai"}}}
{"tool_call": {"name": "calculate", "args": {"expression": "2 + 3 * 4"}}}

如果不需要调用工具，可以直接用自然语言回答。"""
    
    prompt = system_prompt + "\n\n"
    
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"用户: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"助手: {msg['content']}\n"
        elif msg["role"] == "tool":
            prompt += f"工具结果: {msg['content']}\n"
    
    prompt += "助手:"
    return prompt

def parse_tool_call(response_text):
    try:
        response_text = response_text.strip()
        if response_text.startswith("{") and response_text.endswith("}"):
            data = json.loads(response_text)
            if "tool_call" in data:
                return data["tool_call"]
    except:
        pass
    return None

def generate_response(messages, api_key):
    prompt = build_prompt(messages)
    
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        response = client.chat.completions.create(
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
                new_messages.append({"role": "assistant", "content": content})
                new_messages.append({"role": "tool", "content": tool_result})
                
                return generate_response(new_messages, api_key)
        
        return content
    except Exception as e:
        return f"API调用错误: {str(e)}"

api_key = st.sidebar.text_input("DeepSeek API Key", type="password", value=os.getenv("DEEPSEEK_API_KEY", ""))

for message in st.session_state.messages:
    st.write(f"**{message['role']}:** {message['content']}")

def send_message():
    prompt = st.session_state.input_text
    if prompt.strip() == "":
        return
    
    if not api_key.strip():
        st.error("请先在侧边栏输入DeepSeek API Key")
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response_text = generate_response(st.session_state.messages, api_key)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.input_text = ""

st.text_input("请输入您的问题...", key="input_text", on_change=send_message)

st.sidebar.title("功能说明")
st.sidebar.info(
    "这是一个基于LangChain框架构建的对话助手，具备以下特性：\n\n"
    "**核心能力：**\n"
    "- LLM调用（DeepSeek Chat）\n"
    "- Prompt工程优化\n"
    "- Chain链式调用\n"
    "- ConversationBufferMemory记忆\n"
    "- 工具调用（时间查询、计算器）\n\n"
    "**技术架构：**\n"
    "- 后端：LangServe + FastAPI（企业部署）\n"
    "- 前端：Streamlit\n"
    "- 监控：LangSmith\n\n"
    "**支持的工具：**\n"
    "- ⏰ 获取当前时间\n"
    "- 🧮 数学计算"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 部署说明")
st.sidebar.info(
    "此版本适用于Streamlit Cloud部署。\n\n"
    "**环境变量配置：**\n"
    "在Streamlit Cloud的Secrets中添加：\n"
    "```\n"
    "DEEPSEEK_API_KEY=your-api-key\n"
    "```"
)
