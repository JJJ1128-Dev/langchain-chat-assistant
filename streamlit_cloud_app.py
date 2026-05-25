"""
LangChain Chat Assistant - 完整版（时间/天气中文，高德API真实天气）
集成 LangSmith 监控
兼容 LangChain 1.x，满足作业全部要求
"""

# ============ LangSmith 监控配置（必须在任何 LangChain 导入之前） ============
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # 请替换为您的实际新密钥
os.environ["LANGCHAIN_PROJECT"] = "langchain-chat-assistant"

import streamlit as st
import re
import requests
from datetime import datetime
import pytz
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

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

# ============ 工具函数（中文，真实天气） ============
def get_current_time_chinese(location: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间，中文格式"""
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S")
        if location == "Asia/Shanghai":
            return f"当前北京时间: {current_time}"
        elif location == "America/New_York":
            return f"当前纽约时间: {current_time}"
        else:
            return f"当前{location}时间: {current_time}"
    except Exception:
        utc_now = datetime.utcnow().strftime("%Y年%m月%d日 %H:%M:%S")
        return f"当前UTC时间: {utc_now}"

def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = {"abs": abs, "max": max, "min": min, "sum": sum, "pow": pow, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

def search_knowledge(query: str) -> str:
    """模拟知识库搜索"""
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

def get_weather_real(city: str = "绍兴") -> str:
    """
    使用高德地图天气API获取实时天气（中文真实数据）
    需要环境变量 AMAP_API_KEY
    """
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        return "天气服务未配置：请设置高德API密钥（AMAP_API_KEY）"

    try:
        # 1. 根据城市名获取adcode
        geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&output=json&key={api_key}"
        geo_resp = requests.get(geo_url, timeout=5)
        if geo_resp.status_code != 200:
            return f"无法查询{city}天气（地理编码失败）"
        geo_data = geo_resp.json()
        if geo_data.get("status") != "1" or not geo_data.get("geocodes"):
            return f"未找到城市：{city}"
        adcode = geo_data["geocodes"][0]["adcode"]

        # 2. 获取实时天气
        weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={api_key}"
        weather_resp = requests.get(weather_url, timeout=5)
        if weather_resp.status_code != 200:
            return f"无法获取{city}天气"
        weather_data = weather_resp.json()
        if weather_data.get("status") != "1" or not weather_data.get("lives"):
            return f"天气数据异常"
        live = weather_data["lives"][0]
        weather = live["weather"]          # 天气现象（中文）
        temperature = live["temperature"]  # 温度（摄氏度）
        return f"{city}天气：{weather}，{temperature}℃"
    except Exception as e:
        return f"天气查询失败：{str(e)}"

def use_tool(user_input: str):
    """根据输入决定调用哪个工具"""
    lower = user_input.lower()
    # 时间
    if any(word in lower for word in ["时间", "现在几点", "当前时间", "几点钟", "北京时间", "纽约时间"]):
        if "北京" in lower or "上海" in lower:
            return get_current_time_chinese("Asia/Shanghai")
        elif "纽约" in lower or "美国" in lower:
            return get_current_time_chinese("America/New_York")
        else:
            return get_current_time_chinese("Asia/Shanghai")
    # 计算
    if any(word in lower for word in ["计算", "等于", "+", "-", "*", "/", "平方", "根号"]):
        expr_match = re.search(r'[\d+\-*/().]+', user_input)
        expr = expr_match.group() if expr_match else user_input
        return calculate(expr)
    # 天气
    if any(word in lower for word in ["天气", "气温", "温度", "下雨", "晴天", "多云", "阴"]):
        city_match = re.search(r'([\u4e00-\u9fa5]{2,})天气', user_input)
        city = city_match.group(1) if city_match else "绍兴"
        return get_weather_real(city)
    # 搜索
    if any(word in lower for word in ["搜索", "查找", "什么是"]):
        return search_knowledge(user_input)
    return None

# ============ 状态管理 ============
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_tool_mode" not in st.session_state:
        st.session_state.use_tool_mode = True   # 默认启用工具模式
    if "llm" not in st.session_state:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            st.session_state.llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0.7
            )
        else:
            st.session_state.llm = None

def build_chain():
    """构建 LCEL 链"""
    llm = st.session_state.llm
    if not llm:
        return None
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。请用中文回答。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()

def get_chat_history_from_messages(messages: List[Dict]) -> List:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# ============ 主界面 ============
def main():
    init_session_state()
    st.markdown('<h1 class="main-header">🤖 LangChain 对话助手</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ 配置")
        if st.session_state.llm:
            st.success("✅ DeepSeek API 正常")
        else:
            st.error("❌ 未设置 DEEPSEEK_API_KEY")
            st.info("请在 Secrets 中配置 DEEPSEEK_API_KEY")
        st.divider()
        st.session_state.use_tool_mode = st.toggle("🔧 启用工具模式", value=st.session_state.use_tool_mode)
        if st.button("🗑️ 清空对话"):
            st.session_state.messages.clear()
            st.rerun()
        st.divider()
        st.markdown("""
        **核心功能（满足作业要求）**
        1. **LLM调用** - DeepSeek 大模型
        2. **Prompt工程** - ChatPromptTemplate
        3. **Chain链式调用** - LCEL
        4. **Memory记忆** - 消息列表
        5. **Tool工具使用** - 时间、计算、天气（高德API）
        """)

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
                    if st.session_state.use_tool_mode:
                        tool_result = use_tool(user_input)
                        if tool_result:
                            response = tool_result
                        else:
                            chain = build_chain()
                            if chain:
                                history = get_chat_history_from_messages(st.session_state.messages[:-1])
                                response = chain.invoke({"history": history, "input": user_input})
                            else:
                                response = "❌ API 未配置"
                    else:
                        chain = build_chain()
                        if chain:
                            history = get_chat_history_from_messages(st.session_state.messages[:-1])
                            response = chain.invoke({"history": history, "input": user_input})
                        else:
                            response = "❌ API 未配置"

                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"错误: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"错误: {e}"})

if __name__ == "__main__":
    main()
