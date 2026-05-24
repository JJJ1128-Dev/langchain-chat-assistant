"""
LangChain Chat Assistant - 最终稳定版（时间/天气中文显示）
兼容 LangChain 1.x，满足作业全部要求
"""

import streamlit as st
import os
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

# ============ 工具函数（中文版） ============
def get_current_time_chinese(location: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间，返回中文格式"""
    try:
        tz = pytz.timezone(location)
        current_time = datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S")
        # 将时区名称转为中文常用表达
        if location == "Asia/Shanghai":
            tz_name = "北京时间"
        elif location == "America/New_York":
            tz_name = "纽约时间"
        else:
            tz_name = location.replace("_", " ")
        return f"当前{tz_name}: {current_time}"
    except Exception:
        utc_now = datetime.utcnow().strftime("%Y年%m月%d日 %H:%M:%S")
        return f"当前UTC时间: {utc_now}"

def calculate(expression: str) -> str:
    """计算数学表达式（安全模式）"""
    try:
        allowed = {"abs": abs, "max": max, "min": min, "sum": sum, "pow": pow, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

def search_knowledge(query: str) -> str:
    """模拟知识库搜索"""
    return f"知识库搜索结果: 关于'{query}'的相关信息..."

def get_weather_chinese(city: str = "绍兴") -> str:
    """查询实时天气，返回中文描述（优先真实API，失败时模拟）"""
    try:
        # 使用 lang=zh 参数获取中文天气，%C 天气描述，%t 温度
        url = f"https://wttr.in/{city}?format=%C+%t&lang=zh"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.text.strip():
            weather_raw = resp.text.strip()
            # 去除可能的前导加号（温度）
            weather_clean = weather_raw.replace('+', '')
            return f"{city}天气：{weather_clean}"
        else:
            # 模拟数据回退
            return f"{city}天气：晴，22°C（演示数据，网络限制）"
    except Exception:
        return f"{city}天气：多云，20°C（演示数据，API不可达）"

def use_tool(user_input: str):
    """根据输入决定是否调用工具，返回中文结果或 None"""
    lower = user_input.lower()
    # 时间匹配
    if any(word in lower for word in ["时间", "现在几点", "当前时间", "几点钟", "北京时间", "纽约时间", "UTC"]):
        if "北京" in lower or "上海" in lower:
            return get_current_time_chinese("Asia/Shanghai")
        elif "纽约" in lower or "美国" in lower:
            return get_current_time_chinese("America/New_York")
        else:
            return get_current_time_chinese("Asia/Shanghai")
    # 计算匹配
    if any(word in lower for word in ["计算", "等于", "+", "-", "*", "/", "平方", "根号"]):
        expr_match = re.search(r'[\d+\-*/().]+', user_input)
        if expr_match:
            return calculate(expr_match.group())
        else:
            return calculate(user_input)
    # 天气匹配
    if any(word in lower for word in ["天气", "气温", "温度", "下雨", "晴天", "多云", "阴"]):
        city_match = re.search(r'([\u4e00-\u9fa5]{2,})天气', user_input)
        city = city_match.group(1) if city_match else "绍兴"
        return get_weather_chinese(city)
    # 搜索匹配
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
    """构建 LCEL 链（Prompt + LLM + 输出解析）"""
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
    """将消息列表转换为 LangChain 消息格式"""
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
            st.info("请在 Streamlit Cloud Secrets 中配置")
        st.divider()
        st.session_state.use_tool_mode = st.toggle("🔧 启用工具模式", value=st.session_state.use_tool_mode)
        if st.button("🗑️ 清空对话"):
            st.session_state.messages.clear()
            st.rerun()
        st.divider()
        st.markdown("""
        **核心功能（满足作业要求）**
        1. **LLM调用** - DeepSeek 大模型
        2. **Prompt工程** - ChatPromptTemplate + MessagesPlaceholder
        3. **Chain链式调用** - LCEL (`prompt | llm | parser`)
        4. **Memory记忆** - 手动维护消息列表，传递历史
        5. **Tool工具使用** - 时间、计算、天气、搜索（全部中文输出）
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
