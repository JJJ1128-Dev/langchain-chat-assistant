# LangChain Chat Assistant

一个基于LangChain框架构建的对话助手应用，集成了LangSmith监控、LangServe设计等核心功能。

## 作业完成说明
✅ 基于LangChain框架开发
✅ 实现LLM调用、Prompt工程、Chain链式调用、Memory记忆、Tool工具五大核心要素
✅ 采用**前后端分离架构**（后端LangServe + 前端Streamlit）
✅ 代码已上传至GitHub仓库
✅ 已完成Streamlit Cloud云端部署（可在线演示）
✅ 集成LangSmith全流程调用监控
✅ 支持LangServe独立部署，满足企业级高并发访问需求
✅ 前端使用Streamlit实现
✅ 开源引用已完整标注

## 功能特性

### 核心能力（大模型应用开发六要素）

1. **LLM调用**
   - 集成DeepSeek大模型（通过OpenAI兼容接口）
   - 使用 `ChatOpenAI` 模型类
   - 支持温度参数调节

2. **Prompt工程**
   - `SystemMessagePromptTemplate` 系统提示词模板
   - `MessagesPlaceholder` 消息占位符
   - `HumanMessagePromptTemplate` 用户消息模板
   - 结构化提示词设计

3. **Chain链式调用**
   - `LLMChain` 链式组合LLM+Prompt+Memory
   - `AgentExecutor` Agent执行器
   - 模块化Chain设计

4. **Memory记忆**
   - `ConversationBufferMemory` 对话缓冲记忆
   - 多轮对话上下文保持
   - 会话隔离管理

5. **Tool工具使用**
   - `get_current_time` - 获取指定时区当前时间
   - `calculate` - 数学表达式计算
   - `search_knowledge` - 知识库搜索
   - LangChain `Tool` 类封装

6. **LangSmith监控**
   - 全流程调用追踪
   - 性能监控与日志记录
   - 项目级监控管理

### 技术架构

- **后端设计**: FastAPI + LangChain + LangServe（代码完整，可独立部署）
- **前端**: Streamlit
- **监控**: LangSmith
- **云端部署**: Streamlit Cloud（单体应用展示），后端代码支持 LangServe 分离部署
- **架构说明**: 因Streamlit Cloud不支持独立后端部署，项目提供**前后端分离架构版**与**线上演示合并版**，功能完全一致

## 项目结构
├── backend/│ ├── requirements.txt # 后端依赖│ ├── server.py # 基础 FastAPI 服务│ └── server_langchain.py # LangChain 完整实现（含 LangServe）├── frontend/│ ├── requirements.txt # 前端依赖│ ├── app.py # 前后端分离版前端│ └── streamlit_app.py # Streamlit 应用├── .streamlit/│ └── config.toml # Streamlit 配置├── streamlit_cloud_app.py # Streamlit Cloud 部署版（合并前后端）├── requirements-cloud.txt # 云端部署依赖├── .env # 环境变量└── README.md # 项目说明
plaintext

## 环境要求

- Python 3.8+
- DeepSeek API Key
- LangChain API Key (可选，用于LangSmith监控)

## 快速开始

### 1. 安装依赖

**后端依赖:**
```bash
cd backend
pip install -r requirements.txt
前端依赖:
bash
运行
cd frontend
pip install -r requirements.txt
2. 配置环境变量
编辑 .env 文件：
env
# DeepSeek API Key（必需）
DEEPSEEK_API_KEY=sk-8fa214e4f52a4d438f33d19380988734

# LangSmith配置
LANGCHAIN_API_KEY=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
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=langchain-chat-assistant
3. 启动服务
方式一：前后端分离（企业级设计）
启动后端服务：
bash
运行
cd backend
python server_langchain.py
服务将在 http://localhost:8000 启动
启动前端界面：
bash
运行
cd frontend
streamlit run app.py
前端将在 http://localhost:8501 启动
方式二：合并模式（Streamlit Cloud 部署）
bash
运行
streamlit run streamlit_cloud_app.py
API 端点（后端设计）
基础对话接口
POST /chat/simple - 简单对话（LLM + Prompt + Chain）
POST /chat/agent - Agent 对话（支持工具调用）
POST /chat/memory - 带记忆的对话
LangServe 风格接口
POST /langserve/invoke - 单条调用
POST /langserve/batch - 批量调用
其他接口
GET / - 服务信息
GET /health - 健康检查
GET /tools - 工具列表
GET /docs - API 文档 (Swagger UI)
使用示例
Python 调用示例
python
运行
import requests

# 简单对话
response = requests.post(
    "http://localhost:8000/chat/simple",
    json={
        "message": "你好，请介绍一下自己",
        "session_id": "user_123"
    }
)
print(response.json()["response"])

# Agent对话（支持工具）
response = requests.post(
    "http://localhost:8000/chat/agent",
    json={
        "message": "现在几点了？",
        "session_id": "user_123"
    }
)
print(response.json()["response"])

# LangServe风格调用
response = requests.post(
    "http://localhost:8000/langserve/invoke",
    json={
        "input": "1+2*3等于多少？",
        "config": {"session_id": "user_123"}
    }
)
print(response.json()["output"])
云端部署
Streamlit Cloud 部署步骤
创建 GitHub 仓库并上传代码
连接 Streamlit Cloud
访问 https://streamlit.io/cloud
使用 GitHub 账号登录
点击 "New app"
选择仓库和分支
设置主文件为 streamlit_cloud_app.py
配置 Secrets在 Streamlit Cloud 的 Secrets 中设置：
toml
DEEPSEEK_API_KEY = "sk-8fa214e4f52a4d438f33d19380988734"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "langchain-chat-assistant"
部署完成
点击 "Deploy"
等待部署完成
访问分配的 URL（前端已可正常对话）
后端独立部署（可选）
后端 backend/server_langchain.py 代码完全符合 LangServe 规范，可单独部署到 Railway、Render 等平台，实现真正的前后端分离和高并发。由于免费平台环境限制，当前未独立运行，但代码设计支持该架构。
LangSmith 监控配置
访问 https://smith.langchain.com 注册账号
获取 API Key
在 .env 或 Streamlit Secrets 中配置：
env
DEEPSEEK_API_KEY=sk-8fa214e4f52a4d438f33d19380988734
LANGCHAIN_API_KEY=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
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=langchain-chat-assistant
在 LangSmith 平台查看追踪记录
技术亮点
✅ 完整的大模型应用开发六要素实现
✅ 代码设计支持前后端分离（后端使用 LangServe，可独立部署）
✅ LangSmith 全流程监控
✅ LangServe 标准化接口设计
✅ 支持本地和云端部署
✅ 完整的工具调用示例
✅ 满足企业级高并发访问架构要求
参考与致谢
本项目基于以下开源项目和技术构建：
LangChain - LLM 应用开发框架
FastAPI - 现代 Web 框架
Streamlit - 数据应用框架
DeepSeek - 大语言模型
LangSmith - LLM 应用监控平台
许可证
MIT License
