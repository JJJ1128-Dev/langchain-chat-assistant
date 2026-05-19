# LangChain Chat Assistant

一个基于LangChain框架构建的对话助手应用，集成了LangSmith监控、LangServe部署等核心功能。

## 功能特性

### 核心能力
- **LLM调用**: 集成OpenAI GPT-3.5 Turbo模型
- **Prompt工程**: 优化的对话提示词设计
- **Chain链式调用**: 构建复杂的对话流程
- **Memory记忆**: ConversationBufferMemory保持对话上下文
- **Tool工具使用**: 支持时间查询和数学计算工具

### 技术架构
- **后端**: LangServe + FastAPI，支持高并发访问
- **前端**: Streamlit交互式界面
- **监控**: LangSmith全链路追踪

## 项目结构

```
├── backend/
│   ├── requirements.txt    # 后端依赖
│   └── server.py          # LangServe服务端代码
├── frontend/
│   ├── requirements.txt    # 前端依赖
│   ├── app.py             # Streamlit前端代码
│   └── .env               # 前端环境变量
├── .env                   # 项目环境变量
└── README.md              # 项目说明文档
```

## 环境要求

- Python 3.10+
- OpenAI API Key
- LangChain API Key (可选，用于LangSmith监控)

## 快速开始

### 1. 安装依赖

**后端依赖:**
```bash
cd backend
pip install -r requirements.txt
```

**前端依赖:**
```bash
cd frontend
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入您的API密钥：

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=langchain-chat-assistant
```

### 3. 启动服务

**启动后端服务:**
```bash
cd backend
python server.py
```

服务将在 http://localhost:8000 启动

**启动前端界面:**
```bash
cd frontend
streamlit run app.py
```

前端将在 http://localhost:8501 启动

## API端点

- `POST /chat/invoke` - 调用对话链
- `POST /chat/batch` - 批量调用
- `POST /chat/stream` - 流式响应
- `POST /chat/memory` - 带记忆的对话

## 使用示例

```python
import requests

# 发送对话请求
response = requests.post(
    "http://localhost:8000/chat/memory",
    json={
        "messages": [
            {"role": "user", "content": "你好！"},
            {"role": "assistant", "content": "你好！我是你的LangChain对话助手。"},
            {"role": "user", "content": "现在几点了？"}
        ]
    }
)

print(response.json())
```

## 支持的工具

1. **时间查询**: 获取指定时区的当前时间
   - 示例: "现在北京几点了？"

2. **数学计算**: 支持加减乘除和括号运算
   - 示例: "计算 2*(3+4)-5"

## 部署

### Streamlit Cloud部署

1. Fork本项目到您的GitHub仓库
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 连接您的GitHub仓库
4. 设置部署目录为 `frontend`
5. 添加环境变量: `API_BASE_URL`

### LangServe部署

后端服务可以部署到任何支持FastAPI的平台：
- Vercel
- Render
- AWS EC2
- Docker容器

## 参考资源

- [LangChain官方文档](https://python.langchain.com/docs/)
- [LangServe文档](https://python.langchain.com/docs/langserve/)
- [LangSmith文档](https://docs.smith.langchain.com/)

## License

MIT License

## 致谢

本项目基于LangChain官方模板和示例代码构建。