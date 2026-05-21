# GitHub 上传和云端部署指南

## 第一步：上传到GitHub

### 1.1 安装Git（如未安装）

访问 https://git-scm.com/download/win 下载并安装Git。

### 1.2 配置Git

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 1.3 初始化并上传代码

在项目根目录执行以下命令：

```bash
# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交更改
git commit -m "Initial commit: LangChain Chat Assistant with LangSmith and LangServe"

# 创建main分支
git branch -M main

# 添加远程仓库（替换为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/langchain-chat-assistant.git

# 推送到GitHub
git push -u origin main
```

### 1.4 在GitHub创建仓库

1. 访问 https://github.com/new
2. 输入仓库名称：`langchain-chat-assistant`
3. 选择公开(Public)或私有(Private)
4. 不要勾选 "Initialize this repository with a README"（已有README）
5. 点击 "Create repository"

## 第二步：Streamlit Cloud 部署

### 2.1 注册Streamlit Cloud

1. 访问 https://streamlit.io/cloud
2. 点击 "Sign in with GitHub"
3. 授权Streamlit访问你的GitHub仓库

### 2.2 部署应用

1. 点击 "New app"
2. 选择仓库：`yourusername/langchain-chat-assistant`
3. 分支：`main`
4. 主文件路径：`streamlit_cloud_app.py`
5. 点击 "Deploy"

### 2.3 配置Secrets

1. 在应用管理页面点击 "⋮" → "Settings"
2. 点击 "Secrets" 标签
3. 添加以下Secrets：

```toml
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_API_KEY = "lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "langchain-chat-assistant"
```

4. 点击 "Save"

### 2.4 等待部署

- 部署通常需要2-5分钟
- 部署完成后会显示访问URL
- 例如：`https://langchain-chat-assistant-xxx.streamlit.app`

## 第三步：LangSmith监控配置

### 3.1 注册LangSmith账号

1. 访问 https://smith.langchain.com
2. 使用GitHub或邮箱注册
3. 创建新项目：`langchain-chat-assistant`

### 3.2 获取API Key

1. 点击右上角头像 → "Settings"
2. 点击 "API Keys" 标签
3. 点击 "Create API Key"
4. 复制生成的API Key

### 3.3 配置项目

在 `.env` 文件或Streamlit Secrets中添加：

```env
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=langchain-chat-assistant
```

## 第四步：验证部署

### 4.1 本地验证

```bash
# 启动后端
cd backend
python server_langchain.py

# 启动前端（新终端）
cd frontend
streamlit run app.py
```

### 4.2 云端验证

1. 访问Streamlit Cloud分配的URL
2. 检查侧边栏显示：
   - ✅ API连接正常
   - ✅ LangSmith监控已启用
3. 发送测试消息验证对话功能

### 4.3 LangSmith监控验证

1. 访问 https://smith.langchain.com
2. 进入 `langchain-chat-assistant` 项目
3. 查看追踪记录是否显示调用信息

## 第五步：更新和维护

### 5.1 代码更新

```bash
# 修改代码后
git add .
git commit -m "Update: xxx功能"
git push origin main
```

### 5.2 重新部署

Streamlit Cloud会自动检测GitHub推送并重新部署。

## 常见问题

### Q1: 部署后显示API未配置？
**A**: 检查Streamlit Cloud的Secrets是否正确设置了`DEEPSEEK_API_KEY`

### Q2: LangSmith没有追踪记录？
**A**: 确认`LANGCHAIN_TRACING_V2=true`且`LANGCHAIN_API_KEY`正确

### Q3: 如何切换为OpenAI？
**A**: 修改代码中的`openai_api_base`为OpenAI的地址，并设置`OPENAI_API_KEY`

### Q4: 前后端分离如何部署？
**A**: 
- 后端：部署到Heroku、Railway、Render等平台
- 前端：Streamlit Cloud部署
- 修改前端`API_BASE_URL`为后端地址

## 项目文件说明

```
├── backend/server_langchain.py    # 完整LangChain后端（前后端分离用）
├── streamlit_cloud_app.py         # 合并版（Streamlit Cloud用）
├── requirements-cloud.txt         # 云端部署依赖
├── .streamlit/config.toml         # Streamlit配置
└── README.md                      # 项目说明
```

## 技术栈总结

| 组件 | 技术 | 版本 |
|------|------|------|
| LLM框架 | LangChain | 0.0.27 |
| 后端框架 | FastAPI | 0.99.1 |
| 前端框架 | Streamlit | 1.23.1 |
| LLM模型 | DeepSeek | deepseek-chat |
| 监控 | LangSmith | 集成 |
| 部署 | Streamlit Cloud | 免费 |

## 联系支持

- GitHub Issues: 提交问题
- Streamlit论坛: https://discuss.streamlit.io
- LangChain文档: https://python.langchain.com
