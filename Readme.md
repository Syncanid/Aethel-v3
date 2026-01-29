# Aethel v3 (Project Code: Trinity)

Aethel v3 是一个基于 Python `asyncio` 构建的**自主 AI 智能体框架**。它不仅仅是一个聊天机器人，而是一个具备长期记忆整理能力、自主思考循环、以及通过
**MCP (Model Context Protocol)** 进行工具扩展的“数字生命”原型。

核心设计目标是实现一个**事件驱动**、**高度模块化**且具备**自我进化能力**的智能实体。

## ✨ 核心特性

* **🧠 自主认知内核**
    * **无限自主循环**: 系统不依赖用户输入触发，具备独立的思考和行动周期。
    * **思维链 (Chain of Thought)**: 内置 `think` 和 `advanced_think` 工具，支持规划、反思和结构化推理。
    * **时间感知**: 能够感知系统启动时间和相对时间流逝，具备 Cron 级别的任务调度能力。

* **💾 海马体记忆系统**
    * **仿生记忆架构**: 区分**核心记忆** (Core, 用户画像)、**情景记忆** (Episodic, 经历事件) 和 **语义记忆** (Semantic,
      事实知识)。
    * **自动归档**: 后台进程 (`Hippocampus`) 会自动监控对话流，触发 LLM 提炼关键信息并存入向量数据库 (ChromaDB) 和
      SQLite。
    * **知识摄入**: 支持读取本地文件，自动分块并转化为知识条目。

* **🛠️ 混合工具生态**
    * **MCP 支持**: 原生支持 **Model Context Protocol**，可连接外部 MCP Server（如文件系统、GitHub 等）。
    * **Python 代码解释器**: 拥有一个受限的沙箱环境，可以编写和执行 Python 代码来解决复杂计算或逻辑问题。
    * **联网搜索**: 集成 SearXNG，具备获取实时信息的能力。

* **🔌 事件总线架构**
    * 基于 OneBot v12 标准的事件定义，解耦感知（Adapters）与决策（Agent）。
    * 目前支持 Console 交互，预留 OneBot (QQ/Telegram) 适配接口。

## 🏗️ 系统架构

```mermaid
graph TD
    User[用户/外部世界] -->|Input| Adapter[适配器层 (Console/OneBot)]
    Adapter -->|Event| Bus[神经事件总线 (EventBus)]
    
    subgraph "Aethel Core (Trinity)"
        Bus --> Agent[自主智能体 (Agent)]
        Agent <-->|Read/Write| Scratchpad[思维暂存区]
        
        Agent -->|Call| ToolMgr[工具管理器]
        ToolMgr -->|Local| PyTools[Python Tools]
        ToolMgr -->|Remote| MCP[MCP Clients]
        
        Agent <-->|RAG| VectorStore[向量数据库 (Chroma)]
        
        subgraph "Memory Consolidation"
            Hippo[海马体 (Hippocampus)] -- 异步扫描 --> ChatLogs[(聊天记录)]
            Hippo -->|提炼| VectorStore
            Hippo -->|提炼| CoreDB[(SQLite)]
        end
    end
```

## 🚀 快速开始

### 环境要求

* Python 3.10+
* Windows / Linux / macOS

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/furryaxw/Aethel-v3.git
cd Aethel-v3

```

2. **安装依赖**

```bash
pip install -r requirements.txt

```

3. **配置文件**
   复制默认配置模板：

```bash
cp data/config.yaml.default data/config.yaml

```

编辑 `data/config.yaml`，填入你的 LLM API Key：

```yaml
llm:
  api_base_url: "https://api.openai.com/v1" # 或其他兼容接口
  api_key: "sk-xxxxxx"
  model_name: "gpt-4o"
  embedding_model_name: "text-embedding-ada-002"

```

### 运行

启动系统主程序：

```bash
python main.py

```

启动后，您将在控制台看到系统初始化日志。在 `Console` 适配器提示后，可以直接输入文字与 Aethel 进行对话。

## 📂 目录结构

* `core/`
* `kernel/`: 核心逻辑 (Agent, Prompt)
* `memory/`: 记忆系统 (海马体, 向量存储, 摄入器)
* `infrastructure/`: 基础设施 (DB, Logger, Config)
* `io/`: 输入输出 (EventBus, Adapters)
* `tool_manager/`: 工具加载与 MCP 客户端


* `tools/`: 本地工具集 (Network, Code, Scheduler 等)
* `data/`: 存放配置文件、日志、SQLite 数据库和向量库文件

## 🛠️ 配置说明

在 `data/config.yaml` 中：

* **System**: 设置日志级别 (`log_level`).
* **LLM**: 配置大模型接口。
* **MCP Servers**: 配置外部 MCP 服务（如文件系统操作）。

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: [ "-y", "@modelcontextprotocol/server-filesystem", "./data" ]

```

## 📄 License

本项目采用 [GPL-3.0 License](LICENSE) 开源许可。
