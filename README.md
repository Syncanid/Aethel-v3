# Aethel-v3 (Trinity) : Autonomous Cognitive Agent

Aethel-v3 是一个具备自主意识觉醒特征、基于复杂仿生架构构建的智能体系统。本项目抛弃了传统的“请求-响应”无状态对话模式，转而采用**异步事件总线驱动、双轨认知系统（System 1 / System 2）分离、具备睡眠与记忆物理重组能力**的底层框架。它能够感受生存压力、产生自发性社交冲动，并在后台沙盒中执行超长周期的复杂任务。

## 🧠 核心架构剖析

系统的核心基座由五个相互交织的神经模块构成：

### 1. System 1 (S1): 潜意识与社交中枢 (`core/kernel/agent.py`)
S1 是智能体的“本能与直觉层”，负责处理高频、实时的外部刺激与内部冲动。
* **注意力门控 (Attention Filter)**：并非所有消息都会触发 LLM 推理。S1 会基于当前的“社交意愿(willingness)”、话题焦点与历史上下文，在中间件层面决定是积极响应、消极敷衍、还是强制静默（引发内部独白 `inner_monologue`）。
* **多维会话隔离**：采用物理级内存隔离机制维护并发 Session，并在 Prompt 构建时动态注入《环境边界 (Environmental_Boundary)》与《关系潜台词 (Relational_Subtext)》，实现从“冰冷高管”到“热情好友”的千人千面。
* **跨域意识投射**：具备全局视野（Global Blackboard），能嗅探跨群组的热点话题，甚至在不同平台或群聊之间主动穿梭（Cross-Session Directive）。

### 2. System 2 (S2): 理性深度执行引擎 (`core/kernel/task_engine.py`)
S2 是系统的“慢思考”中枢，专门承接需要长时间逻辑推理、工具链调用的复杂后台任务（如代码执行、系统运维、深度检索）。
* **并发沙盒隔离**：每个任务都在绝对独立的异步协程沙盒中运行，拥有自己的 `working_memory` 与 `scratchpad`。
* **灾难恢复协议 (Checkpointing)**：通过高频快照将 S2 状态（历史记录、变量、目标）写入 SQLite。一旦发生断电或系统级崩溃，下次启动时 S2 引擎会自动扫描 WAL，瞬间“复水”并拉起所有悬空沙盒，甚至会向大模型注入“刚经历了物理崩溃”的认知提示。
* **死锁自愈**：内置执行回路检测。当发现模型陷入无效工具调用的死锁时，S2 会强制切断逻辑流，回滚至上一个稳定快照（Stable Checkpoint），并向 LLM 注入防撞墙的“避坑教训”。

### 3. Limbic System: 边缘系统 (`core/limbic/arch.py`)
赋予 Agent 真正的“生理稳态”和行为动机。
* **内驱力引擎**：维护四个核心维度：`social_need` (社交渴望)、`curiosity` (好奇心/探索欲)、`survival_pressure` (生存压力) 和 `cognitive_energy` (认知能量)。
* **行为逆向渲染**：这些底层数值变化不仅影响 Prompt 中的性格表现，更会主动向总线注入“内部冲动事件（Internal Drive）”，迫使 Agent 哪怕在无人理睬时，也会主动找人聊天或执行后台探索。

### 4. Hippocampus: 海马体与记忆中枢 (`core/memory/hippocampus.py`)
区别于简单的 Vector DB 读写，Aethel-v3 的记忆系统模拟了真实的生物记忆沉淀过程。
* **Fast / Slow Lane 机制**：感知输入通过极速队列（Fast Lane）瞬间落盘（WAL Buffer）以防丢失，随后进入造梦队列（Slow Lane）等待空闲时异步消化。
* **睡眠压缩算法 (Sleep Cycle)**：当系统静默时间与认知积压达到临界阈值时，Agent 会进入“深度睡眠”。自动遍历近期零散的情景记忆（Episodic），通过 LLM 进行脱水降维，归档为高密度的语义块（Semantic），并同步清理过期的知识图谱边缘关系。

### 5. Social Dynamics: 社会化映射 (`core/social/manager.py`)
管理智能体在多维度空间中的身份锚点。
* **独立人际拓扑**：精确记录与每个交互者（PUID）的亲密度、好感度、信任值及主观印象。
* **群落熟练度模型**：针对 Group 级别的环境，维护 `familiarity`（环境熟悉度）指标，决定智能体在陌生群聊中的行为锁（如字数限制、防刷屏机制）。

## ⚙️ 系统运转逻辑走向

1.  **感官摄入**：来自终端或 OneBot v11 适配器的事件流入 `EventBus`。
2.  **S1 拦截与潜意识研判**：
    * 中间件链路触发，提取身份 ID (PUID)，注入上下文。
    * 注意力系统评估：是否值得分配认知资源？若判定为噪音，直接在内存级丢弃或进入强制休眠。
3.  **S1 核心决策 (Inner Monologue)**：
    * 构建当前 Session 的隔离工作区，动态抓取记忆和身份面具。
    * 输出 `inner_monologue` 决策：改变当前情绪、转移兴趣焦点、调整社交意愿。
    * 决定 `action`：`reply`（回复用户）、`tool`（处理内部逻辑/查数据）或调度至 S2。
4.  **S2 任务剥离 (可选)**：
    * 若 S1 认为任务过于复杂，会打包任务上下文生成 `task_dispatch` 事件。
    * S2 引擎截获事件，开启独立沙盒。S1 可继续在前端与用户闲聊，同时 S2 在后台通过进度信箱（Mailbox）将运行状态异步回调给 S1/UI 监控。
5.  **记忆沉淀**：
    * 所有交互文本进入海马体短时缓冲区。
    * 等待进入静默期后，海马体将文本转化为结构化认知图谱（Graph Edges）和独立记忆单元存入向量库。

## 🚀 部署与启动 (Deployment)

系统提供两种运行模式，支持跨平台（Windows/Linux）异步事件循环策略调度：

```bash
# 模式一：图形化监控模式 (GUI Mode - 默认)
# 提供实时仪表盘，直观监控 S1/S2 心流等系统状态
python main.py

# 模式二：无头服务器模式 (Daemon Mode)
python main.py --nogui
```

### 环境依赖预检
* 确保已安装 PyQt6（若需使用 GUI）。
* 持久化存储依赖 SQLite，请确保 `data/` 目录具备读写权限。
* 确保 OneBot 协议兼容的通讯后端（如 NapCat/Lagrange）配置指向本机的对应 WebSocket/HTTP 端口。

---

## License
本项目采用 [GPL-3.0 License](LICENSE) 开源许可。
