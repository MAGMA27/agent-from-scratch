# Agent 实战
不依赖langchain、langraph等框架，基于 [nanobot](https://github.com/HKUDS/nanobot) 的拆解，从零搭建一个agent应用。

## 阶段1-基本功能
实现状态机驱动的 agent 循环、工具调用、provider 抽象，可以在终端界面进行多轮对话。
细节：[基本功能笔记](docs/1_base_frameword.md)

## 阶段2-会话持久化
通过 session manager 维护 session 数据类，将会话数据存储到本地实现持久化，并增加 `_session_locks` 与 `_pending_queues`，管理并发与mid-turn injection。
细节：[会话管理笔记](docs/2_session_management.md)

## 阶段3-上下文窗口管理
由于 llm 本身没有记忆，所以上下文窗口管理在一定程度上就是 llm 的记忆管理。会话持久化保留了所有的对话记录 session，但只增不减的话，有超出上下文窗口限制、llm 性能恶化等风险。最基础的上下文窗口管理，不仅要时刻保持上下文在一定阈值内，给 llm 的回复留出余量，还要将超出阈值的对话记录压缩，避免缺失这部分“记忆”。
细节：[会话管理上下文窗口管理笔记](docs/3_context_window_management.md)

## 阶段4-工具系统
工具系统的细节很多，最基本架构是工具定义（按照格式定义工具）、工具发现（自动发现和加载工具）与工具注册（按照格式注册工具，方便tool call以及运行），需要借助参数 schema 体系做封装抽象。此外，还需要做变量的类型检查，危险操作检查等。
细节：[工具系统](docs/4_tool_system.md)

## 阶段5-skill系统
skills 是热加载，并且渐进式披露的，从 agents 源码路径以及 workspace 路径各维护一个 skills 文件夹，workspace 中的会覆盖内置 skills。 skill 的加载和发现主要依靠 SkillsLoader，它负责扫描路径、动态加载、提取 metadata、输出 summary 等。
细节：[skills 系统](docs/5_skill_system.md)

## 阶段6-Hook
hook 系统是安插在 agent run 各个环节的钩子，它可以获取各个环节的流转信息，并且可以作为 harness 组件，在使用工具等操作前及时检查。
细节：[skills 系统](docs/6_hook.md)

## To do
- 阶段7-subagent
- 阶段8-聊天app接入
- 调试：测试
- 梳理各模块架构，绘制流程图

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)