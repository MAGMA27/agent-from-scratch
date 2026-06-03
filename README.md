# agent 实战
不依赖langchain、langraph等框架，从零搭建一个agent应用。

## 阶段1-基本功能
实现状态机驱动的 agent 循环、工具调用、provider 抽象，可以在终端界面进行多轮对话。
细节：[基本功能笔记](docs/1_base_frameword.md)

## 阶段2-会话持久化
通过 session manager 维护 session 数据类，将会话数据存储到本地实现持久化，并增加 `_session_locks` 与 `_pending_queues`，管理并发与mid-turn injection。
细节：[会话管理笔记](docs/2_session_management.md)

## To do
- 阶段3-上下文窗口管理
- 阶段4-skill系统
- 阶段5-工具系统
- 阶段6-聊天app接入
- 调试：结构化日志、测试
- 交互：streaming支持、CLI

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)