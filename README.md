# agent 实战
不依赖langchain、langraph等框架，从零搭建一个agent应用。

## 阶段1-基本功能
实现状态机驱动的 agent 循环、工具调用、provider 抽象，可以在终端界面进行多轮对话。
细节：[基本功能笔记](docs/1_base_frameword.md)

## To do
- 阶段2-会话持久化
- 阶段3-上下文窗口管理
- 阶段4-工具系统
- 阶段5-聊天app接入
- 调试：结构化日志、测试
- 交互：streaming支持、CLI

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)