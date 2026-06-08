# hook
hook 系统是安插在 agent run 各个环节的钩子，它可以获取各个环节的流转信息，并且可以作为 harness 组件，在使用工具等操作前及时检查。

## 实现
在抽象基类 `AgentHook` 中，分别设定 `run` 级、`iteration` 级以及 `streaming` 级别的方法，后续在子类中继承实现即可。

值得一提的是，为了处理多个 `hook` 在一次对话中的实现，定义 `AgentHook` 的子类 `CompositeHook`，接收 `list[AgentHook]`，在指定环节依次调用各hook。

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)