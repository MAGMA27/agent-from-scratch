# 会话管理

会话持久化是angent的核心之一，参考nanobot的实现，会话资产需要以下核心功能：

## 核心数据模型

dataclass Session，代表单个会话，需要维护以下内容：

```txt
   Session
    ├── key: str              # 唯一标识，格式 channel:chat_id
    ├── messages: list[dict]  # [{role, content, timestamp, tool_calls, ...}, ...]
    ├── created_at / updated_at
    ├── metadata: dict        # 自由扩展区（goal_state, webui 标记, title 等）
    └── last_consolidated: int # 内存合并游标，标记多少条消息已被 Consolidator 处理
```

还需要包括核心方法：添加消息、加载消息、容量控制

## 会话持久化
   
用jsonl格式存储会话数据，需要实现写入与加载方法

> 为什么不选择sqlite？
> 1. 访问模式是"按 key 取整坨"，不是"跨行查询"。没有按时间范围查、没有按 role 筛选、没有聚合统计。这种 "key-value" 式的访问，用一个文件就是一个 session 的方式 恰好匹配。换成 SQLite 反而多了一层无用抽象。
> 2. 追加写入天然匹配 JSONL
> 3. 原子写入比 SQLite 的 fsync 语义更可控
> 4. 损坏恢复极其简单，JSONL 的行独立特性意味着，一行损坏不影响其他行。
> 5. 运维友好，不需要连数据库、不需要写 SQL、不需要管理 schema migration。
> 6. 并发控制已经在上层解决了，JSONL 不支持并发写，但 nanobot 的 AgentLoop 已经做了 per-session asyncio.Lock（loop.py (line 872)），同一时刻只有一个协程操作一个 session。不需要数据库的行锁或事务隔离。

> 什么时候应该用 SQLite 替代？
> SQLite 的优势在于 需要跨 session 查询或者单 session 内的增量操作。如果未来需要这些功能，JSONL 就不够了：
> - "找出所有 session 中最近 7 天的对话数量统计"
> - "根据消息内容反向索引，做语义搜索"
> - "支持消息级别的修改 / 删除"
> 到那一天再切也不晚，毕竟 JSONL → SQLite 的迁移只是一次性的数据转换脚本。

维护一个 SessionManager 类，用于读取、保存会话

## 并发管理

为了应对潜在的并发问题，避免session文件损坏，还需要用asyncio.Lock实现并发安全。

在 AgentCore 中添加 self._session_locks，管理会话的锁，然后在锁内执行状态切换操作。

## mid-turn injection

在 AgentCore 中添加加队列存储和路由入口，判断是否有正在处理的session，如果有，将消息加入队列，如果没有，在 process_message 方法中创建队列，上一条 message 结束后循环消费。只返回最后处理的那条消息。

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)