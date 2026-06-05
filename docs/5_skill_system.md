# skill 系统
skills 是热加载的，从 agents 源码路径以及 workspace 路径各维护一个 skills 文件夹，workspace 中的会覆盖内置 skills。 skill 的加载和发现主要依靠 SkillsLoader，它负责扫描路径、动态加载、提取 metadata、输出 summary 等。

# 渐进式披露
skills 以一个 SKILL.md 作为基本文件，其中以 YAML 格式的元数据为开头，写清楚 name、description、requirements 等，后续才是详细描述的正文。

除了特殊的 skill，希望其始终以完整的文本进入提示此外，一般的 skills 首次出现只会载入 name、description、path 等，并告知大模型，如果需要使用skill，可以调用 read_file 工具查看完整文档。这种方式可以节省上下文长度，提高模型表现。

# 参考资料
1. 开源项目[nanobot](https://github.com/HKUDS/nanobot)