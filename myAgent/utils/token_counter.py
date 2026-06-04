import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude 都用这个

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

def count_message_tokens(msg: dict) -> int:
    # role + content 的估算，工具调用另外加
    n = 4  # 每条消息的固定开销
    for key in ("content", "name"):
        if msg.get(key):
            n += count_tokens(str(msg[key]))
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            n += count_tokens(str(tc.get("function", {}).get("arguments", "")))
            n += count_tokens(str(tc.get("function", {}).get("name", "")))
            n += 4
    return n
