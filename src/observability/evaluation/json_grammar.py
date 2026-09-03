"""llama.cpp GBNF：约束 Judge 只输出 JSON（字符串允许 UTF-8 日文）。"""

from __future__ import annotations

# 与 llama.cpp grammars/json.gbnf 对齐；用拼接避免 Python 转义把 \\x 吃掉
JSON_OBJECT_GBNF = "\n".join(
    [
        "root   ::= object",
        'value  ::= object | array | string | number | ("true" | "false" | "null") ws',
        "",
        "object ::=",
        '  "{" ws (',
        '            string ":" ws value',
        '    ("," ws string ":" ws value)*',
        '  )? "}" ws',
        "",
        "array  ::=",
        '  "[" ws (',
        "            value",
        '    ("," ws value)*',
        '  )? "]" ws',
        "",
        "string ::=",
        '  "\\"" (',
        r'    [^"\\\x7F\x00-\x1F] |',
        r'    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})',
        '  )* "\\"" ws',
        "",
        'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws',
        "",
        r"ws ::= ([ \t\n] ws)?",
    ]
)
