"""Sanity check of the orchestrator's text files: no control characters, and no backslash path that was broken by a
newline (a line ending in a lone backslash followed by a word, e.g. 'night\' + newline + 'ondeterminism').
usage: python -I check_text.py <files...>"""
import sys, re
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    ctrl = [(i, hex(ord(c))) for i, c in enumerate(s) if ord(c) < 32 and c not in "\n"]
    broken = [m.start() for m in re.finditer(r"[A-Za-z0-9_]\\\n[a-z]", s)]
    print(f, "control chars:", ctrl[:5], "broken paths:", [s[max(0, i - 20):i + 15] for i in broken[:3]])
