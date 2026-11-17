"""11.11: builds release/live/board/vocab_add_demo.py ("add a word" demo: a typed word joins the top-3 list).
Uses make_live.py's builder unchanged (same known-good core, same checks: every open() is a read, no writes);
only the job list is replaced here. make_live.py and the existing board scripts are not modified.
usage: python -I make_vocab.py"""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import sys
sys.path.insert(0, os.environ.get("MCLIP_ROOT", ".") + "/release/live")
import make_live as ML  # noqa: E402

ML.JOBS = {
    "vocab_add_demo.py": ("Add-a-word demo: camera, v2 top-3; a word typed on the PC joins the word list (runs until stopped)",
                          ["v2:v2_lib.py", "live_lib.py", "vocab_add_main.py"], ["text"], {}),
}

if __name__ == "__main__":
    ML.build()
