import os  # MCLIP_ROOT = the working folder (default: current folder)
"""W5 demo vocabulary: everyday indoor objects + close pairs.

Each entry: (display, prompt_name, imagenet_ids)
  display      = short label drawn on the board (ASCII, <= 12 chars)
  prompt_name  = the {name} put into the prompts ("a photo of a {name}." / the 80 OpenAI ImageNet templates)
  imagenet_ids = ImageNet-1k class indices (open_clip IMAGENET_CLASSNAMES order = ImageNetV2 folder name) whose
                 images are labelled with this entry in the PC accuracy test. [] = no ImageNet class exists:
                 the entry is only a distractor (it can be predicted, but has no test images).
Close pairs (by design): cup/mug, phone/remote, keyboard/laptop, pen/pencil, bottle/glass, monitor/television.
"""
VOCAB = [
    # close pairs
    ("cup",        "cup",               [968]),            # tea cup
    ("mug",        "mug",               [504]),            # coffee mug
    ("phone",      "mobile phone",      [487]),            # mobile phone
    ("remote",     "remote control",    [761]),            # remote control
    ("keyboard",   "computer keyboard", [508]),            # computer keyboard
    ("laptop",     "laptop",            [620, 681]),       # laptop computer, notebook computer
    ("pen",        "pen",               [418, 563]),       # ballpoint pen, fountain pen
    ("pencil",     "pencil",            []),               # no ImageNet class (distractor only)
    ("bottle",     "bottle",            [898, 907, 440, 737]),  # water / wine / beer / soda bottle
    ("glass",      "drinking glass",    [441, 572]),       # beer glass, goblet
    ("monitor",    "computer monitor",  [664]),            # monitor
    ("television", "television",        [851]),            # television
    ("mouse",      "computer mouse",    [673]),            # computer mouse
    # everyday indoor objects
    ("book",       "book",              [921]),            # dust jacket (book covers)
    ("chair",      "chair",             [559, 765, 423]),  # folding, rocking, barber chair
    ("lamp",       "lamp",              [846]),            # table lamp
    ("clock",      "clock",             [409, 530, 892]),  # analog, digital, wall clock
    ("watch",      "wristwatch",        [531]),            # digital watch
    ("backpack",   "backpack",          [414]),
    ("shoe",       "shoe",              [770, 630]),       # sneaker, slip-on shoe
    ("sunglasses", "sunglasses",        [836, 837]),       # two ImageNet "sunglasses" classes
    ("wallet",     "wallet",            [893]),
    ("teddy bear", "teddy bear",        [850]),
    ("banana",     "banana",            [954]),
    ("orange",     "orange",            [950]),
    ("apple",      "apple",             [948]),            # Granny Smith apple
    ("plant",      "potted plant",      [738]),            # plant pot
    ("couch",      "couch",             [831]),
    ("pillow",     "pillow",            [721]),
    ("umbrella",   "umbrella",          [879]),
    ("teapot",     "teapot",            [849]),
    ("toaster",    "toaster",           [859]),
    ("vase",       "vase",              [883]),
    ("plate",      "plate",             [923]),
    ("bowl",       "bowl",              [809, 659]),       # soup bowl, mixing bowl
    ("spoon",      "spoon",             [910]),            # wooden spoon
    ("candle",     "candle",            [470]),
    ("towel",      "towel",             [434]),            # bath towel
    ("printer",    "printer",           [742]),
    ("desk",       "desk",              [526]),
    ("scissors",   "scissors",          []),               # no ImageNet class (distractor only)
]

PAIRS = [("cup", "mug"), ("phone", "remote"), ("keyboard", "laptop"), ("pen", "pencil"), ("bottle", "glass"),
         ("monitor", "television")]


def display_names():
    return [d for d, _, _ in VOCAB]


def prompt_names():
    return [p for _, p, _ in VOCAB]


def imagenet_to_vocab():
    """-> dict imagenet_id -> vocab index (each ImageNet id maps to at most one entry)."""
    m = {}
    for vi, (_, _, ids) in enumerate(VOCAB):
        for c in ids:
            assert c not in m, c
            m[c] = vi
    return m


def check():
    import pathlib
    names = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W0_data\imagenet_classnames.txt").read_text(encoding="utf-8").splitlines()
    assert len(names) == 1000
    d = display_names()
    assert len(set(d)) == len(d)
    assert all(len(x) <= 12 and x.isascii() for x in d)
    out = []
    for disp, pr, ids in VOCAB:
        out.append((disp, pr, [(c, names[c]) for c in ids]))
    return out


if __name__ == "__main__":
    for row in check():
        print(row)
    print(len(VOCAB), "entries;", sum(len(i) for _, _, i in VOCAB), "ImageNet classes mapped")
