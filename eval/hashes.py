"""Print SHA256 + size of the W0 data files (for README.md)."""
import os  # MCLIP_ROOT = the working folder (default: current folder)
import hashlib, pathlib
W = pathlib.Path(os.environ.get("MCLIP_ROOT", ".") + r"\night\W0_data")
names = ["selection_u8.npy", "selection_labels.npy", "test_u8.npy", "test_labels.npy", "calib_pool_u8.npy",
         "calib_pool_labels.npy", "text_imagenet.npy", "text_imagenet_simple.npy", "selection_files.txt", "test_files.txt",
         "calib_pool_files.txt", "unused_files.txt", "imagenet_classnames.txt", "imagenet_templates80.txt"]
for n in names:
    p = W / n
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    print(f"| `{n}` | {p.stat().st_size:,} | `{h.hexdigest()}` |")
