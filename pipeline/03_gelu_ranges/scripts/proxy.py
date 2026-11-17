"""Board-like proxy runner. usage: proxy.py <jobs.json> <out.jsonl> [workers=3] [n=500] [base=onnx path]
jobs.json = list of [name, spec] (spec: see w3lib.apply_spec). Each worker = 1 process, ORT 1 thread (total <= 3 threads).
Already-done names in out.jsonl are skipped. Each line: name, spec, base, metrics, A2 check, boardlike sha, seconds."""
import sys, json, time, pathlib, multiprocessing as mp
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import w3lib as L


def work(args):
    name, spec, n, base = args
    t0 = time.time()
    try:
        m = L.load(base)
        mm = L.apply_spec(m, spec)
        ok, zps = L.a2_check(mm)
        b, rep = L.to_boardlike_bytes(mm, tag=L.safe(name)[:20])
        U, y = L.proxy_set("calibB"); F = L.fp32_ref("calibB")
        U, y, F = U[:n], y[:n], F[:n]
        E = L.embed(b, U, threads=1)
        r = L.metrics(E, F, y)
        out = {"name": name, "spec": spec, "base": str(base), **r, "a2_ok": ok,
               "n_gelu_rewritten": sum("skipped" not in g for g in rep.get("gelu", [])),
               "bl_sha": L._BL.get("sha"), "sec": round(time.time() - t0, 1)}
        d = L.W3 / "results" / "proxy_emb" / L.safe(pathlib.Path(base).stem); d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{L.safe(name)}.npy", E.astype(np.float16))
    except Exception as e:
        import traceback
        out = {"name": name, "spec": spec, "error": traceback.format_exc()[-1500:]}
    return out


if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); outp = pathlib.Path(sys.argv[2])
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    base = sys.argv[5] if len(sys.argv) > 5 else str(L.BASE)
    (L.W3 / "results" / "proxy_emb").mkdir(parents=True, exist_ok=True)
    L.fp32_ref("calibB", threads=3)
    done = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if "error" not in d: done.add(d["name"])
    todo = [(nm, sp, n, base) for nm, sp in jobs if nm not in done]
    print(f"{len(todo)} jobs, {W} workers, n={n}, base={base}", flush=True)
    t0 = time.time()
    with mp.Pool(W) as pool:
        for k, r in enumerate(pool.imap_unordered(work, todo)):
            with open(outp, "a") as f: f.write(json.dumps(r) + "\n")
            if "error" in r: print("ERROR", r["name"], r["error"][-400:], flush=True)
            else: print(f"[{k + 1}/{len(todo)} {time.time() - t0:.0f}s] {r['name']:40s} cos {r['cos']:.4f} agree {r['agree']:.3f} top1 {r['top1']:.3f} a2 {r['a2_ok']} ({r['sec']}s)", flush=True)
