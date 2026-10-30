"""Re-ask every nurse judge reply that stops before its final ANSWER line (the token cap was reached mid-reasoning).

Such replies carry no verdict; the fallback parser would otherwise read a list marker such as "1." as a vote.
Each cell is asked again with the same CoT prompt and the same story text (arm prefill + generated text for
trajectories, the greedy text for greedy decodes), through the same call path the judge was scored with, with a
larger token cap that doubles until the reply ends in an ANSWER line (at most three attempts).

  original judges (gemini, opus, gpt5): src.inference ModelRunner, as the original scoring; the reply goes into the raw
      slot that holds that question (these files store replies in the order q3, q2, q0, q1), the verdict into its slot,
      and a record into <judge>/_reasked.jsonl.
  new judges (haiku55, qwen, deepseek): scripts/score_nurse_extra_judges.ask(); a record is appended to
      <judge>/_calls.jsonl, after which score_nurse_extra_judges.py rebuilds scoring.json and greedy.json from it.

Usage: uv run python scripts/reask_truncated.py --judges gemini haiku55 qwen deepseek [--concurrency 16]
"""
from __future__ import annotations
import argparse, asyncio, datetime, json, os, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from src.scoring.methods.categorical_method import build_categorical_cot_prompt, parse_categorical_response
from src.common.text import strip_thinking_blocks
import score_nurse_extra_judges as X

ROOT = Path(__file__).parent.parent / "out" / "nurse_sub"
TAG = re.compile(r"answer\s*:\s*\**\s*(0|1|yes|no)\b", re.I)
ORIGINAL = {"opus": "anthropic/opus", "gpt5": "openai/gpt-5", "gemini": "gemini/gemini-2.5-flash"}
PERM = [3, 2, 0, 1]  # original judges: raw slot k holds the reply to question PERM[k]
CAPS = [4096, 8192, 16384]

def complete(raw): return bool(TAG.search(raw or ""))

def traj_text(base, i):
    r = base["results"][i]; return strip_thinking_blocks(base["arm_texts"][r["arm"]] + r["text"])

def run_original(judge, conc):
    from src.inference.model_runner import ModelRunner
    p = ROOT / judge / "scoring.json"; d = json.load(open(p)); qs = d["scoring_items"]["categorical_judgements"]
    todo = [(i, PERM[k]) for i, r in enumerate(d["results"]) for k, raw in enumerate(r["method_raw"]["categorical"]) if not complete(raw)]
    print(f"{judge}: {len(todo)} cut-off replies", flush=True)
    if not todo: return
    bk = ROOT / judge / "scoring_before_truncation_reask_backup.json"
    if not bk.exists(): shutil.copy(p, bk)
    runner = ModelRunner(ORIGINAL[judge])
    def one(cell):
        i, q = cell; prompt = build_categorical_cot_prompt(traj_text(d, i), qs[q]); resp = ""
        for cap in CAPS:
            resp = runner.generate(prompt=prompt, max_new_tokens=cap, temperature=0.0, prefilling=runner.skip_thinking_prefix)
            if complete(resp): break
        return i, q, cap, resp
    with ThreadPoolExecutor(max_workers=conc) as ex: out = list(ex.map(one, todo))
    with open(ROOT / judge / "_reasked.jsonl", "a") as f:
        for i, q, cap, resp in out:
            v = parse_categorical_response(resp) if complete(resp) else None
            r = d["results"][i]; old = r["method_scores"]["categorical"][q]
            r["method_scores"]["categorical"][q] = v; r["method_raw"]["categorical"][PERM.index(q)] = resp
            f.write(json.dumps({"kind": "traj", "i": i, "q": q, "question": qs[q], "old_verdict": old, "verdict": v, "raw": resp,
                                "max_new_tokens": cap, "asked_at": datetime.datetime.now().isoformat(),
                                "reason": "original reply stopped before its ANSWER line"}, ensure_ascii=False) + "\n")
    json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
    left = sum(1 for _, _, _, resp in out if not complete(resp))
    print(f"{judge}: re-asked {len(out)}, still without an ANSWER line: {left}", flush=True)

async def run_new(judge, conc):
    spec = dict(X.JUDGES[judge])
    if spec["provider"] == "anthropic":
        import anthropic; client = anthropic.AsyncAnthropic()
    else:
        import openai; client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    base = json.load(open(ROOT / "opus" / "scoring.json")); gbase = json.load(open(ROOT / "opus" / "greedy.json"))
    qs = base["scoring_items"]["categorical_judgements"]; calls = ROOT / judge / "_calls.jsonl"
    latest = {}
    for l in open(calls):
        rec = json.loads(l); latest[(rec["kind"], rec["i"], rec["q"])] = rec
    todo = [k for k, rec in latest.items() if not complete(rec.get("raw"))]
    print(f"{judge}: {len(todo)} cut-off replies", flush=True)
    if not todo: return
    shutil.copy(calls, ROOT / judge / "_calls_before_truncation_reask_backup.jsonl")
    sem = asyncio.Semaphore(conc); lock = asyncio.Lock(); left = 0
    async def one(key):
        nonlocal left
        kind, i, q = key
        text = traj_text(base, i) if kind == "traj" else gbase["arms"][i]["text"]
        prompt = build_categorical_cot_prompt(text, qs[q]); raw = ""; usage = (0, 0); cap = CAPS[0]
        async with sem:
            for cap in CAPS:
                call_spec = {**spec, "max_tokens": cap}
                for attempt in range(6):
                    try:
                        raw, usage = await X.ask(call_spec, client, prompt); break
                    except Exception as e:
                        await asyncio.sleep(min(60, 2 ** attempt))
                if complete(raw): break
        rec = {"kind": kind, "i": i, "q": q, "raw": raw, "verdict": parse_categorical_response(raw) if complete(raw) else None,
               "usage": usage, "text_mode": X.TEXT_MODE, "max_tokens": cap, "reason": "earlier reply stopped before its ANSWER line"}
        async with lock:
            with calls.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not complete(raw): left += 1
    await asyncio.gather(*(one(k) for k in todo))
    print(f"{judge}: re-asked {len(todo)}, still without an ANSWER line: {left}", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judges", nargs="+", required=True); ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    for j in a.judges:
        if j in ORIGINAL: run_original(j, a.concurrency)
        else: asyncio.run(run_new(j, a.concurrency))

if __name__ == "__main__":
    main()
