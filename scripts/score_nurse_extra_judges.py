"""Score the nurse_sub generation tree with additional judges, using the pipeline's own CoT categorical prompt and parser.

Writes out/nurse_sub/<judge>/scoring.json and greedy.json in the same format as the opus/gpt5/gemini runs, so
scripts/merge_judge_runs.py can average any set of judges. Resumable through out/nurse_sub/<judge>/_calls.jsonl.

Usage: uv run python scripts/score_nurse_extra_judges.py --judge haiku55|qwen|deepseek [--concurrency 16]
"""
from __future__ import annotations
import argparse, asyncio, copy, datetime, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scoring.methods.categorical_method import build_categorical_cot_prompt, parse_categorical_response

ROOT = Path(__file__).parent.parent / "out" / "nurse_sub"
JUDGES = {
    "haiku55": {"provider": "anthropic", "model": "claude-haiku-5-5", "label": "anthropic/claude-haiku-5-5"},
    "qwen": {"provider": "openrouter", "model": "qwen/qwen3.7-flash", "label": "openrouter/qwen/qwen3.7-flash",
             "extra_body": {"reasoning": {"enabled": False}}},
    "deepseek": {"provider": "openrouter", "model": "deepseek/deepseek-v3.2", "label": "openrouter/deepseek/deepseek-v3.2",
                 "extra_body": {"reasoning": {"enabled": False}}},
}
MAX_TOKENS = 1024  # the original runs' categorical.max_tokens with use_cot

async def ask(spec, client, prompt):
    if spec["provider"] == "anthropic":
        r = await client.messages.create(model=spec["model"], max_tokens=MAX_TOKENS,  # this model takes no temperature
                                         messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text"), (r.usage.input_tokens, r.usage.output_tokens)
    r = await client.chat.completions.create(model=spec["model"], max_tokens=MAX_TOKENS, temperature=0,
                                             messages=[{"role": "user", "content": prompt}], extra_body=spec.get("extra_body"))
    u = r.usage
    return (r.choices[0].message.content or ""), (getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True, choices=list(JUDGES)); ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args(); spec = JUDGES[a.judge]
    if spec["provider"] == "anthropic":
        import anthropic; client = anthropic.AsyncAnthropic()
    else:
        import openai; client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    base = json.load(open(ROOT / "opus" / "scoring.json")); greedy_base = json.load(open(ROOT / "opus" / "greedy.json"))
    questions = base["scoring_items"]["categorical_judgements"]
    out_dir = ROOT / a.judge; out_dir.mkdir(exist_ok=True); calls_path = out_dir / "_calls.jsonl"
    done = {}
    if calls_path.exists():
        for l in calls_path.open():
            d = json.loads(l); done[(d["kind"], d["i"], d["q"])] = d
    tasks = [("traj", i, q) for i in range(len(base["results"])) for q in range(len(questions))]
    tasks += [("greedy", i, q) for i in range(len(greedy_base["arms"])) for q in range(len(questions))]
    todo = [t for t in tasks if t not in done]
    print(f"judge={a.judge} calls={len(tasks)} todo={len(todo)}", flush=True)
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock(); t0 = time.time(); n = 0
    async def one(kind, i, q):
        nonlocal n
        text = base["results"][i]["text"] if kind == "traj" else greedy_base["arms"][i]["text"]
        prompt = build_categorical_cot_prompt(text, questions[q])
        async with sem:
            for attempt in range(6):
                try:
                    raw, usage = await ask(spec, client, prompt); break
                except Exception as e:
                    wait = min(60, 2 ** attempt); print(f"  {kind}{i}/q{q}: {type(e).__name__}: {str(e)[:120]} -> retry {wait}s", flush=True)
                    await asyncio.sleep(wait)
            else:
                print(f"  {kind}{i}/q{q}: FAILED", flush=True); return
        rec = {"kind": kind, "i": i, "q": q, "raw": raw, "verdict": parse_categorical_response(raw), "usage": usage}
        async with lock:
            with calls_path.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(kind, i, q)] = rec; n += 1
            if n % 500 == 0: print(f"  {n}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)
    await asyncio.gather(*(one(*t) for t in todo))
    # assemble the per-judge files in the original format
    out = copy.deepcopy(base)
    out["metadata"] = {**base["metadata"], "judge_model": spec["label"], "scored_at": datetime.datetime.now().isoformat(),
                       "scoring_file": f"scripts/score_nurse_extra_judges.py --judge {a.judge}"}
    missing = 0
    for i, r in enumerate(out["results"]):
        recs = [done.get(("traj", i, q)) for q in range(len(questions))]
        missing += sum(x is None for x in recs)
        r["method_raw"] = {"categorical": [x["raw"] if x else "" for x in recs]}
        r["method_scores"] = {"categorical": [x["verdict"] if x else None for x in recs]}
    json.dump(out, open(out_dir / "scoring.json", "w"), indent=2, ensure_ascii=False)
    g = copy.deepcopy(greedy_base); g["metadata"] = {**greedy_base["metadata"], "judge_model": spec["label"]}
    for i, arm in enumerate(g["arms"]):
        arm["structure_scores"] = [float(done[("greedy", i, q)]["verdict"]) if done.get(("greedy", i, q)) and done[("greedy", i, q)]["verdict"] is not None else None
                                   for q in range(len(questions))]
    json.dump(g, open(out_dir / "greedy.json", "w"), indent=2, ensure_ascii=False)
    cfg = json.load(open(ROOT / "opus" / "scoring_cfg.json")); cfg["model"] = spec["label"]
    json.dump(cfg, open(out_dir / "scoring_cfg.json", "w"), indent=2)
    unparsed = sum(1 for d in done.values() if d["verdict"] is None)
    tin = sum(d["usage"][0] or 0 for d in done.values()); tout = sum(d["usage"][1] or 0 for d in done.values())
    print(f"done: {len(done)} calls, missing {missing}, unparsed {unparsed}, tokens in {tin} out {tout} -> {out_dir}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
