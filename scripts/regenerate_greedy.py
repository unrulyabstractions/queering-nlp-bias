"""Regenerate the nurse greedy decodes at the study's token cap and score them with every judge.

The greedy decodes in out/nurse_sub/greedy.json were generated with the pipeline default of 128 new tokens (their
config file left max_new_tokens unset), so each story stops mid-sentence; the sampled stories use 512. This script
  1. backs up out/nurse_sub/greedy.json and every <judge>/greedy.json (suffix _128tok_backup.json),
  2. regenerates one temperature-0 decode per arm with Claude Haiku 4.5 at 512 new tokens (src.generation.greedy_paths),
  3. scores each decode on the four questions with all six judges, on arm prefill + generated text as the stories are
     scored, with the CoT prompt; every reply must end in its ANSWER line (cap 1024, re-asked at 4096, 8192, 16384),
  4. writes the new texts and scores into out/nurse_sub/greedy.json and every <judge>/greedy.json, keeps each reply in
     <judge>/_greedy_calls.jsonl, and appends the new judges' records to <judge>/_calls.jsonl.

Usage: uv run python scripts/regenerate_greedy.py
"""
from __future__ import annotations
import asyncio, copy, datetime, json, os, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from src.generation.generation_config import GenerationConfig
from src.generation.greedy_paths import produce_greedy_paths
from src.inference.model_runner import ModelRunner
from src.scoring.methods.categorical_method import build_categorical_cot_prompt, parse_categorical_response
from src.common.text import strip_thinking_blocks
import score_nurse_extra_judges as X

ROOT = Path(__file__).parent.parent / "out" / "nurse_sub"
ORIGINAL = {"opus": "anthropic/opus", "gpt5": "openai/gpt-5", "gemini": "gemini/gemini-2.5-flash"}
NEW = ["haiku55", "qwen", "deepseek"]
CAPS = [1024, 4096, 8192, 16384]
FINAL = re.compile(r"ANSWER\s*:\s*\**\s*[01]\b")

def main():
    gpath = ROOT / "greedy.json"
    for p in [gpath] + [ROOT / j / "greedy.json" for j in list(ORIGINAL) + NEW]:
        bk = p.with_name(p.stem + "_128tok_backup.json")
        if not bk.exists(): shutil.copy(p, bk)
    gen = json.load(open(ROOT / "generation.json"))["config"]
    cfg = GenerationConfig.from_dict(gen) if hasattr(GenerationConfig, "from_dict") else GenerationConfig(**{k: gen[k] for k in gen if k in GenerationConfig.__dataclass_fields__})
    assert cfg.max_new_tokens == 512, cfg.max_new_tokens
    runner = ModelRunner(cfg.model)
    out = produce_greedy_paths(runner, cfg, config_path="out/nurse_sub/generation.json (max_new_tokens 512)")
    new = json.loads(json.dumps(out.to_dict() if hasattr(out, "to_dict") else out.__dict__, default=lambda o: o.__dict__))
    old = json.load(open(gpath.with_name("greedy_128tok_backup.json")))
    arm_texts = json.load(open(ROOT / "opus" / "scoring.json"))["arm_texts"]
    texts = {a["name"]: a["text"] for a in new["arms"]}
    for a in old["arms"]:
        t = texts[a["name"]]; print(f"{a['name']}: old {len(a['text'])} chars, new {len(t)} chars, new starts with old: {t.startswith(a['text'])}", flush=True)
    qs = json.load(open(ROOT / "opus" / "scoring.json"))["scoring_items"]["categorical_judgements"]
    arms = [a["name"] for a in new["arms"]]
    judge_text = {arm: strip_thinking_blocks(arm_texts[arm] + texts[arm]) for arm in arms}
    cells = [(k, arm, q) for k, arm in enumerate(arms) for q in range(4)]
    replies = {}

    def ask_original(j):
        r = ModelRunner(ORIGINAL[j])
        def one(c):
            k, arm, q = c; prompt = build_categorical_cot_prompt(judge_text[arm], qs[q]); resp = ""
            for cap in CAPS:
                resp = r.generate(prompt=prompt, max_new_tokens=cap, temperature=0.0, prefilling=r.skip_thinking_prefix)
                if FINAL.search(resp or ""): break
            return c, cap, resp
        with ThreadPoolExecutor(max_workers=10) as ex:
            for c, cap, resp in ex.map(one, cells): replies[(j,) + c] = (cap, resp)
    for j in ORIGINAL: ask_original(j); print(f"{j}: scored", flush=True)

    async def ask_new(j):
        spec = dict(X.JUDGES[j])
        if spec["provider"] == "anthropic":
            import anthropic; client = anthropic.AsyncAnthropic()
        else:
            import openai; client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        async def one(c):
            k, arm, q = c; prompt = build_categorical_cot_prompt(judge_text[arm], qs[q]); raw = ""; usage = (0, 0)
            for cap in CAPS:
                for attempt in range(6):
                    try:
                        raw, usage = await X.ask({**spec, "max_tokens": cap}, client, prompt); break
                    except Exception:
                        await asyncio.sleep(min(60, 2 ** attempt))
                if FINAL.search(raw or ""): break
            replies[(j,) + c] = (cap, raw, usage)
        await asyncio.gather(*(one(c) for c in cells))
    for j in NEW: asyncio.run(ask_new(j)); print(f"{j}: scored", flush=True)

    now = datetime.datetime.now().isoformat()
    new_meta = {**old.get("metadata", {}), "generated_at": now, "max_new_tokens": 512}
    base = copy.deepcopy(old); base["metadata"] = new_meta
    for a in base["arms"]:
        e = next(x for x in new["arms"] if x["name"] == a["name"])
        for key in ("text", "n_generated_tokens", "token_ids", "prefill_length"): a[key] = e.get(key, a.get(key))
        a.pop("structure_scores", None)
    json.dump(base, open(gpath, "w"), indent=2, ensure_ascii=False)
    left = 0
    for j in list(ORIGINAL) + NEW:
        g = copy.deepcopy(base); g["metadata"] = {**json.load(open(ROOT / j / "greedy_128tok_backup.json")).get("metadata", {}), **new_meta}
        with open(ROOT / j / "_greedy_calls.jsonl", "a") as f:
            for k, a in enumerate(g["arms"]):
                sc = []
                for q in range(4):
                    rec = replies[(j, k, a["name"], q)]; cap, raw = rec[0], rec[1]
                    v = parse_categorical_response(raw) if FINAL.search(raw or "") else None
                    left += v is None; sc.append(None if v is None else float(v))
                    f.write(json.dumps({"arm": a["name"], "q": q, "question": qs[q], "verdict": v, "raw": raw, "max_tokens": cap,
                                        "text_mode": "prefill+generated", "at": now}, ensure_ascii=False) + "\n")
                    if j in NEW:
                        with open(ROOT / j / "_calls.jsonl", "a") as cf:
                            cf.write(json.dumps({"kind": "greedy", "i": k, "q": q, "raw": raw, "verdict": v, "usage": rec[2] if len(rec) > 2 else (0, 0),
                                                 "text_mode": "prefill+generated", "max_tokens": cap, "reason": "greedy decode regenerated at 512 tokens"}, ensure_ascii=False) + "\n")
                a["structure_scores"] = sc
        json.dump(g, open(ROOT / j / "greedy.json", "w"), indent=2, ensure_ascii=False)
    print(f"done; greedy cells without a final ANSWER line: {left}", flush=True)

if __name__ == "__main__":
    main()
