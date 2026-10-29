"""Sample target-model responses for every arm.

Usage: uv run python steering/generate.py --run pilot --n 20 [--model gemini-2.5-flash-lite] [--concurrency 8]
Writes steering/out/<run>/responses.jsonl, one line per sample, resumable (existing ids are skipped).
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import ARMS, ORDER

HERE = Path(__file__).resolve().parent
MAX_TOKENS = 4096

def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.open() if l.strip()}

async def gen_gemini(client, model, prompt, temperature):
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=MAX_TOKENS)
    r = await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)
    um = r.usage_metadata
    usage = {"in": getattr(um, "prompt_token_count", None), "out": getattr(um, "candidates_token_count", None),
             "thinking": getattr(um, "thoughts_token_count", None)}
    finish = None
    try:
        finish = str(r.candidates[0].finish_reason)
    except Exception:
        pass
    return r.text or "", usage, finish

async def gen_openrouter(client, model, prompt, temperature):
    r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                             temperature=temperature, max_tokens=MAX_TOKENS)
    ch = r.choices[0]; u = r.usage
    usage = {"in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None),
             "cost": getattr(u, "cost", None), "provider": getattr(r, "provider", None), "model_served": r.model}
    return ch.message.content or "", usage, ch.finish_reason

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "gemini"])
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--arms", default=",".join(ORDER))
    args = ap.parse_args()
    if args.provider == "gemini":
        from google import genai
        client = genai.Client()  # GEMINI_API_KEY
    else:
        import openai
        client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out_dir = HERE / "out" / args.run; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "responses.jsonl"
    done = existing_ids(out)
    todo = [(arm, i) for arm in args.arms.split(",") for i in range(args.n) if f"{arm}-{i:03d}" not in done]
    print(f"run={args.run} model={args.model} todo={len(todo)} done={len(done)}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    t0 = time.time(); n_ok = 0
    async def one(arm, i):
        nonlocal n_ok
        rid = f"{arm}-{i:03d}"
        prompt = ARMS[arm]["text"]
        async with sem:
            for attempt in range(6):
                try:
                    if args.provider == "gemini":
                        text, usage, finish = await gen_gemini(client, args.model, prompt, args.temperature)
                    else:
                        text, usage, finish = await gen_openrouter(client, args.model, prompt, args.temperature)
                    break
                except Exception as e:  # rate limits, transient errors
                    wait = min(60, 2 ** attempt + 1)
                    print(f"  {rid}: {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s", flush=True)
                    await asyncio.sleep(wait)
            else:
                print(f"  {rid}: FAILED", flush=True); return
        rec = {"id": rid, "arm": arm, "group": ARMS[arm]["group"], "signal": ARMS[arm]["signal"], "idx": i,
               "model": args.model, "provider": args.provider, "temperature": args.temperature, "max_tokens": MAX_TOKENS,
               "prompt": prompt, "response": text, "usage": usage, "finish": finish,
               "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        async with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  {n_ok}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)
    await asyncio.gather(*(one(a, i) for a, i in todo))
    print(f"done: {n_ok} new, total {len(existing_ids(out))} in {out}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
