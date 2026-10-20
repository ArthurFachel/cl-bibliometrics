"""Extra, independent LLM labellers (one paper per call, same guide).

    python -m clbib.llm_label label [--src to_label.csv] [--out labels_deepseek.csv]
    python -m clbib.llm_label label --src test_to_label.csv --out labels_deepseek_test.csv
    python -m clbib.llm_label label --model hermes:claude-sonnet-5 --out labels_sonnet.csv
    python -m clbib.llm_label compare [--split dev|test]   # pairwise kappa, all labellers

The labeller sees only title + abstract and the labelling guide below (same
definitions as validation/README.md). It never sees the rules or other labels.
Backends:
  - default: OpenAI-compatible DeepInfra API (DEEPINFRA_API_KEY), model
    deepseek-ai/DeepSeek-V4-Flash, temperature 0.
  - "hermes:<model>": a one-shot `hermes chat --safe-mode` call on the
    anthropic provider (no plugins, memory, rules or tools), used for Claude
    Sonnet. The CLI does not expose temperature.
Rows already in the output file are skipped, so the run is resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request

import polars as pl

from . import config as C

VAL = C.ROOT / "validation"
BASE = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
LABELS = ["is_cl", "replay", "regularization", "architecture", "optimization", "peft_llm",
          "plasticity", "theory", "survey_benchmark", "online", "federated", "rl", "application"]

GUIDE = """You label research papers from their title and abstract only. Return 1 when the label applies, else 0.
- is_cl: the paper is actually about continual / lifelong / incremental learning or catastrophic forgetting (0 = only mentions it in passing, or is about something else such as adult education)
- replay: stores, generates or replays past samples or features
- regularization: constrains updates by importance weights (EWC, SI, MAS), distillation or other loss penalties
- architecture: PROPOSES a new structure: grows the network, isolates parameters with masks, adds experts or task-specific modules. Analysis papers that only study which existing components to specialize do not count
- optimization: projects or constrains gradients (orthogonal, null space, GEM) or targets the loss landscape
- peft_llm: works with LLMs, VLMs or pre-trained foundation models, or uses LoRA, prompts or adapters
- plasticity: studies loss of plasticity, dormant neurons, resets or trainability
- theory: gives a formal or theoretical analysis (bounds, proofs, analytic models)
- survey_benchmark: is a survey or review, or contributes a benchmark or dataset
- online: uses online, task-free or streaming settings, or concept drift
- federated: uses federated learning
- rl: uses reinforcement learning or agents
- application: applies CL to a named domain outside ML itself (medical, fault diagnosis, remote sensing, speech...)
Labels are multi-label. Judge only what the text supports.
Answer with one JSON object and nothing else: {"is_cl":0|1, ... all 13 keys ..., "notes":"short reason for doubtful calls or empty"}"""


def parse(txt: str, labels: list[str] = LABELS) -> dict:
    d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    missing = [k for k in labels if k not in d]
    if missing:
        raise ValueError(f"missing keys {missing}")
    return {k: int(bool(int(d[k]))) for k in labels} | {"notes": str(d.get("notes", ""))}


def call_hermes(title: str, abstract: str, model: str, guide: str = GUIDE, labels: list[str] = LABELS) -> dict:
    prompt = f"{guide}\n\nTitle: {title}\n\nAbstract: {abstract}"
    last = ""
    for attempt in range(4):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as q:
            q.write(prompt)
        try:
            r = subprocess.run(["hermes", "chat", "--safe-mode", "--oneshot", "-Q", "-m", model,
                                "--provider", "anthropic", "-t", "", "--query-file", q.name],
                               capture_output=True, text=True, timeout=300)
            body = "\n".join(l for l in r.stdout.splitlines() if not l.startswith(("Session:", "session_id:")))
            return parse(body, labels)
        except (subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as e:
            last = str(e)
            time.sleep(3 * (attempt + 1))
        finally:
            os.unlink(q.name)
    raise RuntimeError(f"hermes call failed after retries: {last}")


def call(title: str, abstract: str, model: str, guide: str = GUIDE, labels: list[str] = LABELS) -> dict:
    if model.startswith("hermes:"):
        return call_hermes(title, abstract, model.split(":", 1)[1], guide, labels)
    key = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    if not key:
        sys.exit("DEEPINFRA_API_KEY not set")
    body = {"model": model, "temperature": 0, "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": guide},
                         {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"}]}
    r = urllib.request.Request(BASE, data=json.dumps(body).encode(), method="POST")
    r.add_header("Authorization", f"Bearer {key}")
    r.add_header("Content-Type", "application/json")
    last = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                txt = json.loads(resp.read())["choices"][0]["message"]["content"]
            return parse(txt, labels)
        except (urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as e:
            last = str(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {last}")


def label(src: str, out: str, model: str, workers: int = 1, guide: str = GUIDE,
          labels: list[str] = LABELS) -> None:
    rows = pl.read_csv(VAL / src, infer_schema_length=None).select("id", "first_year", "title", "abstract")
    path = VAL / out
    done = set(pl.read_csv(path)["id"].to_list()) if path.exists() else set()
    todo = [r for r in rows.iter_rows(named=True) if r["id"] not in done]
    cols = ["id", "first_year", "title", "abstract"] + labels + ["notes", "model"]
    lock, n = threading.Lock(), [len(done)]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not done:
            w.writeheader()

        def one(r: dict) -> None:
            lab = call(r["title"], r["abstract"] or "", model, guide, labels)
            with lock:
                w.writerow({**r, **lab, "model": model})
                f.flush()
                n[0] += 1
                print(f"{n[0]}/{rows.height} {r['id']}", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(one, r) for r in todo]:
                fut.result()
    print(f"-> {path}")


def kappa(x: list[int], y: list[int]) -> tuple[float, float | None]:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return po, ((po - pe) / (1 - pe) if pe < 1 else None)


LABELLERS = {"claude": "labels_llm{sfx}.csv", "deepseek": "labels_deepseek{sfx}.csv",
             "sonnet": "labels_sonnet{sfx}.csv"}


def compare(split: str = "dev") -> None:
    """Pairwise agreement between every LLM labeller on one split.

    claude = the original silver labels (labels_llm*.csv).
    """
    sfx = "" if split == "dev" else "_test"
    labs = {}
    for name, f in LABELLERS.items():
        path = VAL / f.format(sfx=sfx)
        if path.exists():
            labs[name] = pl.read_csv(path, infer_schema_length=None).select(["id"] + LABELS)
    names = list(labs)
    rows = []
    for k in LABELS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                j = labs[a].select("id", k).join(labs[b].select("id", k), on="id", suffix="_b")
                x = j[k].fill_null(0).cast(pl.Int8).to_list()
                y = j[k + "_b"].fill_null(0).cast(pl.Int8).to_list()
                po, ka = kappa(x, y)
                rows.append({"label": k, "pair": f"{a}-{b}", "n": len(x), "pos_a": sum(x), "pos_b": sum(y),
                             "agree": round(po, 2), "kappa": None if ka is None else round(ka, 2)})
    res = pl.DataFrame(rows)
    out = C.RESULTS / f"agreement_labellers_{split}.csv"
    res.write_csv(out)
    wide = res.pivot(on="pair", index="label", values="kappa")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
        print(f"kappa by pair ({split})")
        print(wide)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "compare"])
    ap.add_argument("--src", default="to_label.csv")
    ap.add_argument("--out", default="labels_deepseek.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "label":
        label(a.src, a.out, a.model, a.workers)
    else:
        compare(a.split)


if __name__ == "__main__":
    main()
