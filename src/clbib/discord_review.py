"""Blind human labelling of the validation sample through Discord native polls.

One message per paper (title, year, abstract) with a multi-select poll. The
collector reads ONLY the votes of the reviewer's user id and writes
validation/labels_human.csv in the same format as the LLM labels.

    python -m clbib.discord_review post --channel ID [--limit N]
    python -m clbib.discord_review collect --channel ID --user USER_ID

Needs DISCORD_BOT_TOKEN in the environment (never stored in the repo).
State (message id per paper) is kept in data/discord_review_state.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import polars as pl

from . import config as C

API = "https://discord.com/api/v10"
STATE = C.ROOT / "data" / "discord_review_state.json"
SOURCE = C.ROOT / "validation" / "human_review.csv"
OUT = C.ROOT / "validation" / "labels_human.csv"

# Poll answers (Discord max 10). Order matters: answer_id = index + 1.
ANSWERS = [
    ("is_not_cl", "Não é CL"),
    ("replay", "replay"),
    ("regularization", "regularization"),
    ("architecture", "architecture"),
    ("peft_llm", "peft_llm (LLM/VLM, LoRA, prompt, adapter)"),
    ("theory", "theory (prova, bound, modelo analítico)"),
    ("survey_benchmark", "survey / novo benchmark ou dataset"),
    ("online", "online / task-free / streaming"),
    ("application", "application (domínio fora de ML)"),
    ("none", "Nenhuma das acima"),
]
NOT_COVERED = ["optimization", "plasticity", "federated", "rl"]
MAX_ABSTRACT = 1550
SEED = 20261019


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not t:
        sys.exit("DISCORD_BOT_TOKEN not set")
    return t


def req(method: str, path: str, body: dict | None = None) -> dict | list:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method)
    r.add_header("Authorization", f"Bot {token()}")
    r.add_header("User-Agent", "DiscordBot (cl-bibliometrics, 1.0)")
    if data:
        r.add_header("Content-Type", "application/json")
    for _ in range(5):
        try:
            with urllib.request.urlopen(r, timeout=45) as resp:
                raw = resp.read().decode() or "{}"
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="ignore")
            if e.code == 429:
                time.sleep(float(json.loads(txt).get("retry_after", 1)) + 0.5)
                continue
            raise RuntimeError(f"{method} {path} -> {e.code}: {txt[:300]}") from e
        except urllib.error.URLError:
            time.sleep(5)
            continue
    raise RuntimeError(f"{method} {path}: retries exhausted")


def render(i: int, n: int, row: dict) -> str:
    abstract = (row["abstract"] or "").strip()
    if len(abstract) > MAX_ABSTRACT:
        abstract = abstract[:MAX_ABSTRACT].rsplit(" ", 1)[0] + " ... (cortado)"
    return (f"**[{i}/{n}] {row['title']}** ({row['first_year']})\n\n{abstract}\n\n"
            f"<{row['id']}>")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"posted": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def post(channel: str, limit: int | None) -> None:
    rows = pl.read_csv(SOURCE, infer_schema_length=None).select("id", "first_year", "title", "abstract").to_dicts()
    random.Random(SEED).shuffle(rows)  # blind to the stratified order
    st = load_state()
    st["channel"] = channel
    n, done = len(rows), 0
    for i, row in enumerate(rows, start=1):
        if row["id"] in st["posted"]:
            continue
        if limit is not None and done >= limit:
            break
        poll = {"question": {"text": "Quais rótulos se aplicam? Marque todos (ou 'Nenhuma')."},
                "answers": [{"poll_media": {"text": label[:55]}} for _, label in ANSWERS],
                "duration": 336, "allow_multiselect": True}
        msg = req("POST", f"/channels/{channel}/messages",
                  {"content": render(i, n, row)[:1990], "poll": poll, "allowed_mentions": {"parse": []}})
        st["posted"][row["id"]] = {"message_id": msg["id"], "position": i}
        save_state(st)
        done += 1
        time.sleep(1.2)
    print(f"posted {done}; total posted {len(st['posted'])}/{n}")


def collect(channel: str, user: str) -> None:
    st = load_state()
    src = pl.read_csv(SOURCE, infer_schema_length=None).select("id", "first_year", "title", "abstract")
    rows, answered = [], 0
    for pid, info in st["posted"].items():
        votes = set()
        for aid, (key, _) in enumerate(ANSWERS, start=1):
            users = req("GET", f"/channels/{channel}/polls/{info['message_id']}/answers/{aid}?limit=100")
            if any(u["id"] == user for u in users.get("users", [])):
                votes.add(key)
            time.sleep(0.25)
        if not votes:
            continue  # not reviewed yet
        answered += 1
        rec = {"id": pid, "is_cl": 0 if "is_not_cl" in votes else 1}
        for key, _ in ANSWERS:
            if key not in ("is_not_cl", "none"):
                rec[key] = int(key in votes)
        conflict = "none" in votes and len(votes - {"none", "is_not_cl"}) > 0
        rec["notes"] = "conflict: 'Nenhuma' plus tags" if conflict else ""
        rows.append(rec)
    if not rows:
        print("no votes from that user yet")
        return
    lab = pl.DataFrame(rows).with_columns([pl.lit(None, dtype=pl.Int64).alias(k) for k in NOT_COVERED])
    out = src.join(lab, on="id", how="inner")
    out.write_csv(OUT)
    print(f"{answered}/{len(st['posted'])} papers reviewed -> {OUT}")
    print(f"not covered by the poll (left blank): {', '.join(NOT_COVERED)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["post", "collect"])
    ap.add_argument("--channel", required=True)
    ap.add_argument("--user")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "post":
        post(a.channel, a.limit)
    else:
        if not a.user:
            sys.exit("--user is required for collect")
        collect(a.channel, a.user)


if __name__ == "__main__":
    main()
