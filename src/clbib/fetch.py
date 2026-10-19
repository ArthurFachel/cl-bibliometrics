"""Download the corpus from OpenAlex with cursor pagination.

Raw pages are stored gzipped, one file per page, so a crash resumes from the
last saved cursor and the raw data stays auditable.

    python -m clbib.fetch            # fetch or resume every query
    python -m clbib.fetch --force    # wipe and refetch
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from . import config as C


def build_filter(spec: dict) -> str:
    terms = "|".join(f'"{t}"' for t in spec["terms"])
    f = f"title_and_abstract.search:{terms}"
    if spec["extra_filter"]:
        f += "," + spec["extra_filter"]
    return f


def api_key() -> str | None:
    """OPENALEX_API_KEY env var, else the secrets file (never hardcoded)."""
    import os
    from pathlib import Path
    k = os.environ.get("OPENALEX_API_KEY")
    if k:
        return k.strip()
    p = Path.home() / ".hermes" / ".secrets" / "openalex_api_key"
    k = p.read_text().strip() if p.exists() else ""
    if p.exists() and not k:
        sys.exit(f"{p} exists but is empty: the key was not saved. Re-run the save command.")
    return k or None


def get_json(params: dict, retries: int = 5) -> dict:
    key = api_key()
    if key:
        params = {**params, "api_key": key}
    url = C.OPENALEX + "?" + urllib.parse.urlencode(params, safe=":|,\"/")
    req = urllib.request.Request(url, headers={"User-Agent": f"cl-bibliometrics (mailto:{C.MAILTO})"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                body = e.read().decode(errors="replace")[:300]
                sys.exit(f"OpenAlex {e.code}: {body}\nSet OPENALEX_API_KEY or ~/.hermes/.secrets/openalex_api_key "
                         "(free key: https://openalex.org/settings/api).")
            wait = int(e.headers.get("Retry-After") or 2 ** attempt * 3)
            print(f"  retry {attempt + 1}/{retries} in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt * 3
            print(f"  retry {attempt + 1}/{retries} in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"OpenAlex failed after {retries} tries (filter={params.get('filter')})")


def fetch_query(name: str, spec: dict, force: bool) -> None:
    out = C.RAW / name
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("complete"):
        print(f"[{name}] complete ({manifest['n_works']} works), skipping. Use --force to refetch.")
        return

    pages = sorted(out.glob("page_*.json.gz"))
    cursor, n = "*", 0
    if pages:
        with gzip.open(pages[-1], "rt") as fh:
            last = json.load(fh)
        cursor = last["meta"].get("next_cursor")
        n = sum(len(json.load(gzip.open(p, "rt"))["results"]) for p in pages)
        print(f"[{name}] resuming after {len(pages)} pages ({n} works)")

    flt = build_filter(spec)
    page_no = len(pages)
    expected = None
    while cursor:
        data = get_json({"filter": flt, "per_page": 200, "cursor": cursor,
                         "select": ",".join(C.SELECT), "mailto": C.MAILTO})
        expected = data["meta"].get("count") or expected
        if not data["results"]:
            break
        page_no += 1
        with gzip.open(out / f"page_{page_no:04d}.json.gz", "wt") as fh:
            json.dump(data, fh)
        n += len(data["results"])
        cursor = data["meta"].get("next_cursor")
        if page_no % 10 == 0:
            print(f"[{name}] {n}/{expected}")
        time.sleep(0.2)

    manifest = {"query": name, "filter": flt, "terms": spec["terms"],
                "n_works": n, "openalex_count": expected, "pages": page_no,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "complete": True}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] done: {n} works (OpenAlex reported {expected})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name, spec in C.QUERIES.items():
        fetch_query(name, spec, args.force)


if __name__ == "__main__":
    main()
