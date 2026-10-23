"""
fetch_public_corpus.py -- Builds data/base.jsonl from the two museums' own
public collection APIs. No credentials, no private endpoint.

    python src/fetch_public_corpus.py                  # 2,500 + 2,500 records
    python src/fetch_public_corpus.py --per-source 500 # smaller, faster
    python src/main.py fetch                           # same thing via the CLI

Sources:
  Cleveland Museum of Art   https://openaccess-api.clevelandart.org/api/artworks/
      Open Access metadata, CC0. (A record's `share_license_status` describes
      the IMAGE's rights -- metadata is CC0 either way. No images are
      downloaded; image_path is only filled for CC0 images.)
  Art Institute of Chicago  https://api.artic.edu/api/v1/artworks
      CC0, EXCEPT the `description` field, which is CC BY 4.0 and needs
      attribution -- see the README's Data section and data/SOURCES.json.

WHY A FIXED SEED LIST IS ALWAYS INCLUDED: the demo questions (main.py's
DEMO_QUESTIONS, demo_phoenix_evals.py, the fixture) name specific works -- Coney
Island Beach, The Casting, Railroad Landscape #33. A bulk page-through would
include them only by luck, and a question about a record that isn't in the
corpus tests the abstention gate rather than what it says it tests. Seeds are
fetched by ID first; any that 404 are REPORTED, not silently skipped.

Output is ingest.py's "structured" format, so the existing pipeline runs
unchanged:  fetch -> data/base.jsonl -> ingest -> data/normalized.jsonl.

Every API page is cached under data/raw/, so re-running is free, offline, and
reproducible; pass --refresh to re-download. Nothing is dropped silently: every
record that is skipped is counted by reason in the summary and the manifest.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

try:
    import requests
except ImportError:  # requests is in requirements.txt; fail with guidance, not a trace
    requests = None

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "base.jsonl")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "SOURCES.json")

CMA_BASE = "https://openaccess-api.clevelandart.org/api/artworks/"
AIC_BASE = "https://api.artic.edu/api/v1/artworks"

CMA_INSTITUTION = "Cleveland Museum of Art"
AIC_INSTITUTION = "Art Institute of Chicago"

CMA_FIELDS = ("id,accession_number,title,creators,creation_date,technique,"
              "measurements,type,description,wall_description,"
              "share_license_status,images")
AIC_FIELDS = ("id,title,artist_title,date_display,medium_display,dimensions,"
              "classification_title,description,short_description,"
              "main_reference_number,image_id,is_public_domain")

# AIC asks API clients to identify themselves with this header, and without
# it requests are served a bot challenge instead of JSON. Override the contact
# with PUBLIC_API_CONTACT so requests identify whoever is running the fetch.
AIC_USER_AGENT = f"museum-qa-evals ({os.getenv('PUBLIC_API_CONTACT', 'github.com/gargi-09')})"

# AIC documents a 60 requests/minute limit for anonymous clients.
AIC_MIN_INTERVAL_S = 1.05
CMA_MIN_INTERVAL_S = 0.25
CMA_PAGE_SIZE = 1000
AIC_PAGE_SIZE = 100
MAX_RETRIES = 4

# Works the demo questions are about. All verified to exist on the public APIs
# when this was written; the fetch reports any that have since been withdrawn.
SEED_IDS = [
    "cma-117571",  # Reginald Marsh, Coney Island Beach
    "cma-171726",  # Leon Levinstein, Coney Island
    "cma-157127",  # Andreas Feininger, Coney Island
    "aic-59820",   # Reginald Marsh, Tattoo and Haircut
    "cma-160794",  # Louise Bourgeois, Untitled (Ode to My Mother portfolio)
    "cma-160059",  # Louise Bourgeois, Ste Sebastienne
    "cma-168184",  # Omer Fast, The Casting
    "cma-169223",  # Jeff Brouws, Railroad Landscape #33 ... Pine Plains
    "cma-167554",  # Helen Frankenthaler, Humming Gold
    "aic-265559",  # Ray Johnson, Untitled (Taoist Collages)
]


# ---------------------------------------------------------------------------
# HTTP with politeness, retries and a page cache
# ---------------------------------------------------------------------------

class PoliteClient:
    def __init__(self, refresh=False):
        if requests is None:
            raise RuntimeError("`requests` is not installed -- run `pip install -r requirements.txt`.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": AIC_USER_AGENT,
                                     "AIC-User-Agent": AIC_USER_AGENT})
        self.refresh = refresh
        self._last_call = {}
        self.network_calls = 0
        self.cache_hits = 0

    def get_json(self, url, params, cache_key, min_interval):
        cache_path = os.path.join(RAW_DIR, cache_key + ".json")
        if not self.refresh and os.path.exists(cache_path):
            self.cache_hits += 1
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        host = url.split("/")[2]
        for attempt in range(1, MAX_RETRIES + 1):
            wait = min_interval - (time.time() - self._last_call.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.time()
            self.network_calls += 1
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.exceptions.RequestException as e:
                err = f"network error: {e}"
            else:
                if resp.status_code == 404:
                    return None
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        # A 200 that is not JSON is a challenge/maintenance page,
                        # not data. Treated as a failure, never parsed.
                        err = "200 OK but body is not JSON (bot challenge or outage page?)"
                    else:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False)
                        return data
                else:
                    err = f"HTTP {resp.status_code}"
                    if resp.status_code not in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"{url} -> {err}: {resp.text[:300]}")
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(int(retry_after), 120))
            backoff = 2 ** attempt
            print(f"[fetch]   {err} -- retry {attempt}/{MAX_RETRIES} in {backoff}s")
            time.sleep(backoff)
        raise RuntimeError(f"{url} failed after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Field mapping -> ingest.py "structured" records
# ---------------------------------------------------------------------------

def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


NON_AUTHOR_ROLES = {"publisher", "printer", "published by", "printed by",
                    "distributor", "retailer", "former attribution"}


def _cma_artist(creators):
    """CMA creator descriptions read 'Reginald Marsh (American, born France,
    1898-1954)'. Keep the name only: nationality and life dates are not the
    artist field, and leaving them in would stop entity resolution matching
    the same person as recorded by AIC ('Reginald Marsh').

    Publishers and printers are dropped: CMA lists them as creators too, and
    measured on cma-160794 that turned the artist into 'Louise Bourgeois and
    Editions du Solstice and Harlan & Weaver' -- which would defeat both the
    entity-resolution match and any question about who made the work."""
    names = []
    for c in creators or []:
        if (c.get("role") or "").lower() in NON_AUTHOR_ROLES:
            continue
        desc = _clean(c.get("description"))
        if not desc:
            continue
        name = re.split(r"\s+\(", desc, maxsplit=1)[0].strip()
        if name and name not in names:
            names.append(name)
    return " and ".join(names) or None


def map_cma(rec):
    image = (rec.get("images") or {}).get("web") or {}
    cc0_image = rec.get("share_license_status") == "CC0"
    return {
        "id": f"cma-{rec['id']}",
        "institution": CMA_INSTITUTION,
        "accession_number": _clean(rec.get("accession_number")),
        "format": "structured",
        "title": _clean(rec.get("title")),
        "artist": _cma_artist(rec.get("creators")),
        "year": _clean(rec.get("creation_date")),
        "medium": _clean(rec.get("technique")),
        "dimensions": _clean(rec.get("measurements")),
        "classification": _clean(rec.get("type")),
        "description": _clean(rec.get("description") or rec.get("wall_description")),
        "image_path": image.get("url") if cc0_image else None,
        "source": "cma_open_access",
        "license": "CC0-1.0",
        "image_license_status": rec.get("share_license_status"),
    }


def map_aic(rec):
    has_image = rec.get("image_id") and rec.get("is_public_domain")
    description = rec.get("description")
    return {
        "id": f"aic-{rec['id']}",
        "institution": AIC_INSTITUTION,
        "accession_number": _clean(rec.get("main_reference_number")),
        "format": "structured",
        "title": _clean(rec.get("title")),
        "artist": _clean(rec.get("artist_title")),
        "year": _clean(rec.get("date_display")),
        "medium": _clean(rec.get("medium_display")),
        "dimensions": _clean(rec.get("dimensions")),
        "classification": _clean(rec.get("classification_title")),
        "description": _clean(description or rec.get("short_description")),
        "image_path": (f"https://www.artic.edu/iiif/2/{rec['image_id']}/full/843,/0/default.jpg"
                       if has_image else None),
        "source": "aic_api",
        "license": "CC0-1.0; description CC-BY-4.0" if description else "CC0-1.0",
    }


def skip_reason(record):
    """Why a mapped record is not usable, or None. A record with no title and
    no description has nothing for retrieval to match or the model to cite."""
    if not record["title"] and not record["description"]:
        return "no title and no description"
    if not record["description"]:
        return "no description"
    return None


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_by_id(client, rid):
    prefix, _, num = rid.partition("-")
    if prefix == "cma":
        data = client.get_json(f"{CMA_BASE}{num}", {"fields": CMA_FIELDS},
                               f"cma/id_{num}", CMA_MIN_INTERVAL_S)
        return map_cma(data["data"]) if data and data.get("data") else None
    if prefix == "aic":
        data = client.get_json(f"{AIC_BASE}/{num}", {"fields": AIC_FIELDS},
                               f"aic/id_{num}", AIC_MIN_INTERVAL_S)
        return map_aic(data["data"]) if data and data.get("data") else None
    raise ValueError(f"Unknown id prefix in {rid!r} -- expected cma-<n> or aic-<n>")


def fetch_cma_bulk(client, want, seen, skipped):
    """Pages the CMA catalogue in its default (stable, id-based) order."""
    out, skip = [], 0
    while len(out) < want:
        data = client.get_json(CMA_BASE, {"limit": CMA_PAGE_SIZE, "skip": skip,
                                          "fields": CMA_FIELDS},
                               f"cma/page_{skip}", CMA_MIN_INTERVAL_S)
        rows = (data or {}).get("data") or []
        if not rows:
            break
        for raw in rows:
            rec = map_cma(raw)
            if rec["id"] in seen:
                continue
            reason = skip_reason(rec)
            if reason:
                skipped[f"cma: {reason}"] += 1
                continue
            seen.add(rec["id"])
            out.append(rec)
            if len(out) >= want:
                break
        skip += CMA_PAGE_SIZE
        print(f"[fetch]   CMA: {len(out)}/{want}")
    return out


def fetch_aic_bulk(client, want, seen, skipped):
    """AIC search, restricted server-side to artworks that HAVE a description
    and sorted by id so the sample is the same on every run.

    KEYSET PAGINATION, not page numbers: the search endpoint refuses any
    request reaching past its first 1,000 results with a 403 ("You have
    requested too many results"), measured on the first full fetch. So every
    request is page 1 of "id > the last id already seen", which never leaves
    that window however many records are wanted."""
    out, after_id = [], 0
    while len(out) < want:
        params = {"query[bool][must][0][exists][field]": "description",
                  "query[bool][must][1][range][id][gt]": after_id,
                  "sort[id][order]": "asc", "limit": AIC_PAGE_SIZE, "fields": AIC_FIELDS}
        data = client.get_json(f"{AIC_BASE}/search", params, f"aic/search_after_{after_id}",
                               AIC_MIN_INTERVAL_S)
        rows = (data or {}).get("data") or []
        if not rows:
            break
        after_id = rows[-1]["id"]
        for raw in rows:
            rec = map_aic(raw)
            if rec["id"] in seen:
                continue
            reason = skip_reason(rec)
            if reason:
                skipped[f"aic: {reason}"] += 1
                continue
            seen.add(rec["id"])
            out.append(rec)
            if len(out) >= want:
                break
        print(f"[fetch]   AIC: {len(out)}/{want}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(per_source=2500, out_path=DEFAULT_OUT, ids=None, refresh=False, seeds=True):
    """Returns the manifest dict. `ids` fetches exactly those records and no
    bulk pages (used to rebuild the fixture corpus)."""
    client = PoliteClient(refresh=refresh)
    skipped = Counter()
    seen = set()
    records, missing = [], []

    wanted_ids = list(ids) if ids else (SEED_IDS if seeds else [])
    if wanted_ids:
        print(f"[fetch] Fetching {len(wanted_ids)} record(s) by id...")
    for rid in wanted_ids:
        rec = fetch_by_id(client, rid)
        if rec is None:
            missing.append(rid)
            print(f"[fetch]   {rid}: NOT FOUND on the public API (withdrawn or unpublished)")
            continue
        seen.add(rec["id"])
        records.append(rec)

    counts = Counter()
    if not ids:
        print(f"[fetch] Cleveland Museum of Art: {per_source} records...")
        records += fetch_cma_bulk(client, per_source, seen, skipped)
        print(f"[fetch] Art Institute of Chicago: {per_source} records...")
        records += fetch_aic_bulk(client, per_source, seen, skipped)

    for r in records:
        counts[r["institution"]] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": os.path.relpath(out_path, REPO_ROOT).replace(os.sep, "/"),
        "record_count": len(records),
        "by_institution": dict(counts),
        "seed_ids_requested": wanted_ids,
        "seed_ids_missing": missing,
        "skipped_by_reason": dict(skipped),
        "network_calls": client.network_calls,
        "cache_hits": client.cache_hits,
        "sources": {
            CMA_INSTITUTION: {"api": CMA_BASE, "license": "CC0 1.0 (metadata)",
                              "terms": "https://www.clevelandart.org/open-access"},
            AIC_INSTITUTION: {"api": AIC_BASE,
                              "license": "CC0 1.0, except `description`: CC BY 4.0",
                              "attribution": "Descriptions courtesy of the Art Institute of "
                                             "Chicago, licensed CC BY 4.0",
                              "terms": "https://www.artic.edu/terms"},
        },
    }
    if out_path == DEFAULT_OUT:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[fetch] Wrote {len(records)} records to {manifest['output']}  "
          f"({', '.join(f'{k}: {v}' for k, v in counts.items())})")
    if skipped:
        print("[fetch] Skipped (not written):")
        for reason, n in skipped.most_common():
            print(f"  {reason:<40} {n}")
    if missing:
        print(f"[fetch] {len(missing)} requested id(s) not found: {missing}")
    print(f"[fetch] {client.network_calls} network call(s), {client.cache_hits} cache hit(s)")
    return manifest


FIXTURE_PATH = os.path.join(REPO_ROOT, "src", "fixtures", "museum_fixture.jsonl")


def build_fixture(refresh=False):
    """The offline demo's corpus: the seed records, fetched from the public
    APIs and normalized by the real ingest code, so the fixture contains
    nothing that was typed by hand. Committed to the repo so fixture mode
    needs no network at all."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest import normalize_record

    raw_path = os.path.join(RAW_DIR, "fixture_base.jsonl")
    run(out_path=raw_path, ids=SEED_IDS, refresh=refresh)
    with open(raw_path, "r", encoding="utf-8") as f:
        rows = [normalize_record(json.loads(line)) for line in f if line.strip()]
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[fetch] Fixture: wrote {len(rows)} normalized records to "
          f"{os.path.relpath(FIXTURE_PATH, REPO_ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build data/base.jsonl from the CMA and AIC public APIs.")
    p.add_argument("--per-source", type=int, default=2500,
                   help="records per museum, excluding seeds (default 2500 -> ~5,000 total)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--ids", default=None,
                   help="comma-separated ids (cma-<n>,aic-<n>) to fetch instead of bulk pages")
    p.add_argument("--no-seeds", action="store_true", help="skip the demo-question seed records")
    p.add_argument("--refresh", action="store_true", help="ignore the data/raw/ page cache")
    p.add_argument("--fixture", action="store_true",
                   help="rebuild src/fixtures/museum_fixture.jsonl (normalized) from the seed ids")
    args = p.parse_args(argv)

    if args.fixture:
        try:
            build_fixture(refresh=args.refresh)
        except RuntimeError as e:
            print(f"[fetch] ERROR: {e}")
            return 1
        return 0

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    try:
        run(per_source=args.per_source, out_path=args.out, ids=ids,
            refresh=args.refresh, seeds=not args.no_seeds)
    except RuntimeError as e:
        print(f"[fetch] ERROR: {e}")
        print("[fetch] Pages fetched so far are cached in data/raw/ -- re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
