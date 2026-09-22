"""
common.py -- shared machinery for SetuBid near-duplicate detection (Question 2).

Everything that more than one stage needs lives here:

  * corpus loading                     load_notices()
  * per-portal boilerplate learning    learn_boilerplate()
  * text normalisation variants        normalise()          [part (a), decision 2]
  * shingling variants                 shingle()            [part (a), decision 1]
  * exact Jaccard                      jaccard_exact()
  * MinHash sketching                  minhash_signatures() [part (b)]
  * banded LSH keys                    band_keys()          [part (c)]

Design notes that matter for the report:

  - A "notice" is reduced to a SET of 64-bit shingle hashes. Similarity is the
    Jaccard coefficient of two such sets. Everything downstream (sketching,
    banding, thresholds) is defined with respect to that one number.

  - Hashing is deterministic across process restarts (no PYTHONHASHSEED
    dependence, no random.seed() at import). That is a hard requirement: the
    LSH buckets are persisted in PostgreSQL and must still be meaningful
    tomorrow night. Every hash below is a pure function of its input bytes
    plus a fixed integer seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../dmw/q2
DATA = os.path.join(os.path.dirname(ROOT), "data_2")
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
LOGS = os.path.join(ROOT, "logs")
for _d in (CACHE, RESULTS, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)

U64 = np.uint64


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def save_json(name, obj):
    p = os.path.join(RESULTS, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    log("wrote " + p)
    return p


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def load_notices():
    """The 12,000 notices, in a stable order (sorted by notice_id)."""
    import glob
    parts = sorted(glob.glob(os.path.join(DATA, "notices", "part-*.csv")))
    if not parts:
        raise FileNotFoundError("no notice parts under %s/notices" % DATA)
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.sort_values("notice_id").reset_index(drop=True)
    df["body"] = df["body"].fillna("")
    df["title"] = df["title"].fillna("")
    return df


def load_labels():
    return pd.read_csv(os.path.join(DATA, "labelled_pairs.csv"))


# --------------------------------------------------------------------------
# hashing primitives (deterministic, restart-stable)
# --------------------------------------------------------------------------
_M1 = U64(0xBF58476D1CE4E5B9)
_M2 = U64(0x94D049BB133111EB)
_GOLD = U64(0x9E3779B97F4A7C15)
_S30, _S27, _S31, _S1 = U64(30), U64(27), U64(31), U64(1)


def splitmix64(x):
    """Vectorised splitmix64 finaliser. Strong avalanche, uint64 in/out."""
    x = x.astype(U64, copy=True)
    x ^= x >> _S30
    x *= _M1
    x ^= x >> _S27
    x *= _M2
    x ^= x >> _S31
    return x


def _stable_token_id(tok):
    """Deterministic 64-bit id for a token (blake2b, fixed key)."""
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")


class _TokenTable:
    """Memoised token -> uint64 map so we hash each distinct word once."""

    def __init__(self):
        self._d = {}

    def ids(self, toks):
        d = self._d
        out = np.empty(len(toks), dtype=U64)
        for i, t in enumerate(toks):
            v = d.get(t)
            if v is None:
                v = _stable_token_id(t)
                d[t] = v
            out[i] = v
        return out


_TOKENS = _TokenTable()


# --------------------------------------------------------------------------
# normalisation  -- part (a), decision 2: what is signal and what is noise
# --------------------------------------------------------------------------
_RE_WS = re.compile(r"[ \t]+")
_RE_MULTINL = re.compile(r"\n{2,}")

_RE_REF_LABELLED = re.compile(
    r"(reference number|ref\.? no\.?|tender (?:reference|id|no\.?)|nit no\.?|bid number)"
    r"\s*[:\-]?\s*\S+", re.I)
_RE_REF_SHAPE = re.compile(
    r"\b(?:[a-z]{2,6}[-/][0-9]{2,4}(?:[-/][0-9a-z]{1,6}){1,3}"
    r"|[a-z]{2,6}[-/][0-9]{4,}"
    r"|[0-9]{2,4}[-/][0-9]{2,6}[-/][0-9]{2,6})\b", re.I)

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_RE_DATE = re.compile(
    r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-\s]?(?:" + _MONTHS + r")[a-z]*[-,\s]+\d{2,4}"
    r"|(?:" + _MONTHS + r")[a-z]*[-\s]+\d{1,2}\s*,?\s*\d{2,4})\b", re.I)

_RE_MONEY_CR = re.compile(r"(?:rs\.?|inr|rupees)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr\.?|crore[s]?)\b", re.I)
_RE_MONEY_LAKH = re.compile(r"(?:rs\.?|inr|rupees)?\s*([\d,]+(?:\.\d+)?)\s*(?:lakh[s]?|lac[s]?)\b", re.I)
_RE_MONEY_PLAIN = re.compile(r"\b(?:rs\.?|inr|rupees)\s*([\d,]+(?:\.\d+)?)\s*(?:/-|only)?", re.I)
_RE_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_RE_PUNCT = re.compile(r"[^\w\s<>]+")


def _money_token(amount):
    """Canonical token for a rupee amount, rounded to 3 significant figures.

    'Rs. 4,50,00,000/-', 'Rs. 450.00 lakh', 'INR 4.500 Cr' and '45000000' all
    collapse to the same token; two tenders whose values differ by more than
    ~0.5% do not.
    """
    if amount <= 0:
        return "<money>"
    mag = int(math.floor(math.log10(amount)))
    if mag >= 2:
        sig = int(round(amount / (10 ** (mag - 2))))
        return "<money_%de%d>" % (sig, mag - 2)
    return "<money_%d>" % int(round(amount))


def _canon_money(text):
    def cr(m):
        try:
            return " " + _money_token(float(m.group(1).replace(",", "")) * 1e7) + " "
        except ValueError:
            return " <money> "

    def lakh(m):
        try:
            return " " + _money_token(float(m.group(1).replace(",", "")) * 1e5) + " "
        except ValueError:
            return " <money> "

    def plain(m):
        try:
            return " " + _money_token(float(m.group(1).replace(",", ""))) + " "
        except ValueError:
            return " <money> "

    text = _RE_MONEY_CR.sub(cr, text)
    text = _RE_MONEY_LAKH.sub(lakh, text)
    text = _RE_MONEY_PLAIN.sub(plain, text)
    return text


def _line_key(line):
    """Boilerplate identity of a line: case- and punctuation-insensitive, but
    DIGIT-SENSITIVE.

    Digits are deliberately kept. An earlier version normalised numbers away,
    which made 'Estimated cost put to tender: Rs. 7,60,00,000/-' look identical
    across every notice on the portal and deleted the single most discriminating
    line in the document. A line only counts as template if it repeats
    verbatim, values and all.
    """
    k = _RE_PUNCT.sub(" ", line.lower())
    return _RE_WS.sub(" ", k).strip()


def normalise(text, variant, boiler=None):
    """Turn raw body text into the string that will be shingled.

    variant:
      'raw'   -- lower-case + whitespace collapse only. Keeps every digit,
                 every reference number and all portal boilerplate.
      'mask'  -- raw + boilerplate lines removed + dates/refs/ALL numbers
                 replaced by a single placeholder each.
      'canon' -- raw + boilerplate lines removed + refs and dates masked, but
                 monetary amounts canonicalised to a value token, so the
                 *amount* survives while its *format* does not.
    """
    t = text.lower()

    if variant != "raw" and boiler:
        keep = [ln for ln in t.split("\n") if _line_key(ln) not in boiler]
        t = "\n".join(keep)

    if variant in ("mask", "canon"):
        t = _RE_REF_LABELLED.sub(" <ref> ", t)
        t = _RE_DATE.sub(" <date> ", t)
        if variant == "canon":
            t = _canon_money(t)
        t = _RE_REF_SHAPE.sub(" <ref> ", t)
        t = _RE_NUM.sub(" <num> ", t)

    t = _RE_PUNCT.sub(" ", t)
    t = _RE_WS.sub(" ", t)
    t = _RE_MULTINL.sub("\n", t)
    return t.strip()


def learn_boilerplate(df, min_notices=20, frac=0.60, min_len=25):
    """Learn, per portal, which lines are template rather than content.

    Mechanical rule: a line whose normalised form appears in more than `frac`
    of that portal's notices carries no information *within* that portal, so it
    cannot help separate two notices from it -- and across portals it actively
    hurts, because it is the thing that makes 'everything on P001 look like
    everything else on P001'. Nothing about any specific portal is hard-coded;
    the rule is run over the corpus and the blocks fall out.
    """
    per_portal_lines = defaultdict(Counter)
    per_portal_n = Counter()
    for portal, body in zip(df.portal_id.values, df.body.values):
        per_portal_n[portal] += 1
        seen = set()
        for ln in body.split("\n"):
            k = _line_key(ln)
            if len(k) >= min_len and k not in seen:
                seen.add(k)
                per_portal_lines[portal][k] += 1

    boiler = {}
    for portal, cnt in per_portal_lines.items():
        n = per_portal_n[portal]
        if n < min_notices:
            boiler[portal] = set()
            continue
        boiler[portal] = set(k for k, c in cnt.items() if c / n > frac)
    return boiler


# --------------------------------------------------------------------------
# shingling -- part (a), decision 1: how finely the text is decomposed
# --------------------------------------------------------------------------
def shingle(text, scheme):
    """Return the notice's shingle set as a sorted array of unique uint64.

    scheme: 'w3' | 'w5' | 'c5' | 'c9'  (word k-grams / character k-grams)
    """
    if scheme.startswith("w"):
        k = int(scheme[1:])
        toks = text.split()
        if len(toks) < k:
            toks = toks + ["<pad>"] * (k - len(toks))
        ids = _TOKENS.ids(toks)
        n = len(ids) - k + 1
        if n <= 0:
            return np.zeros(0, dtype=U64)
        acc = np.zeros(n, dtype=U64)
        for j in range(k):
            acc = acc * _GOLD + ids[j:j + n]
        h = splitmix64(acc)
    else:
        k = int(scheme[1:])
        b = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(U64)
        n = len(b) - k + 1
        if n <= 0:
            return np.zeros(0, dtype=U64)
        acc = np.zeros(n, dtype=U64)
        for j in range(k):
            acc = acc * _GOLD + b[j:j + n]
        h = splitmix64(acc)
    return np.unique(h)


class PostingIndex:
    """Inverted index over shingle values, for exact all-against-one Jaccard.

    Lets us compute the exact Jaccard of one notice against all 12,000 in a
    couple of milliseconds, which is what makes an UNBIASED measurement of
    retrieval recall possible: pick anchor notices at random, score them
    against the entire corpus, and there is no sampler in the loop to bias
    which pairs we look at.
    """

    def __init__(self, store):
        n = len(store)
        doc = np.repeat(np.arange(n, dtype=np.int32), np.diff(store.offs))
        order = np.argsort(store.flat, kind="stable")
        self.sf = store.flat[order]
        self.sd = doc[order]
        self.sizes = np.diff(store.offs).astype(np.int64)
        self.n = n

    def jaccard_against_all(self, a_shingles, a_size=None):
        """Exact Jaccard of one shingle set against every notice. Returns (n,)."""
        if a_size is None:
            a_size = len(a_shingles)
        if a_size == 0:
            return np.zeros(self.n)
        lo = np.searchsorted(self.sf, a_shingles, "left")
        hi = np.searchsorted(self.sf, a_shingles, "right")
        take = hi - lo
        if take.sum() == 0:
            return np.zeros(self.n)
        idx = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        inter = np.bincount(self.sd[idx], minlength=self.n).astype(np.int64)
        union = a_size + self.sizes - inter
        return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def jaccard_exact(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(len(a) + len(b) - inter)


# --------------------------------------------------------------------------
# a whole-corpus shingle store (ragged, cached on disk)
# --------------------------------------------------------------------------
@dataclass
class ShingleStore:
    ids: list
    flat: np.ndarray
    offs: np.ndarray

    def __getitem__(self, i):
        return self.flat[self.offs[i]:self.offs[i + 1]]

    def __len__(self):
        return len(self.ids)

    @property
    def sizes(self):
        return np.diff(self.offs)


def build_shingles(df, variant, scheme, boiler=None, use_cache=True, df_stop=None):
    """Shingle the whole corpus under one (normalisation, granularity) choice.

    df_stop: if set, drop shingles whose corpus document-frequency exceeds this
             fraction (the part (e) mitigation). None = keep everything.
    """
    tag = "sh_%s_%s" % (variant, scheme) + ("_df%s" % df_stop if df_stop else "")
    cpath = os.path.join(CACHE, tag + ".npz")
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return ShingleStore(list(z["ids"]), z["flat"], z["offs"])

    if boiler is None and variant != "raw":
        boiler = learn_boilerplate(df)

    t0 = time.time()
    chunks, offs = [], [0]
    for portal, body, title in zip(df.portal_id.values, df.body.values, df.title.values):
        bset = boiler.get(portal, set()) if boiler else None
        txt = normalise(title + "\n" + body, variant, bset)
        s = shingle(txt, scheme)
        chunks.append(s)
        offs.append(offs[-1] + len(s))
    flat = np.concatenate(chunks) if chunks else np.zeros(0, U64)
    offs = np.asarray(offs, dtype=np.int64)
    store = ShingleStore(list(df.notice_id.values), flat, offs)
    log("shingled %d notices [%s/%s] mean |S|=%.0f in %.1fs"
        % (len(store), variant, scheme, store.sizes.mean(), time.time() - t0))

    if df_stop is not None:
        store = apply_df_stoplist(store, df_stop)

    if use_cache:
        np.savez_compressed(cpath, ids=np.array(store.ids, dtype=object),
                            flat=store.flat, offs=store.offs)
    return store


def apply_df_stoplist(store, max_df):
    """Remove shingles that occur in more than `max_df` of all notices."""
    n = len(store)
    uniq, counts = np.unique(store.flat, return_counts=True)
    stop = uniq[counts > max_df * n]
    if stop.size == 0:
        return store
    keep_mask = ~np.isin(store.flat, stop)
    new_chunks, new_offs = [], [0]
    for i in range(n):
        lo, hi = store.offs[i], store.offs[i + 1]
        seg = store.flat[lo:hi][keep_mask[lo:hi]]
        new_chunks.append(seg)
        new_offs.append(new_offs[-1] + len(seg))
    kept = sum(len(c) for c in new_chunks)
    log("df-stoplist max_df=%s: removed %d shingle types, %.1f%% of postings"
        % (max_df, stop.size, (1 - kept / max(len(store.flat), 1)) * 100))
    return ShingleStore(store.ids, np.concatenate(new_chunks),
                        np.asarray(new_offs, dtype=np.int64))


# --------------------------------------------------------------------------
# MinHash -- part (b)
# --------------------------------------------------------------------------
def minhash_signatures(store, K, seed=20240917, use_cache=True, tag=""):
    """(N, K) uint64 matrix of MinHash values.

    Permutation i is the map  x -> splitmix64(x + c_i)  with c_i a fixed
    64-bit constant derived from `seed`. Each is a strongly-mixing bijection
    of the 64-bit universe, so the min over a set is a MinHash under the usual
    idealisation, and P[sig_i(A) == sig_i(B)] = J(A, B).
    """
    cpath = os.path.join(CACHE, "sig_%s_K%d_s%d.npy" % (tag, K, seed))
    if use_cache and tag and os.path.exists(cpath):
        return np.load(cpath)

    rng = np.random.default_rng(seed)
    consts = rng.integers(1, 2 ** 63 - 1, size=K, dtype=np.int64).astype(U64)
    N = len(store)
    sig = np.full((N, K), np.iinfo(np.uint64).max, dtype=U64)
    t0 = time.time()
    for i in range(N):
        s = store[i]
        if s.size == 0:
            continue
        m = splitmix64(s[None, :] + consts[:, None])
        sig[i] = m.min(axis=1)
    log("minhash K=%d for %d notices in %.1fs" % (K, N, time.time() - t0))
    if use_cache and tag:
        np.save(cpath, sig)
    return sig


def jaccard_sketch(sig_a, sig_b):
    """Estimated Jaccard from two signature matrices (row-wise)."""
    if sig_a.ndim == 1:
        return float((sig_a == sig_b).mean())
    return (sig_a == sig_b).mean(axis=1)


# --------------------------------------------------------------------------
# banded LSH -- part (c)
# --------------------------------------------------------------------------
def band_keys(sig, b, r):
    """(N, b) int64 band keys. Row i, band j is a hash of the signature slice
    [j*r : (j+1)*r]. Signed int64 so it drops straight into a Postgres BIGINT."""
    N, K = sig.shape
    assert b * r <= K, "b*r=%d exceeds K=%d" % (b * r, K)
    out = np.empty((N, b), dtype=np.int64)
    for j in range(b):
        seg = sig[:, j * r:(j + 1) * r]
        acc = np.full(N, U64(j + 1) * _GOLD, dtype=U64)
        for c in range(r):
            acc = splitmix64(acc ^ seg[:, c]) * _GOLD + U64(c + 1)
        out[:, j] = (splitmix64(acc) >> _S1).astype(np.int64)
    return out


def lsh_pairs_from_bands(keys, cap=None):
    """Candidate pairs (i<j) from band keys -- in-memory reference impl.

    Returns (pairs, bucket_sizes, (skipped_buckets, skipped_members)).
    `cap` drops any bucket with more than `cap` members (part (e) mitigation).
    """
    N, b = keys.shape
    pairs = set()
    sizes = []
    skipped = 0
    skipped_members = 0
    for j in range(b):
        order = np.argsort(keys[:, j], kind="stable")
        k = keys[order, j]
        start = 0
        for e in range(1, len(k) + 1):
            if e == len(k) or k[e] != k[start]:
                mem = order[start:e]
                if len(mem) > 1:
                    sizes.append(len(mem))
                    if cap is not None and len(mem) > cap:
                        skipped += 1
                        skipped_members += len(mem)
                    else:
                        mem = np.sort(mem)
                        for x in range(len(mem)):
                            ix = int(mem[x])
                            for y in range(x + 1, len(mem)):
                                pairs.add((ix, int(mem[y])))
                start = e
    return pairs, np.asarray(sizes), (skipped, skipped_members)


# --------------------------------------------------------------------------
# LSH theory helpers
# --------------------------------------------------------------------------
def p_candidate(j, b, r):
    j = np.asarray(j, dtype=float)
    return 1.0 - (1.0 - j ** r) ** b


def lsh_threshold(b, r):
    return (1.0 / b) ** (1.0 / r)


# --------------------------------------------------------------------------
# PostgreSQL
# --------------------------------------------------------------------------
PG = dict(host=os.environ.get("PGHOST", "localhost"),
          port=int(os.environ.get("PGPORT", 5432)),
          user=os.environ.get("PGUSER", "postgres"),
          password=os.environ.get("PGPASSWORD", "postgres"),
          dbname=os.environ.get("PGDATABASE", "setubid"))


def pg_connect(dbname=None):
    import psycopg2
    cfg = dict(PG)
    if dbname:
        cfg["dbname"] = dbname
    return psycopg2.connect(**cfg)


def pg_ensure_db():
    con = pg_connect(dbname="postgres")
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (PG["dbname"],))
        if not cur.fetchone():
            cur.execute('CREATE DATABASE "%s"' % PG["dbname"])
            log("created database " + PG["dbname"])
    con.close()
