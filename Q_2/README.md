# Question 2 — SetuBid near-duplicate detection

Near-duplicate detection over 12,000 procurement notices from 260 portals,
built to finish inside a 20-minute nightly window on one machine.

**Read [`REPORT.md`](REPORT.md) first** — it is the answer. This file is only
about running the code.

## What it does

```
notice text
   ↓  normalise      learned portal boilerplate removed, refs/dates masked,
   ↓                 monetary amounts canonicalised to their value
   ↓  shingle        word 5-grams → set of 64-bit hashes
   ↓  sketch         K = 384 MinHash rows, 1,536 bytes/notice
   ↓  band           76 bands × 5 rows → PostgreSQL lsh_bucket, buckets capped at 100
   ↓  probe          covering composite B-tree, Index Only Scan, Heap Fetches 0
   ↓  filter         sketch estimate ≥ 0.46
   ↓  verify         exact Jaccard re-derived from the stored body
   ↓  merge          J ≥ 0.54, complete linkage
opportunity cards with ids that never move
```

## Requirements

- Python 3.10+ with `numpy`, `pandas`, `matplotlib`, `psycopg2`
- PostgreSQL 14+ running locally

Connection settings come from the usual environment variables and default to
`localhost:5432`, user `postgres`, password `postgres`, database `setubid`
(created automatically):

```bash
export PGHOST=localhost PGPORT=5432 PGUSER=postgres PGPASSWORD=postgres PGDATABASE=setubid
```

The corpus is expected at `../data_2/` relative to this directory —
`notices/part-*.csv`, `labelled_pairs.csv`, `portal_profiles.md`.

## Running it

```bash
python src/run_all.py
```

Runs all six stages in order, writes `logs/*.log`, `results/*.json`,
`figures/*.png`, and renders `screenshots/*.png`. About 6–7 minutes cold, most
of it the one-off K = 768 signature cache that every later stage reuses.

Individual stages:

```bash
python src/run_all.py a        # part (a)  what "similar" means
python src/run_all.py b        # part (b)  the size of the reduced form
python src/run_all.py c        # part (c)  sublinear retrieval, risk priced
python src/run_all.py d        # part (d)  schema and access path  [needs PostgreSQL]
python src/run_all.py e        # part (e)  the skew and what mitigating it costs
python src/run_all.py f        # the nightly job end to end        [needs PostgreSQL]
```

Stages b–f reuse caches written by earlier stages, so run `b` before `c`, `e`
or `f` on a clean checkout. Deleting `cache/` forces a full recomputation.

Stages `d` and `f` both drop and recreate the schema, so run them one at a
time. After `f`, the database holds the state the report describes; the psql
evidence in `screenshots/11_psql_session.png` was captured from it.

## Where things are

| | |
|---|---|
| the answer | `REPORT.md` |
| the K = 384 derivation | docstring of `src/s02_sketch_size.py` (written before the measurement) |
| the schema | `src/schema.sql` |
| query plans | `logs/d_plans.txt` |
| the psql evidence queries | `src/evidence_queries.sql` |
| every number, machine-readable | `results/*.json` |

## A note on `data_2/_truth/`

The corpus ships with generator ground truth. It is deliberately unused: the
question states `labelled_pairs.csv` is the only trustworthy label source, so
no tuning decision and no reported metric consults it. Quality numbers come
from the 900 adjudicated pairs or from label-free unbiased sampling.
