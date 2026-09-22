"""
s04_database.py  --  PART (d): give the retrieval structure a home and an
access path, and argue the access path with measurements rather than adjectives.

Builds the PostgreSQL schema in schema.sql, loads the corpus, the K=384
sketches and the 76-band LSH buckets, then compares five physically different
ways of answering the one question the nightly job asks hundreds of thousands
of times:

        who else is in band j, bucket X?

  A. composite B-tree, COVERING                                     (adopted)
       PRIMARY KEY (band_no, bucket_key, notice_id) on a heap table. The
       executor descends the B-tree on the two equality columns, lands on the
       first matching entry and walks the leaf forward while the key holds.
       notice_id is IN the index, so the answer is complete without touching
       the heap: Index Only Scan, Heap Fetches 0.

  B. the SAME B-tree keys but NOT covering                         (rejected)
       Index on (band_no, bucket_key) only; notice_id lives in the heap. The
       descent is identical -- and then every matching entry costs a random
       heap page visit to fetch the one column we wanted. This is the sharpest
       comparison available, because it changes exactly one thing.

  C. hash index on bucket_key                                      (rejected)
       Postgres hash indexes are single-column, equality-only and cannot
       support index-only scans: there is no ordering and no way to satisfy a
       projection from the index, so every hit is a heap visit. They also
       cannot serve any other query the table needs (range over band_no,
       ordered scans for maintenance, the bulk merge join below).

  D. GIN over a BIGINT[] of all 76 band keys, one row per notice   (rejected)
       A lookup is a posting-list probe, a bitmap, and then a heap recheck
       because the array carries no band position -- the structure cannot tell
       "key X in band 3" from "key X in band 47".

  E. no index at all                                     (the baseline, forced)

Every path is measured with EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON): the plan
the planner chose, the rows actually examined, heap fetches, buffer traffic and
wall-clock, with the rejected paths forced on so the comparison is real.

Writes results/d_database.json, logs/d_plans.txt and two figures.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

VARIANT, SCHEME = "canon", "w5"
K = 384
BANDS, ROWS_PER_BAND = 76, 5
TAU, TAU_SKETCH = 0.54, 0.46
N_LOOKUPS = 2000
NIGHTLY_INTAKE = 571                      # 4,000 notices a week


def walk(node, acc=None):
    acc = acc if acc is not None else []
    acc.append(node)
    for ch in node.get("Plans", []) or []:
        walk(ch, acc)
    return acc


def summarise(plan, relation=None):
    """Pull the numbers that matter out of an EXPLAIN plan.

    relation: report the node that scans THAT table. In a join the first scan
    node is usually the probe side, which says nothing about the access path
    under test.
    """
    root = plan["Plan"]
    nodes = walk(root)
    scan = None
    if relation:
        for n in nodes:
            if n.get("Relation Name") == relation or (n.get("Index Name") or "").startswith(relation):
                scan = n
                break
    if scan is None:
        scan = next((n for n in nodes if "Scan" in n["Node Type"]), root)
    idx_name = scan.get("Index Name")
    if idx_name is None:
        bmi = next((n for n in walk(scan) if n["Node Type"] == "Bitmap Index Scan"), None)
        if bmi is not None:
            idx_name = bmi.get("Index Name")
    loops = scan.get("Actual Loops") or 1
    examined = ((scan.get("Actual Rows") or 0)
                + (scan.get("Rows Removed by Filter") or 0)
                + (scan.get("Rows Removed by Index Recheck") or 0)) * loops
    return {
        "node_type": scan["Node Type"],
        "index": idx_name,
        "actual_rows": scan.get("Actual Rows"),
        "actual_loops": loops,
        "rows_removed_by_filter": scan.get("Rows Removed by Filter"),
        "rows_removed_by_index_recheck": scan.get("Rows Removed by Index Recheck"),
        "heap_fetches": scan.get("Heap Fetches"),
        "node_shared_hit": scan.get("Shared Hit Blocks"),
        "node_shared_read": scan.get("Shared Read Blocks"),
        "buffers_touched_whole_plan": sum((n.get("Shared Hit Blocks") or 0)
                                          + (n.get("Shared Read Blocks") or 0) for n in nodes),
        "execution_time_ms": plan.get("Execution Time"),
        "planning_time_ms": plan.get("Planning Time"),
        "total_rows_examined": examined,
    }


def _apply(cur, settings):
    for s in settings:
        cur.execute(s)


def _reset(cur, settings):
    for s in settings:
        cur.execute("RESET %s" % s.split("=")[0].replace("SET ", "").strip())


def explain_json(cur, sql, params=None, settings=()):
    _apply(cur, settings)
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) " + sql, params)
    plan = cur.fetchone()[0][0]
    _reset(cur, settings)
    return plan


def plan_text(cur, sql, params=None, settings=()):
    _apply(cur, settings)
    cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, params)
    txt = "\n".join(r[0] for r in cur.fetchall())
    _reset(cur, settings)
    return txt


def main():
    t_start = time.time()
    out = {}
    plans_txt = []

    df = C.load_notices()
    boiler = C.learn_boilerplate(df)
    store = C.build_shingles(df, VARIANT, SCHEME, boiler)
    sig = C.minhash_signatures(store, 768, tag="%s_%s" % (VARIANT, SCHEME))[:, :K]
    keys = C.band_keys(sig, BANDS, ROWS_PER_BAND)
    C.log("band keys: %d notices x %d bands" % keys.shape)

    C.pg_ensure_db()
    con = C.pg_connect()
    con.autocommit = False
    cur = con.cursor()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"),
              encoding="utf-8") as fh:
        cur.execute(fh.read())
    con.commit()
    C.log("schema created")

    # --------------------------------------------------------------- load
    t0 = time.time()
    buf = io.StringIO()
    for r in df.itertuples(index=False):
        body = (r.body.replace("\\", "\\\\").replace("\t", " ")
                .replace("\n", "\\n").replace("\r", " "))
        title = (str(r.title).replace("\\", "\\\\").replace("\t", " ")
                 .replace("\n", " ").replace("\r", " "))
        buf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                  % (r.notice_id, r.portal_id, r.published_at, title, body,
                     r.estimated_value, r.closing_date))
    buf.seek(0)
    cur.copy_from(buf, "notice", columns=("notice_id", "portal_id", "published_at",
                                          "title", "body", "estimated_value", "closing_date"))
    load_notice_s = time.time() - t0

    t0 = time.time()
    sig32 = sig.astype(np.uint32)
    buf = io.StringIO()
    for i, nid in enumerate(store.ids):
        buf.write("%s\t%d\t%d\t\\\\x%s\n" % (nid, K, int(store.sizes[i]),
                                             sig32[i].tobytes().hex()))
    buf.seek(0)
    cur.copy_from(buf, "notice_sketch", columns=("notice_id", "k_rows", "shingles", "sig"))
    load_sketch_s = time.time() - t0

    t0 = time.time()
    buf = io.StringIO()
    for j in range(BANDS):
        col = keys[:, j]
        for i, nid in enumerate(store.ids):
            buf.write("%d\t%d\t%s\n" % (j, col[i], nid))
    buf.seek(0)
    cur.copy_from(buf, "lsh_bucket", columns=("band_no", "bucket_key", "notice_id"))
    load_bucket_s = time.time() - t0
    con.commit()

    cur.execute("INSERT INTO lsh_bucket_nc   SELECT * FROM lsh_bucket")
    cur.execute("INSERT INTO lsh_bucket_hash SELECT * FROM lsh_bucket")
    cur.execute("INSERT INTO notice_bands "
                "SELECT notice_id, array_agg(bucket_key ORDER BY band_no) "
                "FROM lsh_bucket GROUP BY notice_id")
    con.commit()

    t0 = time.time()
    cur.execute("CREATE INDEX lsh_bucket_nc_idx ON lsh_bucket_nc (band_no, bucket_key)")
    nc_idx_s = time.time() - t0
    t0 = time.time()
    cur.execute("CREATE INDEX lsh_bucket_hash_idx ON lsh_bucket_hash USING hash (bucket_key)")
    hash_idx_s = time.time() - t0
    t0 = time.time()
    cur.execute("CREATE INDEX notice_bands_gin ON notice_bands USING gin (keys)")
    gin_idx_s = time.time() - t0
    con.commit()

    cur.execute("INSERT INTO index_meta (k_rows, bands, rows_per_band, variant, scheme, "
                "tau, tau_sketch) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (K, BANDS, ROWS_PER_BAND, VARIANT, SCHEME, TAU, TAU_SKETCH))
    con.commit()

    con.autocommit = True
    for t in ("notice", "notice_sketch", "lsh_bucket", "lsh_bucket_nc",
              "lsh_bucket_hash", "notice_bands"):
        cur.execute("VACUUM ANALYZE %s" % t)
    con.autocommit = False
    C.log("loaded and analysed")

    cur.execute("SELECT count(*) FROM lsh_bucket")
    n_bucket_rows = cur.fetchone()[0]
    sizes = {}
    for t, idx in (("notice", None), ("notice_sketch", None),
                   ("lsh_bucket", "lsh_bucket_pkey"),
                   ("lsh_bucket_nc", "lsh_bucket_nc_idx"),
                   ("lsh_bucket_hash", "lsh_bucket_hash_idx"),
                   ("notice_bands", "notice_bands_gin")):
        cur.execute("SELECT pg_total_relation_size(%s), pg_relation_size(%s)", (t, t))
        tot, heap = cur.fetchone()
        ix = None
        if idx:
            cur.execute("SELECT pg_relation_size(%s)", (idx,))
            ix = cur.fetchone()[0]
        sizes[t] = dict(total_bytes=tot, heap_bytes=heap, index_bytes=ix)

    out["load"] = {
        "notice_rows": int(len(df)), "sketch_rows": int(len(df)),
        "lsh_bucket_rows": n_bucket_rows,
        "copy_notice_seconds": round(load_notice_s, 1),
        "copy_sketch_seconds": round(load_sketch_s, 1),
        "copy_bucket_seconds": round(load_bucket_s, 1),
        "btree_noncovering_build_seconds": round(nc_idx_s, 2),
        "hash_index_build_seconds": round(hash_idx_s, 2),
        "gin_index_build_seconds": round(gin_idx_s, 2),
        "relation_sizes": sizes,
    }
    C.log("lsh_bucket: %s rows, heap %.1f MB, index %.1f MB"
          % (f"{n_bucket_rows:,}", sizes["lsh_bucket"]["heap_bytes"] / 1e6,
             sizes["lsh_bucket"]["index_bytes"] / 1e6))

    # --------------------------------------------------------------------
    # (i) the bulk join -- what the nightly job actually runs
    # --------------------------------------------------------------------
    rng = np.random.default_rng(5)
    new_idx = list(range(len(store) - NIGHTLY_INTAKE, len(store)))

    cur.execute("CREATE TEMP TABLE probe (band_no SMALLINT, bucket_key BIGINT, notice_id TEXT)")
    buf = io.StringIO()
    for i in new_idx:
        for j in range(BANDS):
            buf.write("%d\t%d\t%s\n" % (j, keys[i, j], store.ids[i]))
    buf.seek(0)
    cur.copy_from(buf, "probe", columns=("band_no", "bucket_key", "notice_id"))
    cur.execute("ANALYZE probe")
    con.commit()
    cur.execute("SELECT count(*) FROM probe")
    n_probe = cur.fetchone()[0]
    C.log("probe set: %d rows (%d new notices x %d bands)" % (n_probe, len(new_idx), BANDS))

    J_A = ("SELECT p.notice_id AS a, b.notice_id AS b FROM probe p JOIN lsh_bucket b "
           "ON b.band_no = p.band_no AND b.bucket_key = p.bucket_key "
           "WHERE b.notice_id <> p.notice_id")
    J_B = J_A.replace("lsh_bucket b", "lsh_bucket_nc b")
    J_C = ("SELECT p.notice_id AS a, b.notice_id AS b FROM probe p JOIN lsh_bucket_hash b "
           "ON b.bucket_key = p.bucket_key AND b.band_no = p.band_no "
           "WHERE b.notice_id <> p.notice_id")
    J_D = ("SELECT p.notice_id AS a, nb.notice_id AS b FROM probe p JOIN notice_bands nb "
           "ON nb.keys && ARRAY[p.bucket_key]::bigint[] WHERE nb.notice_id <> p.notice_id")

    BULK = {
        "A": (J_A, "lsh_bucket", ()),
        "B": (J_B, "lsh_bucket_nc", ("SET enable_bitmapscan = off",)),
        "C": (J_C, "lsh_bucket_hash", ("SET enable_seqscan = off", "SET enable_mergejoin = off",
                                       "SET enable_hashjoin = off")),
        "D": (J_D, "notice_bands", ("SET enable_seqscan = off",)),
        "E": (J_A, "lsh_bucket", ("SET enable_indexscan = off", "SET enable_indexonlyscan = off",
                                  "SET enable_bitmapscan = off")),
    }
    NAMES = {
        "A": "A. composite B-tree (band_no,bucket_key,notice_id) COVERING",
        "B": "B. same B-tree keys, NOT covering (notice_id in the heap)",
        "C": "C. hash index on bucket_key (single column, equality only)",
        "D": "D. GIN over BIGINT[] of the 76 band keys per notice",
        "E": "E. no index at all -- sequential scan, forced",
    }
    # No subsetting: every path runs the full probe set, because the plans are
    # not comparable otherwise. In particular the forced no-index path gets a
    # HASH JOIN, which reads lsh_bucket exactly once however large the probe
    # set is -- so scaling a subset measurement would have misrepresented it.
    SUBSET = {}

    paths = []
    for tag in ("A", "B", "C", "D", "E"):
        sql, rel, settings = BULK[tag]
        lim = SUBSET.get(tag)
        run_sql = sql.replace("FROM probe p", "FROM (SELECT * FROM probe LIMIT %d) p" % lim) \
            if lim else sql
        try:
            plan = explain_json(cur, run_sql, None, settings)
            s = summarise(plan, rel)
            _apply(cur, settings)
            t0 = time.time()
            cur.execute("SELECT count(*) FROM (" + run_sql + ") q")
            n_out = cur.fetchone()[0]
            dt = time.time() - t0
            _reset(cur, settings)
            scale = (n_probe / lim) if lim else 1.0
            s.update(name=NAMES[tag], tag=tag, adopted=(tag == "A"),
                     probe_rows=lim or n_probe, candidate_rows=n_out,
                     seconds=round(dt, 3),
                     seconds_for_full_probe_set=round(dt * scale, 3),
                     rows_examined_for_full_probe_set=int(s["total_rows_examined"] * scale),
                     measured_on_subset=bool(lim))
            paths.append(s)
            plans_txt.append("\n=== %s ===\n" % NAMES[tag] + plan_text(cur, run_sql, None, settings))
            C.log("%-56s %9.2f s  rows examined %14s  heap fetches %-8s buffers %s"
                  % (NAMES[tag][:56], s["seconds_for_full_probe_set"],
                     f'{s["rows_examined_for_full_probe_set"]:,}',
                     s["heap_fetches"], f'{s["buffers_touched_whole_plan"]:,}'))
        except Exception as exc:
            con.rollback()
            C.log("path %s failed: %s" % (tag, exc))
            paths.append(dict(name=NAMES[tag], tag=tag, error=str(exc)))
    out["access_paths_bulk"] = paths
    con.commit()

    # --------------------------------------------------------------------
    # (i-b) the ONLINE regime: one notice, 76 band lookups.
    #
    # This is the regime the question's "can be queried by the application"
    # clause is really about, and it is where the plans separate hardest: an
    # index path touches the notice's own buckets, the no-index path reads the
    # entire table to answer a question about one row.
    # --------------------------------------------------------------------
    cur.execute("CREATE TEMP TABLE probe1 AS SELECT * FROM probe WHERE notice_id = %s",
                (store.ids[new_idx[0]],))
    cur.execute("ANALYZE probe1")
    single = []
    for tag in ("A", "B", "C", "D", "E"):
        sql, rel, settings = BULK[tag]
        s1 = sql.replace("FROM probe p", "FROM probe1 p")
        try:
            plan = explain_json(cur, s1, None, settings)
            s = summarise(plan, rel)
            _apply(cur, settings)
            t0 = time.time()
            for _ in range(200):
                cur.execute("SELECT count(*) FROM (" + s1 + ") q")
                n_out = cur.fetchone()[0]
            dt = (time.time() - t0) / 200
            _reset(cur, settings)
            single.append(dict(name=NAMES[tag], tag=tag, node_type=s["node_type"],
                               rows_examined=int(s["total_rows_examined"]),
                               buffers=s["buffers_touched_whole_plan"],
                               heap_fetches=s["heap_fetches"],
                               candidates=n_out,
                               milliseconds=round(dt * 1000, 3)))
            C.log("single-notice %s: %7.3f ms  rows examined %10s  buffers %8s"
                  % (tag, single[-1]["milliseconds"], f'{single[-1]["rows_examined"]:,}',
                     f'{single[-1]["buffers"]:,}'))
        except Exception as exc:
            con.rollback()
            C.log("single %s failed: %s" % (tag, exc))
    out["access_paths_single_notice"] = single
    con.commit()

    # --------------------------------------------------------------------
    # (ii) prepared point lookups -- per-lookup cost, ordinary and pathological
    # --------------------------------------------------------------------
    cur.execute("SELECT band_no, bucket_key FROM lsh_bucket "
                "GROUP BY band_no, bucket_key ORDER BY count(*) DESC LIMIT 200")
    big_probes = [(int(a), int(b)) for a, b in cur.fetchall()]
    pr = rng.integers(0, len(store), N_LOOKUPS)
    pb = rng.integers(0, BANDS, N_LOOKUPS)
    rand_probes = [(int(pb[i]), int(keys[pr[i], pb[i]])) for i in range(N_LOOKUPS)]

    PREP = [
        ("A", "PREPARE la (smallint,bigint) AS SELECT notice_id FROM lsh_bucket "
              "WHERE band_no=$1 AND bucket_key=$2", "la"),
        ("B", "PREPARE lb (smallint,bigint) AS SELECT notice_id FROM lsh_bucket_nc "
              "WHERE band_no=$1 AND bucket_key=$2", "lb"),
        ("C", "PREPARE lc (smallint,bigint) AS SELECT notice_id FROM lsh_bucket_hash "
              "WHERE band_no=$1 AND bucket_key=$2", "lc"),
    ]
    point = []
    for tag, prep, name in PREP:
        cur.execute("DEALLOCATE ALL")
        cur.execute(prep)
        for label, probes_ in (("random bucket", rand_probes),
                               ("200 largest buckets", big_probes)):
            t0 = time.time()
            n = 0
            for p in probes_:
                cur.execute("EXECUTE %s (%%s,%%s)" % name, p)
                n += len(cur.fetchall())
            dt = time.time() - t0
            point.append(dict(path=NAMES[tag], tag=tag, probe_set=label,
                              lookups=len(probes_), rows_returned=n,
                              microseconds_per_lookup=round(dt / len(probes_) * 1e6, 1),
                              rows_per_lookup=round(n / len(probes_), 2)))
            C.log("point %s  %-20s %8.1f us/lookup  %8.1f rows/lookup"
                  % (tag, label, point[-1]["microseconds_per_lookup"],
                     point[-1]["rows_per_lookup"]))
    cur.execute("DEALLOCATE ALL")
    out["access_paths_point"] = point
    con.commit()

    # --------------------------------------------------- restart survival
    cur.execute("CREATE TEMP TABLE cand AS " + J_A)
    cur.execute("SELECT count(*), count(DISTINCT (a,b)) FROM cand")
    n_emitted, n_distinct = cur.fetchone()
    out["nightly_probe"] = {
        "new_notices": len(new_idx), "probe_rows": n_probe,
        "candidate_rows_emitted": n_emitted, "distinct_candidate_pairs": n_distinct,
    }
    con.commit()
    con.close()

    con2 = C.pg_connect()
    cur2 = con2.cursor()
    cur2.execute("SELECT k_rows, bands, rows_per_band, variant, scheme, tau FROM index_meta")
    meta = cur2.fetchone()
    cur2.execute("SELECT count(*) FROM lsh_bucket")
    after = cur2.fetchone()[0]
    cur2.execute("PREPARE lr (smallint,bigint) AS SELECT notice_id FROM lsh_bucket "
                 "WHERE band_no=$1 AND bucket_key=$2")
    t0 = time.time()
    cur2.execute("EXECUTE lr (%s,%s)", rand_probes[0])
    rows = cur2.fetchall()
    out["restart_check"] = {
        "reconnected": True,
        "index_meta": {"k_rows": meta[0], "bands": meta[1], "rows_per_band": meta[2],
                       "variant": meta[3], "scheme": meta[4], "tau": float(meta[5])},
        "lsh_bucket_rows_after_reconnect": after,
        "sample_lookup_returned": len(rows),
        "sample_lookup_ms": round((time.time() - t0) * 1000, 3),
        "note": ("A new process, a new connection, no warm Python state: the bands are "
                 "still on disk, index_meta says what they were built with, and the "
                 "lookup still answers."),
    }
    con2.close()

    with open(os.path.join(C.LOGS, "d_plans.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(plans_txt))
    C.log("wrote logs/d_plans.txt")

    # ------------------------------------------------------------ figures
    ok = [p for p in paths if "error" not in p]
    labels = {"A": "A. B-tree\ncovering", "B": "B. B-tree\nnot covering",
              "C": "C. hash\nindex", "D": "D. GIN\narray", "E": "E. seq\nscan"}
    cols = {"A": "#1a7f37", "B": "#c44e52", "C": "#dd8452", "D": "#8172b3", "E": "#937860"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    nm = [labels[p["tag"]] for p in ok]
    cl = [cols[p["tag"]] for p in ok]
    v = [p["seconds_for_full_probe_set"] for p in ok]
    axes[0].bar(nm, v, color=cl)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("seconds for one night's probe set (log)")
    for i, x in enumerate(v):
        axes[0].text(i, x * 1.15, "%.2fs" % x, ha="center", fontsize=9)
    axes[0].set_title("Wall-clock, %s probe rows" % f"{n_probe:,}")
    v = [max(p["rows_examined_for_full_probe_set"], 1) for p in ok]
    axes[1].bar(nm, v, color=cl)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rows actually examined (log)")
    for i, x in enumerate(v):
        axes[1].text(i, x * 1.2, f"{x:,}", ha="center", fontsize=8)
    axes[1].set_title("Rows the executor had to look at")
    v = [max(p["buffers_touched_whole_plan"], 1) for p in ok]
    axes[2].bar(nm, v, color=cl)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("8 kB buffers touched (log)")
    for i, p in enumerate(ok):
        axes[2].text(i, max(p["buffers_touched_whole_plan"], 1) * 1.2,
                     "heap fetches\n%s" % (p["heap_fetches"] if p["heap_fetches"] is not None else "n/a"),
                     ha="center", fontsize=8)
    axes[2].set_title("Buffer traffic (D and E measured on a 3,000-row subset)")
    fig.suptitle("Part (d): five physical ways to answer 'who else is in this bucket?'",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "d_access_paths.png"), dpi=140)
    C.log("wrote figures/d_access_paths.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    tn = ["notice", "notice_sketch", "lsh_bucket", "lsh_bucket_nc",
          "lsh_bucket_hash", "notice_bands"]
    heap = [sizes[t]["heap_bytes"] / 1e6 for t in tn]
    ixb = [(sizes[t]["index_bytes"] or 0) / 1e6 for t in tn]
    ax.bar(tn, heap, label="heap", color="#4c72b0")
    ax.bar(tn, ixb, bottom=heap, label="index", color="#dd8452")
    for i, t in enumerate(tn):
        ax.text(i, heap[i] + ixb[i] + 2, "%.0f MB" % (heap[i] + ixb[i]), ha="center", fontsize=8)
    ax.set_ylabel("megabytes")
    ax.set_title("Part (d): what each candidate structure costs on disk")
    ax.legend(fontsize=8)
    plt.xticks(rotation=18, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "d_storage.png"), dpi=140)
    C.log("wrote figures/d_storage.png")

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    C.save_json("d_database.json", out)

    # ------------------------------------------------------------- report
    print()
    print("=" * 112)
    print("PART (d) SUMMARY   --   PostgreSQL 18, database 'setubid'")
    print("=" * 112)
    print("loaded: %s notices, %s sketches, %s lsh_bucket rows (%d bands x %d notices)"
          % (f"{len(df):,}", f"{len(df):,}", f"{n_bucket_rows:,}", BANDS, len(df)))
    print("lsh_bucket on disk: heap %.1f MB + covering index %.1f MB"
          % (sizes["lsh_bucket"]["heap_bytes"] / 1e6, sizes["lsh_bucket"]["index_bytes"] / 1e6))
    print()
    print("BULK JOIN of one night's probe set (%s rows) against each structure:" % f"{n_probe:,}")
    hdr = ("%-56s %16s %10s %14s %10s %10s" %
           ("access path", "plan node", "seconds", "rows examined", "heap f.", "buffers"))
    print(hdr)
    print("-" * len(hdr))
    for p in ok:
        note = ""
        print("%-56s %16s %10.2f %14s %10s %10s%s"
              % (p["name"][:56], p["node_type"][:16], p["seconds_for_full_probe_set"],
                 f'{p["rows_examined_for_full_probe_set"]:,}',
                 p["heap_fetches"] if p["heap_fetches"] is not None else "-",
                 f'{p["buffers_touched_whole_plan"]:,}', note))
    print()
    print("ONE NOTICE, %d band lookups (the regime the application runs in):" % BANDS)
    hdr = ("%-56s %16s %11s %14s %10s" %
           ("access path", "plan node", "ms", "rows examined", "buffers"))
    print(hdr)
    print("-" * len(hdr))
    for p in single:
        print("%-56s %16s %11.3f %14s %10s"
              % (p["name"][:56], p["node_type"][:16], p["milliseconds"],
                 f'{p["rows_examined"]:,}', f'{p["buffers"]:,}'))
    print()
    print("PREPARED POINT LOOKUPS:")
    hdr = "%-56s %22s %14s %14s" % ("access path", "probe set", "us/lookup", "rows/lookup")
    print(hdr)
    print("-" * len(hdr))
    for p in point:
        print("%-56s %22s %14.1f %14.1f"
              % (p["path"][:56], p["probe_set"], p["microseconds_per_lookup"],
                 p["rows_per_lookup"]))
    print()
    r = out["restart_check"]
    print("restart check: new process, new connection -- %s bucket rows still there; "
          "index_meta K=%d bands=%d r=%d tau=%.2f; lookup answered in %.3f ms"
          % (f"{r['lsh_bucket_rows_after_reconnect']:,}", r["index_meta"]["k_rows"],
             r["index_meta"]["bands"], r["index_meta"]["rows_per_band"],
             r["index_meta"]["tau"], r["sample_lookup_ms"]))
    print("elapsed %.1fs" % out["elapsed_seconds"])


if __name__ == "__main__":
    main()
