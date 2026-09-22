"""
make_screenshots.py -- render the captured console output of each stage as a
PNG, so the evidence can be pasted into a report.

These are not photographs of a terminal: they are the actual stdout of the
runs in logs/*.log and logs/d_plans.txt, rendered with a monospace font. The
text is byte-for-byte what the scripts printed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import common as C

SHOTS = os.path.join(C.ROOT, "screenshots")
os.makedirs(SHOTS, exist_ok=True)

BG = "#12161c"
FG = "#d7dde5"
ACCENT = "#7ee787"
DIM = "#8b949e"


def render(text, out_name, title, max_cols=118):
    lines = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        while len(ln) > max_cols:
            lines.append(ln[:max_cols])
            ln = "    " + ln[max_cols:]
        lines.append(ln)
    while lines and not lines[-1].strip():
        lines.pop()
    n = len(lines)
    FS, LS = 7.6, 1.32
    line_in = FS * LS / 72.0
    fig_h = max(2.0, n * line_in + 0.62)
    fig_w = 0.0815 * max_cols + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.012, 0.995, title, color=ACCENT, fontsize=10, va="top", ha="left",
            family="monospace", weight="bold", transform=ax.transAxes)
    body = "\n".join(lines)
    ax.text(0.012, 1.0 - 0.40 / fig_h, body, color=FG, fontsize=FS, va="top",
            ha="left", family="monospace", transform=ax.transAxes, linespacing=LS)
    p = os.path.join(SHOTS, out_name)
    _save_with_retry(fig, p)
    plt.close(fig)
    print("wrote %s  (%d lines)" % (p, n))


def _save_with_retry(fig, path, attempts=6):
    """Write the PNG, retrying briefly.

    On Windows a freshly written file in a scanned directory can still be held
    open when the next write lands, which surfaces as OSError 22. Writing to a
    buffer and replacing the file is both atomic and retryable.
    """
    import io as _io
    import time as _time
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=155, facecolor=BG)
    data = buf.getvalue()
    last = None
    for i in range(attempts):
        try:
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last = exc
            _time.sleep(0.25 * (i + 1))
    raise last


def tail_from(path, marker, extra_head=0):
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    i = txt.find(marker)
    if i < 0:
        return txt[-6000:]
    j = max(0, txt.rfind("\n", 0, i))
    return txt[j:].strip("\n")


def head_of(path, n_lines):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "\n".join(fh.read().split("\n")[:n_lines])


def section(path, start_marker, end_marker=None, limit=90):
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    i = txt.find(start_marker)
    if i < 0:
        return ""
    rest = txt[i:]
    if end_marker:
        j = rest.find(end_marker, len(start_marker))
        if j > 0:
            rest = rest[:j]
    return "\n".join(rest.split("\n")[:limit]).rstrip()


def main():
    L = C.LOGS
    jobs = [
        ("s01_similarity.log", "PART (a) SUMMARY",
         "01_part_a_similarity.png",
         "$ python src/s01_similarity.py     # PART (a): what 'similar' means"),
        ("s02_sketch_size.log", "PART (b) SUMMARY",
         "02_part_b_sketch_size.png",
         "$ python src/s02_sketch_size.py    # PART (b): K fixed by argument, then measured"),
        ("s03_lsh_tuning.log", "PART (c) SUMMARY",
         "03_part_c_lsh_tuning.png",
         "$ python src/s03_lsh_tuning.py     # PART (c): sublinear retrieval, risk priced"),
        ("s04_database.log", "PART (d) SUMMARY",
         "04_part_d_database.png",
         "$ python src/s04_database.py       # PART (d): schema and access path"),
        ("s05_skew.log", "PART (e) SUMMARY",
         "05_part_e_skew.png",
         "$ python src/s05_skew.py           # PART (e): where the design betrays you"),
        ("s06_nightly.log", "NIGHTLY PIPELINE",
         "06_nightly_pipeline.png",
         "$ python src/s06_nightly.py        # the system as a nightly job"),
    ]
    for log, marker, out, title in jobs:
        p = os.path.join(L, log)
        if not os.path.exists(p):
            print("missing " + p)
            continue
        render(tail_from(p, marker), out, title)

    # the query plans are the evidence for part (d); give them their own shots
    plans = os.path.join(L, "d_plans.txt")
    if os.path.exists(plans):
        render(section(plans, "=== A.", "=== B.", limit=40),
               "07_plan_A_covering_btree.png",
               "$ EXPLAIN (ANALYZE, BUFFERS)   # A: covering composite B-tree (ADOPTED)")
        render(section(plans, "=== B.", "=== C.", limit=40),
               "08_plan_B_noncovering.png",
               "$ EXPLAIN (ANALYZE, BUFFERS)   # B: same keys, NOT covering (rejected)")
        render(section(plans, "=== C.", "=== D.", limit=40),
               "09_plan_C_hash_index.png",
               "$ EXPLAIN (ANALYZE, BUFFERS)   # C: hash index on bucket_key (rejected)")
        render(section(plans, "=== E.", None, limit=40),
               "10_plan_E_seq_scan.png",
               "$ EXPLAIN (ANALYZE, BUFFERS)   # E: no index, forced (baseline)")

    # a live psql session: the schema exists, has rows, and answers
    sqlout = os.path.join(L, "psql_session.txt")
    if os.path.exists(sqlout):
        render(open(sqlout, encoding="utf-8", errors="replace").read(),
               "11_psql_session.png",
               "$ psql -U postgres -d setubid    # the index is relational data, "
               "and it survived the job")


if __name__ == "__main__":
    main()
