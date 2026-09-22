"""
run_all.py -- run every stage in order, capturing each stage's console output
to logs/, then render the screenshots.

    python src/run_all.py            # everything
    python src/run_all.py a b c      # only those stages

Stage 02 builds the K=768 signature cache (about 70 s the first time) which
every later stage reuses, so the first full run is slower than the numbers each
stage reports for itself.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)

STAGES = [
    ("a", "s01_similarity.py", "part (a): what 'similar' means"),
    ("b", "s02_sketch_size.py", "part (b): the size of the reduced form"),
    ("c", "s03_lsh_tuning.py", "part (c): sublinear retrieval, risk priced"),
    ("d", "s04_database.py", "part (d): schema and access path"),
    ("e", "s05_skew.py", "part (e): the skew, and what mitigating it costs"),
    ("f", "s06_nightly.py", "the system as a nightly job"),
]


def run(script, log_name):
    path = os.path.join(HERE, script)
    log = os.path.join(LOGS, log_name)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        p.wait()
    return p.returncode, time.time() - t0


def main():
    want = [a.lower() for a in sys.argv[1:]] or [s[0] for s in STAGES]
    results = []
    for tag, script, desc in STAGES:
        if tag not in want:
            continue
        print("\n" + "=" * 78)
        print("RUNNING %s  --  %s" % (script, desc))
        print("=" * 78, flush=True)
        rc, secs = run(script, script.replace(".py", ".log"))
        results.append((script, rc, secs))
        if rc != 0:
            print("!! %s exited with %d -- see logs/%s"
                  % (script, rc, script.replace(".py", ".log")))

    print("\n" + "=" * 78)
    print("ALL STAGES")
    print("=" * 78)
    for script, rc, secs in results:
        print("  %-24s %s  %7.1f s" % (script, "ok " if rc == 0 else "FAIL", secs))

    if all(rc == 0 for _, rc, _ in results):
        print("\nrendering screenshots ...")
        subprocess.call([sys.executable, os.path.join(HERE, "make_screenshots.py")])


if __name__ == "__main__":
    main()
