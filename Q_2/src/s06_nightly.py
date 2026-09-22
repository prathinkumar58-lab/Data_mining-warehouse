"""
s06_nightly.py  --  the whole thing, running as the nightly job, against the
clock and against the head of product's second constraint.

This is the piece that turns parts (a)-(e) into a system:

    sketch the new notices  ->  insert their bands  ->  probe the index
    ->  sketch filter at tau - delta  ->  exact Jaccard, re-derived from the
    stored body  ->  merge decision at tau  ->  attach to an opportunity,
    minting a new id only when there is nothing to attach to

Two things are demonstrated that the earlier stages could not be:

  * THE BUDGET. Wall-clock for a full cold rebuild and for one night's intake,
    stage by stage, against 20 minutes.

  * BOOKMARKS. "The card id a bidder bookmarks today must still point at the
    same opportunity next month." Opportunity ids are minted once, derived from
    the founding notice, and never reissued. When tonight's intake shows that
    two existing opportunities were always the same thing, the younger id does
    not disappear -- it becomes an alias of the older one, so the bookmark
    still resolves. Three consecutive nights are run and the ids are checked.

Writes results/f_nightly.json.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

VARIANT, SCHEME = "canon", "w5"
K, BANDS, ROWS_PER_BAND = 384, 76, 5
TAU, TAU_SKETCH = 0.54, 0.46
BUCKET_CAP = 100
BUDGET_SECONDS = 20 * 60
NIGHTS = 3
INTAKE = 571


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *a):
        self.t[self._name] = self.t.get(self._name, 0.0) + time.time() - self._t0
        return False


class Clusters:
    """Opportunity identity, with a stability guarantee and a linkage policy.

    Identity
    --------
    An id is OPP-<founding notice id>. It is minted once and never reissued or
    renumbered. Absorbing one opportunity into another does not delete the
    absorbed id: it records an alias, so a bookmark taken against it still
    resolves.

    Linkage
    -------
    policy='single'  -- transitive closure: a notice joins the opportunity of
                        ANY notice it merged with, and two opportunities that
                        share a merged pair become one. This is the obvious
                        thing to do and it is wrong: similarity is not
                        transitive, so A~B and B~C drags A and C into one card
                        even when A and C are nothing alike. Measured, not
                        assumed -- see results/f_nightly.json.

    policy='leader'  -- a notice joins an opportunity only if it is above tau
                        against that opportunity's CANONICAL notice, and two
                        opportunities merge only if their canonicals are above
                        tau. Chains cannot form, because every member is within
                        tau of one fixed notice. The canonical never changes,
                        which is also what the bookmark guarantee wants. It
                        still permits a star of radius tau around the canonical,
                        and that is where its residual false merges come from.

    policy='complete' -- ADOPTED. A notice joins only if it clears tau against
                        EVERY existing member, and two opportunities merge only
                        if every cross pair clears tau. The card then has a
                        guaranteed diameter: any two notices a bidder sees on
                        one card are themselves above tau. Affordable because
                        cards are small (largest 9 members), so the extra work
                        is a handful of exact Jaccards per candidate.
    """

    def __init__(self, policy="leader", tau=TAU, sim=None):
        self.policy = policy
        self.tau = tau
        self.sim = sim                 # sim(a, b) -> exact Jaccard
        self.of = {}                   # notice_id -> opportunity_id
        self.members = {}              # opportunity_id -> set of notice_id
        self.canonical = {}            # opportunity_id -> notice_id
        self.born = {}                 # opportunity_id -> night minted
        self.alias = {}                # absorbed id -> surviving id
        self.alias_events = []

    def resolve(self, opp_id):
        seen = set()
        while opp_id in self.alias and opp_id not in seen:
            seen.add(opp_id)
            opp_id = self.alias[opp_id]
        return opp_id

    def mint(self, notice_id, night):
        opp = "OPP-" + notice_id
        self.of[notice_id] = opp
        self.members[opp] = {notice_id}
        self.canonical[opp] = notice_id
        self.born[opp] = night
        return opp

    def _absorb(self, survivor, other, night):
        for m in self.members.pop(other, set()):
            self.of[m] = survivor
            self.members[survivor].add(m)
        self.alias[other] = survivor
        self.canonical.pop(other, None)
        self.alias_events.append({"night": night, "absorbed": other, "into": survivor})

    def attach(self, notice_id, partners, night):
        opps = sorted({self.resolve(self.of[p]) for p in partners if p in self.of})
        if not opps:
            return self.mint(notice_id, night)

        if self.policy == "single":
            survivor = min(opps, key=lambda o: (self.born.get(o, 0), o))
            for o in opps:
                if o != survivor:
                    self._absorb(survivor, o, night)
            self.of[notice_id] = survivor
            self.members[survivor].add(notice_id)
            return survivor

        if self.policy == "complete":
            # every member of the card must be within tau of the new notice
            scored = []
            for o in opps:
                ms = self.members.get(o, ())
                worst = min((self.sim(notice_id, m) for m in ms), default=0.0)
                if worst >= self.tau:
                    scored.append((worst, o))
            if not scored:
                return self.mint(notice_id, night)
            scored.sort(key=lambda so: (-so[0], so[1]))
            survivor = scored[0][1]
            for s, o in scored[1:]:
                cross = min(self.sim(x, y) for x in self.members[survivor]
                            for y in self.members[o])
                if cross >= self.tau:
                    self._absorb(survivor, o, night)
            self.of[notice_id] = survivor
            self.members[survivor].add(notice_id)
            return survivor

        # leader: the notice must clear tau against the opportunity's canonical
        scored = [(self.sim(notice_id, self.canonical[o]), o) for o in opps
                  if o in self.canonical]
        scored = [(s, o) for s, o in scored if s >= self.tau]
        if not scored:
            return self.mint(notice_id, night)
        scored.sort(key=lambda so: (-so[0], so[1]))
        survivor = scored[0][1]
        # two opportunities merge only if their canonicals themselves agree
        for s, o in scored[1:]:
            if self.sim(self.canonical[survivor], self.canonical[o]) >= self.tau:
                self._absorb(survivor, o, night)
        self.of[notice_id] = survivor
        self.members[survivor].add(notice_id)
        return survivor

    def seed_from_pairs(self, ids, merged_pairs, night=0):
        """Build the initial clustering over an existing corpus, using the same
        rule the nightly job will use, so the two are consistent."""
        adj = {}
        for a, b, _, _ in merged_pairs:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        for nid in sorted(ids):
            partners = [p for p in adj.get(nid, []) if p in self.of]
            self.attach(nid, partners, night)


def sig_to_bytes(row):
    return row.astype(np.uint32).tobytes()


def bytes_to_sig(b):
    return np.frombuffer(b, dtype=np.uint32)


def main():
    t_start = time.time()
    out = {}
    T = Timer()

    df = C.load_notices()
    lab = C.load_labels()
    boiler = C.learn_boilerplate(df)

    # tonight's intake is the most recently published material, held out
    df = df.sort_values(["published_at", "notice_id"]).reset_index(drop=True)
    hold = NIGHTS * INTAKE
    base_df = df.iloc[:len(df) - hold].reset_index(drop=True)
    nights = [df.iloc[len(df) - hold + i * INTAKE: len(df) - hold + (i + 1) * INTAKE]
              .reset_index(drop=True) for i in range(NIGHTS)]
    C.log("base corpus %d notices; %d nights of %d" % (len(base_df), NIGHTS, INTAKE))

    con = C.pg_connect()
    con.autocommit = False
    cur = con.cursor()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"),
              encoding="utf-8") as fh:
        cur.execute(fh.read())
    cur.execute("CREATE TABLE IF NOT EXISTS suppressed_bucket ("
                "band_no SMALLINT, bucket_key BIGINT, members INT, "
                "suppressed_at TIMESTAMPTZ DEFAULT now(), "
                "PRIMARY KEY (band_no, bucket_key))")
    con.commit()

    # ------------------------------------------------------------------
    # COLD BUILD: the full corpus as it stands, sketched and indexed
    # ------------------------------------------------------------------
    with T("cold: shingle"):
        base_store = C.build_shingles(base_df, VARIANT, SCHEME, boiler, use_cache=False)
    with T("cold: minhash"):
        base_sig = C.minhash_signatures(base_store, K, use_cache=False)
    with T("cold: band"):
        base_keys = C.band_keys(base_sig, BANDS, ROWS_PER_BAND)

    with T("cold: load notice+sketch"):
        buf = io.StringIO()
        for r in base_df.itertuples(index=False):
            body = (r.body.replace("\\", "\\\\").replace("\t", " ")
                    .replace("\n", "\\n").replace("\r", " "))
            title = (str(r.title).replace("\\", "\\\\").replace("\t", " ")
                     .replace("\n", " ").replace("\r", " "))
            buf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                      % (r.notice_id, r.portal_id, r.published_at, title, body,
                         r.estimated_value, r.closing_date))
        buf.seek(0)
        cur.copy_from(buf, "notice", columns=("notice_id", "portal_id", "published_at",
                                              "title", "body", "estimated_value",
                                              "closing_date"))
        buf = io.StringIO()
        for i, nid in enumerate(base_store.ids):
            buf.write("%s\t%d\t%d\t\\\\x%s\n" % (nid, K, int(base_store.sizes[i]),
                                                 sig_to_bytes(base_sig[i]).hex()))
        buf.seek(0)
        cur.copy_from(buf, "notice_sketch",
                      columns=("notice_id", "k_rows", "shingles", "sig"))

    with T("cold: load buckets"):
        buf = io.StringIO()
        for j in range(BANDS):
            col = base_keys[:, j]
            for i, nid in enumerate(base_store.ids):
                buf.write("%d\t%d\t%s\n" % (j, col[i], nid))
        buf.seek(0)
        cur.copy_from(buf, "lsh_bucket", columns=("band_no", "bucket_key", "notice_id"))
        con.commit()

    with T("cold: apply bucket cap"):
        cur.execute("""
            WITH big AS (
              SELECT band_no, bucket_key, count(*) AS n
              FROM lsh_bucket GROUP BY band_no, bucket_key HAVING count(*) > %s)
            INSERT INTO suppressed_bucket (band_no, bucket_key, members)
            SELECT band_no, bucket_key, n FROM big
            ON CONFLICT (band_no, bucket_key) DO UPDATE SET members = EXCLUDED.members
        """, (BUCKET_CAP,))
        cur.execute("""DELETE FROM lsh_bucket b USING suppressed_bucket s
                       WHERE b.band_no = s.band_no AND b.bucket_key = s.bucket_key""")
        n_suppressed = cur.rowcount
        con.commit()
    con.autocommit = True
    cur.execute("VACUUM ANALYZE lsh_bucket")
    cur.execute("VACUUM ANALYZE notice_sketch")
    con.autocommit = False

    cur.execute("SELECT count(*) FROM suppressed_bucket")
    n_sup_buckets = cur.fetchone()[0]
    C.log("cold build done; suppressed %d oversized buckets (%d memberships)"
          % (n_sup_buckets, n_suppressed))

    # cluster the base corpus so the nights have something to attach to
    with T("cold: cluster base corpus"):
        sig_by_id = {nid: base_sig[i] for i, nid in enumerate(base_store.ids)}
        store_by_id = {nid: base_store[i] for i, nid in enumerate(base_store.ids)}

        def sim(a, b):
            return C.jaccard_exact(store_by_id[a], store_by_id[b])

        clusters = Clusters(policy="complete", sim=sim)   # adopted
        shadow = Clusters(policy="single", sim=sim)       # measured, not shipped
        leader = Clusters(policy="leader", sim=sim)       # measured, not shipped
        cur.execute("""
            SELECT a.notice_id, b.notice_id
            FROM lsh_bucket a JOIN lsh_bucket b
              ON a.band_no = b.band_no AND a.bucket_key = b.bucket_key
             AND a.notice_id < b.notice_id
            GROUP BY a.notice_id, b.notice_id
        """)
        base_pairs = cur.fetchall()
        merged = []
        for a, b in base_pairs:
            js = float((sig_by_id[a] == sig_by_id[b]).mean())
            if js < TAU_SKETCH:
                continue
            je = C.jaccard_exact(store_by_id[a], store_by_id[b])
            if je >= TAU:
                merged.append((a, b, js, je))
        clusters.seed_from_pairs(base_store.ids, merged, night=0)
        shadow.seed_from_pairs(base_store.ids, merged, night=0)
        leader.seed_from_pairs(base_store.ids, merged, night=0)

    # persist the cold-build clustering and the index metadata, so the database
    # is a complete, self-describing artefact rather than a half-filled one
    with T("cold: persist clusters"):
        cur.execute("DELETE FROM index_meta")
        cur.execute("INSERT INTO index_meta (k_rows, bands, rows_per_band, variant, "
                    "scheme, tau, tau_sketch, bucket_cap) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (K, BANDS, ROWS_PER_BAND, VARIANT, SCHEME, TAU, TAU_SKETCH, BUCKET_CAP))
        buf = io.StringIO()
        for o, ms in clusters.members.items():
            buf.write("%s\t%s\t%d\n" % (o, clusters.canonical.get(o, sorted(ms)[0]), len(ms)))
        buf.seek(0)
        cur.copy_from(buf, "opportunity",
                      columns=("opportunity_id", "canonical_notice_id", "member_count"))
        buf = io.StringIO()
        for nid, o in clusters.of.items():
            buf.write("%s\t%s\n" % (nid, clusters.resolve(o)))
        buf.seek(0)
        cur.copy_from(buf, "notice_opportunity",
                      columns=("notice_id", "opportunity_id"))
        con.commit()

    out["cold_build"] = {
        "notices": len(base_df),
        "candidate_pairs_examined": len(base_pairs),
        "merged_pairs": len(merged),
        "opportunities": len(clusters.members),
        "suppressed_buckets": n_sup_buckets,
        "suppressed_memberships": n_suppressed,
        "seconds_by_stage": {k: round(v, 2) for k, v in T.t.items() if k.startswith("cold")},
        "total_seconds": round(sum(v for k, v in T.t.items() if k.startswith("cold")), 1),
    }
    C.log("cold build: %s candidate pairs -> %s merges -> %s opportunities in %.1f s"
          % (f"{len(base_pairs):,}", f"{len(merged):,}", f"{len(clusters.members):,}",
             out["cold_build"]["total_seconds"]))

    # snapshot ids for the bookmark test
    bookmark_sample = sorted(clusters.members.keys())[:5000]
    bookmarks = {opp: sorted(clusters.members[opp])[0] for opp in bookmark_sample}

    # ------------------------------------------------------------------
    # THE NIGHTS
    # ------------------------------------------------------------------
    night_rows = []
    all_merged = list(merged)
    for n_i, nd in enumerate(nights, start=1):
        NT = Timer()
        with NT("1 sketch"):
            st = C.build_shingles(nd, VARIANT, SCHEME, boiler, use_cache=False)
            sg = C.minhash_signatures(st, K, use_cache=False)
            kk = C.band_keys(sg, BANDS, ROWS_PER_BAND)

        with NT("2 insert notices and sketches"):
            buf = io.StringIO()
            for r in nd.itertuples(index=False):
                body = (r.body.replace("\\", "\\\\").replace("\t", " ")
                        .replace("\n", "\\n").replace("\r", " "))
                title = (str(r.title).replace("\\", "\\\\").replace("\t", " ")
                         .replace("\n", " ").replace("\r", " "))
                buf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                          % (r.notice_id, r.portal_id, r.published_at, title, body,
                             r.estimated_value, r.closing_date))
            buf.seek(0)
            cur.copy_from(buf, "notice", columns=("notice_id", "portal_id", "published_at",
                                                  "title", "body", "estimated_value",
                                                  "closing_date"))
            buf = io.StringIO()
            for i, nid in enumerate(st.ids):
                buf.write("%s\t%d\t%d\t\\\\x%s\n" % (nid, K, int(st.sizes[i]),
                                                     sig_to_bytes(sg[i]).hex()))
            buf.seek(0)
            cur.copy_from(buf, "notice_sketch",
                          columns=("notice_id", "k_rows", "shingles", "sig"))
            con.commit()

        with NT("3 probe the index"):
            cur.execute("DROP TABLE IF EXISTS probe")
            cur.execute("CREATE TEMP TABLE probe (band_no SMALLINT, bucket_key BIGINT, "
                        "notice_id TEXT)")
            buf = io.StringIO()
            for i, nid in enumerate(st.ids):
                for j in range(BANDS):
                    buf.write("%d\t%d\t%s\n" % (j, kk[i, j], nid))
            buf.seek(0)
            cur.copy_from(buf, "probe", columns=("band_no", "bucket_key", "notice_id"))
            cur.execute("ANALYZE probe")
            cur.execute("""
                SELECT DISTINCT p.notice_id, b.notice_id
                FROM probe p
                JOIN lsh_bucket b ON b.band_no = p.band_no AND b.bucket_key = p.bucket_key
                WHERE b.notice_id <> p.notice_id
            """)
            cand = cur.fetchall()

        with NT("4 sketch filter"):
            for i, nid in enumerate(st.ids):
                sig_by_id[nid] = sg[i]
                store_by_id[nid] = st[i]
            survivors = []
            missing = 0
            for a, b in cand:
                sa, sb = sig_by_id.get(a), sig_by_id.get(b)
                if sa is None or sb is None:
                    missing += 1
                    continue
                js = float((sa == sb).mean())
                if js >= TAU_SKETCH:
                    survivors.append((a, b, js))

        with NT("5 exact verification"):
            decisions = []
            for a, b, js in survivors:
                je = C.jaccard_exact(store_by_id[a], store_by_id[b])
                decisions.append((a, b, js, je, "merge" if je >= TAU else "reject"))

        with NT("6 assign opportunities"):
            by_new = {}
            for a, b, js, je, d in decisions:
                if d != "merge":
                    continue
                for x, y in ((a, b), (b, a)):
                    if x in set(st.ids):
                        by_new.setdefault(x, []).append((y, je))
            new_ids = list(st.ids)
            for nid in new_ids:
                partners = [p for p, _ in by_new.get(nid, [])]
                clusters.attach(nid, partners, n_i)
                shadow.attach(nid, partners, n_i)
                leader.attach(nid, partners, n_i)

        with NT("7 persist decisions and clusters"):
            buf = io.StringIO()
            for a, b, js, je, d in decisions:
                lo, hi = (a, b) if a < b else (b, a)
                buf.write("%s\t%s\t%.6f\t%.6f\t%s\n" % (lo, hi, js, je, d))
            buf.seek(0)
            cur.copy_from(buf, "merge_decision",
                          columns=("notice_id_a", "notice_id_b", "j_sketch", "j_exact",
                                   "decision"))
            # opportunities: insert any new ids, then the memberships for tonight
            cur.execute("SELECT opportunity_id FROM opportunity")
            known = set(r[0] for r in cur.fetchall())
            fresh = [o for o in clusters.members if o not in known]
            if fresh:
                buf = io.StringIO()
                for o in fresh:
                    buf.write("%s\t%s\t%d\n" % (o, sorted(clusters.members[o])[0],
                                                len(clusters.members[o])))
                buf.seek(0)
                cur.copy_from(buf, "opportunity",
                              columns=("opportunity_id", "canonical_notice_id",
                                       "member_count"))
            buf = io.StringIO()
            for nid in new_ids:
                buf.write("%s\t%s\n" % (nid, clusters.resolve(clusters.of[nid])))
            buf.seek(0)
            cur.copy_from(buf, "notice_opportunity",
                          columns=("notice_id", "opportunity_id"))
            if clusters.alias_events:
                cur.execute("SELECT alias_id FROM opportunity_alias")
                have = set(r[0] for r in cur.fetchall())
                rows = [e for e in clusters.alias_events if e["absorbed"] not in have]
                if rows:
                    buf = io.StringIO()
                    for e in rows:
                        buf.write("%s\t%s\n" % (e["absorbed"], e["into"]))
                    buf.seek(0)
                    cur.copy_from(buf, "opportunity_alias",
                                  columns=("alias_id", "opportunity_id"))
            con.commit()

        with NT("8 index the new notices"):
            buf = io.StringIO()
            for i, nid in enumerate(st.ids):
                for j in range(BANDS):
                    buf.write("%d\t%d\t%s\n" % (j, kk[i, j], nid))
            buf.seek(0)
            cur.copy_from(buf, "lsh_bucket", columns=("band_no", "bucket_key", "notice_id"))
            # maintain the cap incrementally: only buckets touched tonight can grow
            cur.execute("""
                WITH touched AS (SELECT DISTINCT band_no, bucket_key FROM probe),
                     big AS (
                       SELECT b.band_no, b.bucket_key, count(*) AS n
                       FROM lsh_bucket b JOIN touched t
                         ON t.band_no = b.band_no AND t.bucket_key = b.bucket_key
                       GROUP BY b.band_no, b.bucket_key HAVING count(*) > %s)
                INSERT INTO suppressed_bucket (band_no, bucket_key, members)
                SELECT band_no, bucket_key, n FROM big
                ON CONFLICT (band_no, bucket_key) DO UPDATE SET members = EXCLUDED.members
            """, (BUCKET_CAP,))
            cur.execute("""DELETE FROM lsh_bucket b USING suppressed_bucket s
                           WHERE b.band_no = s.band_no AND b.bucket_key = s.bucket_key""")
            con.commit()

        all_merged.extend((a, b, js, je) for a, b, js, je, d in decisions if d == "merge")
        tot = sum(NT.t.values())
        night_rows.append({
            "night": n_i,
            "new_notices": len(nd),
            "candidate_pairs": len(cand),
            "survived_sketch_filter": len(survivors),
            "exact_verifications": len(decisions),
            "merges": sum(1 for d in decisions if d[4] == "merge"),
            "new_opportunities": sum(1 for nid in new_ids
                                     if clusters.resolve(clusters.of[nid]) == "OPP-" + nid),
            "aliases_created_total": len(clusters.alias_events),
            "seconds_by_stage": {k: round(v, 2) for k, v in NT.t.items()},
            "total_seconds": round(tot, 2),
            "budget_used_pct": round(100 * tot / BUDGET_SECONDS, 2),
        })
        C.log("night %d: %d new, %s candidates, %d verified, %d merges, %.1f s "
              "(%.1f%% of the 20-minute budget)"
              % (n_i, len(nd), f"{len(cand):,}", len(decisions),
                 night_rows[-1]["merges"], tot, night_rows[-1]["budget_used_pct"]))

    out["nights"] = night_rows

    # ------------------------------------------------------------------
    # BOOKMARKS: do the ids still point at the same opportunity?
    # ------------------------------------------------------------------
    broken, redirected, intact = 0, 0, 0
    for opp, member in bookmarks.items():
        now = clusters.resolve(opp)
        if member not in clusters.of:
            broken += 1
        elif now == opp:
            intact += 1
        elif clusters.of[member] == now:
            redirected += 1
        else:
            broken += 1
    cur.execute("SELECT count(*) FROM opportunity_alias")
    n_alias = cur.fetchone()[0]
    out["bookmark_stability"] = {
        "bookmarks_taken_after_cold_build": len(bookmarks),
        "ids_still_valid_unchanged": intact,
        "ids_resolved_through_an_alias": redirected,
        "ids_that_stopped_resolving": broken,
        "aliases_recorded_in_db": n_alias,
        "guarantee": ("An opportunity id is OPP-<founding notice id>. It is minted once "
                      "and never reissued. Absorption writes an alias row rather than "
                      "deleting the id, so every bookmark ever issued still resolves -- "
                      "%d of %d unchanged, %d through an alias, %d broken."
                      % (intact, len(bookmarks), redirected, broken)),
    }
    C.log("bookmarks: %d unchanged, %d via alias, %d broken" % (intact, redirected, broken))

    # ------------------------------------------------------------------
    # END-TO-END QUALITY on the labelled pairs, including transitivity
    # ------------------------------------------------------------------
    def score_clustering(cl):
        same_together = same_total = diff_together = diff_total = 0
        fm = []
        for r in lab.itertuples(index=False):
            a, b = r.notice_id_a, r.notice_id_b
            if a not in cl.of or b not in cl.of:
                continue
            together = cl.resolve(cl.of[a]) == cl.resolve(cl.of[b])
            if r.label == "same":
                same_total += 1
                same_together += together
            else:
                diff_total += 1
                diff_together += together
                if together:
                    fm.append([a, b])
        sizes = np.array([len(v) for v in cl.members.values()])
        return {
            "same_pairs_in_one_opportunity": int(same_together),
            "same_pairs_total": int(same_total),
            "recall": round(same_together / max(same_total, 1), 4),
            "different_pairs_in_one_opportunity": int(diff_together),
            "different_pairs_total": int(diff_total),
            "false_merge_rate": round(diff_together / max(diff_total, 1), 5),
            "false_merge_examples": fm[:10],
            "opportunities": int(len(cl.members)),
            "reduction_pct": round(100 * (1 - len(cl.members) / len(cl.of)), 2),
            "largest_cluster": int(sizes.max()),
            "size_histogram": {str(k): int(v) for k, v in
                               zip(*np.unique(sizes, return_counts=True))},
        }

    adopted = score_clustering(clusters)
    single = score_clustering(shadow)
    leader_s = score_clustering(leader)
    out["end_to_end_on_labels"] = adopted
    out["end_to_end_on_labels"]["note"] = (
        "Measured on the FINAL cluster assignment, so transitivity is included: a pair "
        "counts as merged if the system put the two notices in the same opportunity, "
        "directly or through a chain. This is where a threshold that is perfectly safe "
        "PAIRWISE can still produce a false merge, which is why it is measured and not "
        "assumed.")
    out["linkage_comparison"] = {
        "single_linkage_transitive_closure": single,
        "leader_linkage_canonical_check": leader_s,
        "complete_linkage_all_members": adopted,
        "finding": (
            "Pairwise, tau = %.2f puts ZERO of the 621 adjudicated 'different' pairs "
            "above the line. Closing the merge relation transitively puts %d of them "
            "into one card anyway (%.2f%%) and grows a runaway cluster of %d notices: "
            "similarity is not transitive and single linkage does not care. Anchoring "
            "every member to the card's canonical notice cuts that to %d (%.3f%%) but "
            "still allows a star of radius tau, which is where the last one comes from. "
            "Requiring the new notice to clear tau against EVERY existing member takes "
            "false merges to %d and caps the largest card at %d. The price is recall: "
            "%d of %d adjudicated duplicates end up on one card, against %d under single "
            "linkage. Under a 50:1 ratio that is the trade the ratio exists to make."
            % (TAU, single["different_pairs_in_one_opportunity"],
               100 * single["false_merge_rate"], single["largest_cluster"],
               leader_s["different_pairs_in_one_opportunity"],
               100 * leader_s["false_merge_rate"],
               adopted["different_pairs_in_one_opportunity"], adopted["largest_cluster"],
               adopted["same_pairs_in_one_opportunity"], adopted["same_pairs_total"],
               single["same_pairs_in_one_opportunity"])),
    }
    out["clusters"] = {
        "notices": int(len(clusters.of)),
        "opportunities": adopted["opportunities"],
        "cards_removed": int(len(clusters.of) - adopted["opportunities"]),
        "reduction_pct": adopted["reduction_pct"],
        "size_histogram": adopted["size_histogram"],
        "largest_cluster": adopted["largest_cluster"],
    }

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stages = list(night_rows[0]["seconds_by_stage"].keys())
    bottom = np.zeros(len(night_rows))
    cmap = plt.get_cmap("tab10")
    for si, sname in enumerate(stages):
        v = np.array([r["seconds_by_stage"].get(sname, 0) for r in night_rows])
        axes[0].bar([r["night"] for r in night_rows], v, bottom=bottom,
                    label=sname, color=cmap(si % 10))
        bottom += v
    axes[0].axhline(BUDGET_SECONDS, color="k", ls="--", label="20-minute budget")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("night")
    axes[0].set_ylabel("seconds (log)")
    axes[0].set_xticks([r["night"] for r in night_rows])
    axes[0].set_title("Nightly wall-clock by stage\n(total %.1f s against a %d s budget)"
                      % (night_rows[-1]["total_seconds"], BUDGET_SECONDS))
    axes[0].legend(fontsize=7, ncol=2)
    ks = sorted(int(k) for k in out["clusters"]["size_histogram"])
    vs = [out["clusters"]["size_histogram"][str(k)] for k in ks]
    axes[1].bar([str(k) for k in ks], vs, color="#4c72b0")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("notices per opportunity")
    axes[1].set_ylabel("opportunities")
    axes[1].set_title("%s notices collapsed to %s cards (-%.1f%%)"
                      % (f"{out['clusters']['notices']:,}",
                         f"{out['clusters']['opportunities']:,}",
                         out["clusters"]["reduction_pct"]))
    fig.suptitle("The system running as a nightly job")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "f_nightly.png"), dpi=140)
    C.log("wrote figures/f_nightly.png")

    con.close()
    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    C.save_json("f_nightly.json", out)

    # ------------------------------------------------------------- report
    print()
    print("=" * 96)
    print("NIGHTLY PIPELINE -- the whole system against the clock")
    print("=" * 96)
    cb = out["cold_build"]
    print("cold build on %s notices: %.1f s total" % (f"{cb['notices']:,}", cb["total_seconds"]))
    for k, v in cb["seconds_by_stage"].items():
        print("    %-28s %8.2f s" % (k, v))
    print("    suppressed %d oversized buckets (%s memberships) at cap %d"
          % (cb["suppressed_buckets"], f"{cb['suppressed_memberships']:,}", BUCKET_CAP))
    print()
    print("nights:")
    hdr = ("%6s %8s %14s %12s %10s %10s %12s" %
           ("night", "new", "candidates", "verified", "merges", "seconds", "% of budget"))
    print(hdr)
    print("-" * len(hdr))
    for r in night_rows:
        print("%6d %8d %14s %12d %10d %10.1f %11.2f%%"
              % (r["night"], r["new_notices"], f"{r['candidate_pairs']:,}",
                 r["exact_verifications"], r["merges"], r["total_seconds"],
                 r["budget_used_pct"]))
    print()
    print("  stage breakdown, last night:")
    for k, v in night_rows[-1]["seconds_by_stage"].items():
        print("    %-32s %8.2f s" % (k, v))
    print()
    bs = out["bookmark_stability"]
    print("bookmark stability over %d nights: %d unchanged, %d resolved through an alias, "
          "%d broken (of %d ids issued)"
          % (NIGHTS, bs["ids_still_valid_unchanged"], bs["ids_resolved_through_an_alias"],
             bs["ids_that_stopped_resolving"], bs["bookmarks_taken_after_cold_build"]))
    print()
    print("end-to-end on the adjudicated pairs (final clusters, transitivity included):")
    hdr = ("%-34s %16s %18s %16s %10s" %
           ("linkage policy", "duplicates kept", "FALSE MERGES", "opportunities",
            "biggest"))
    print(hdr)
    print("-" * len(hdr))
    for nm, e in (("single linkage (transitive closure)", single),
                  ("leader linkage (canonical check)", leader_s),
                  ("complete linkage (all members)  <=", adopted)):
        print("%-34s %9d / %-4d %11d / %-4d %16d %10d"
              % (nm, e["same_pairs_in_one_opportunity"], e["same_pairs_total"],
                 e["different_pairs_in_one_opportunity"], e["different_pairs_total"],
                 e["opportunities"], e["largest_cluster"]))
    print()
    print("  " + out["linkage_comparison"]["finding"])
    c = out["clusters"]
    print()
    print("cards: %s notices -> %s opportunities (%.1f%% fewer cards), largest cluster %d"
          % (f"{c['notices']:,}", f"{c['opportunities']:,}", c["reduction_pct"],
             c["largest_cluster"]))
    print("elapsed %.1fs" % out["elapsed_seconds"])


if __name__ == "__main__":
    main()
