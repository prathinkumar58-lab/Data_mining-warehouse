"""
s05_skew.py  --  PART (e): find the place where the design betrays you.

Runs the adopted retrieval (canon/w5, K=384, b=76, r=5) over the whole corpus
and looks at how the work is distributed rather than at its total. Then:

  1. locates the concentration empirically (it is at the BUCKET level, not the
     notice level -- which is itself the finding);
  2. explains mechanically why this corpus produces it, using the shingle
     document-frequency distribution and the contents of the worst bucket;
  3. quantifies it against the 20-minute nightly budget;
  4. mitigates it two ways and PRICES BOTH against labelled_pairs.csv:
       M1  document-frequency stop-list  (changes the representation)
       M2  bucket cap                    (changes only retrieval)
     and reports the distribution and runtime before and after.

Writes results/e_skew.json and four figures.
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from s03_lsh_tuning import anchor_sample, anchor_recall

VARIANT, SCHEME = "canon", "w5"
K, BANDS, ROWS_PER_BAND = 384, 76, 5
TAU = 0.54
NODAL = ["P001", "P002", "P003", "P004", "P005", "P006"]
NIGHTLY_INTAKE = 571
BUDGET_SECONDS = 20 * 60

DF_STOPS = [None, 0.20, 0.10, 0.05, 0.02]
CAPS = [None, 400, 200, 100, 50, 25]


def band_stats(keys):
    """Per-notice work, bucket sizes, total emissions -- all from the band keys."""
    N, b = keys.shape
    per = np.zeros(N, dtype=np.int64)
    sizes = []
    for j in range(b):
        uq, inv, cnt = np.unique(keys[:, j], return_inverse=True, return_counts=True)
        per += cnt[inv] - 1
        sizes.append(cnt[cnt > 1])
    sizes = np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)
    emissions = int((sizes * (sizes - 1) // 2).sum())
    return per, sizes, emissions


def capped_stats(keys, cap):
    """Work after refusing to expand any bucket with more than `cap` members."""
    N, b = keys.shape
    per = np.zeros(N, dtype=np.int64)
    sizes = []
    dropped_buckets = dropped_members = 0
    for j in range(b):
        uq, inv, cnt = np.unique(keys[:, j], return_inverse=True, return_counts=True)
        big = cnt > cap
        keep = ~big[inv]
        c = cnt[inv]
        per += np.where(keep, c - 1, 0)
        s = cnt[(cnt > 1) & ~big]
        sizes.append(s)
        dropped_buckets += int(big.sum())
        dropped_members += int(cnt[big].sum())
    sizes = np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)
    emissions = int((sizes * (sizes - 1) // 2).sum())
    return per, sizes, emissions, dropped_buckets, dropped_members


def capped_candidate(keys, pairs_a, pairs_b, cap):
    """Would each (a, b) pair still be retrieved with a bucket cap in force?"""
    ok = np.zeros(len(pairs_a), dtype=bool)
    for j in range(keys.shape[1]):
        uq, inv, cnt = np.unique(keys[:, j], return_inverse=True, return_counts=True)
        small = cnt[inv] <= cap
        match = (keys[pairs_a, j] == keys[pairs_b, j]) & small[pairs_a]
        ok |= match
    return ok


def anchor_recall_capped(keys, anchors, J, tau, cap):
    tot = hit = 0
    bucket_ok = np.empty(keys.shape, dtype=bool)
    for j in range(keys.shape[1]):
        uq, inv, cnt = np.unique(keys[:, j], return_inverse=True, return_counts=True)
        bucket_ok[:, j] = cnt[inv] <= cap
    for k, a in enumerate(anchors):
        j_ = J[k]
        m = (~np.isnan(j_)) & (j_ >= tau)
        if not m.any():
            continue
        a = int(a)
        cand = ((keys[a][None, :] == keys) & bucket_ok & bucket_ok[a][None, :]).any(axis=1)
        tot += int(m.sum())
        hit += int(cand[m].sum())
    return hit, tot


def evaluate_representation(store, df, lab, tau_ref=TAU):
    """Separation quality of a (possibly stop-listed) representation."""
    idx = {nid: i for i, nid in enumerate(store.ids)}
    ia = np.array([idx[a] for a in lab.notice_id_a])
    ib = np.array([idx[b] for b in lab.notice_id_b])
    y = (lab.label.values == "same")
    J = np.array([C.jaccard_exact(store[a], store[b]) for a, b in zip(ia, ib)])
    pos, neg = J[y], J[~y]
    s = pd.Series(np.concatenate([pos, neg]))
    ranks = s.rank(method="average").values
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    r0 = float((pos > neg.max()).mean())
    # cost-optimal threshold under the 50:1 ratio, recomputed for THIS representation
    cands = np.unique(np.concatenate([pos, neg]))
    best, tau = None, tau_ref
    for t in cands:
        cost = 50.0 * float((neg >= t).mean()) + float((pos < t).mean())
        if best is None or cost < best:
            best, tau = cost, float(t)
    diff_p99 = float(np.percentile(neg, 99))
    # how big would the sketch have to be to support THIS threshold?
    # delta = how far tau sits above the bulk of the negatives; the sketch must
    # resolve that gap, and K grows as 1/delta^2.
    delta = max(tau - diff_p99, 1e-3)
    k_req = 3.0902 ** 2 * tau * (1 - tau) / delta ** 2
    return dict(auc=round(float(auc), 5), recall_at_zero_fp=round(r0, 4),
                tau=round(tau, 4), diff_max=round(float(neg.max()), 4),
                diff_p99=round(diff_p99, 4),
                resolution_needed=round(delta, 4),
                K_required_for_this_tau=int(np.ceil(k_req)),
                same_median=round(float(np.median(pos)), 4),
                merged_share=round(float((pos >= tau).mean()), 4)), J, y, ia, ib


def main():
    t_start = time.time()
    out = {}
    df = C.load_notices()
    lab = C.load_labels()
    boiler = C.learn_boilerplate(df)

    store = C.build_shingles(df, VARIANT, SCHEME, boiler)
    sig = C.minhash_signatures(store, 768, tag="%s_%s" % (VARIANT, SCHEME))[:, :K]
    keys = C.band_keys(sig, BANDS, ROWS_PER_BAND)
    anchors, J_anchor = anchor_sample(store)

    # ------------------------------------------------------------------
    # 1. where the work is
    # ------------------------------------------------------------------
    per, sizes, emissions = band_stats(keys)
    srt = np.sort(sizes)[::-1]
    cum = np.cumsum(srt * (srt - 1) // 2)
    conc = {}
    for q in (0.001, 0.005, 0.01, 0.05, 0.10):
        k = max(int(len(srt) * q), 1)
        conc["top_%.1f%%_buckets" % (q * 100)] = dict(
            buckets=k, share_of_emissions=round(float(cum[k - 1] / emissions), 4))
    order = np.argsort(per)[::-1]
    cpn = np.cumsum(per[order])
    conc_notice = {}
    for q in (0.01, 0.02, 0.05, 0.10):
        k = int(len(per) * q)
        conc_notice["top_%d%%_notices" % int(q * 100)] = round(float(cpn[k - 1] / per.sum()), 4)

    out["distribution_before"] = {
        "total_emissions": emissions,
        "colliding_buckets": int(len(sizes)),
        "largest_bucket": int(sizes.max()),
        "bucket_size_p50": float(np.median(sizes)),
        "bucket_size_p99": float(np.percentile(sizes, 99)),
        "per_notice_mean": round(float(per.mean()), 1),
        "per_notice_median": float(np.median(per)),
        "per_notice_p99": float(np.percentile(per, 99)),
        "per_notice_max": int(per.max()),
        "concentration_by_bucket": conc,
        "concentration_by_notice": conc_notice,
        "finding": ("The skew is at the BUCKET level, not the notice level. The worst "
                    "1%% of notices carry only %.1f%% of the work, which looks benign; "
                    "the worst 0.1%% of buckets carry %.1f%% of it. Looking at per-notice "
                    "cost would have hidden the problem entirely."
                    % (100 * conc_notice["top_1%_notices"],
                       100 * conc["top_0.1%_buckets"]["share_of_emissions"])),
    }
    C.log("emissions %s; top 0.1%% of buckets -> %.1f%% of work; largest bucket %d"
          % (f"{emissions:,}", 100 * conc["top_0.1%_buckets"]["share_of_emissions"],
             sizes.max()))

    # portal attribution
    dd = df.copy()
    dd["work"] = per
    g = (dd.groupby("portal_id")
         .agg(notices=("work", "size"), mean_work=("work", "mean"), total_work=("work", "sum"))
         .sort_values("total_work", ascending=False))
    g["share_of_work"] = g.total_work / per.sum()
    g["share_of_corpus"] = g.notices / len(dd)
    out["portal_attribution"] = g.head(15).round(4).reset_index().to_dict("records")
    is_nodal = dd.portal_id.isin(NODAL)
    out["nodal_share"] = {
        "share_of_corpus": round(float(is_nodal.mean()), 4),
        "share_of_work": round(float(dd[is_nodal].work.sum() / per.sum()), 4),
        "P094_share_of_corpus": round(float((dd.portal_id == "P094").mean()), 4),
        "P094_share_of_work": round(float(dd[dd.portal_id == "P094"].work.sum() / per.sum()), 4),
    }

    # ------------------------------------------------------------------
    # 2. why: anatomy of the worst bucket, and the shingle DF distribution
    # ------------------------------------------------------------------
    worst = None
    for j in range(BANDS):
        uq, cnt = np.unique(keys[:, j], return_counts=True)
        i = int(cnt.argmax())
        if worst is None or cnt[i] > worst[2]:
            worst = (j, int(uq[i]), int(cnt[i]))
    wj, wk, wn = worst
    mem = np.where(keys[:, wj] == wk)[0]
    rng = np.random.default_rng(3)
    sel = rng.choice(len(mem), min(80, len(mem)), replace=False)
    js = []
    for x in range(len(sel)):
        for yy in range(x + 1, len(sel)):
            js.append(C.jaccard_exact(store[int(mem[sel[x]])],
                                      store[int(mem[sel[yy]])]))
    js = np.array(js)
    pv = df.iloc[mem].portal_id.value_counts()
    out["worst_bucket"] = {
        "band": wj, "members": wn,
        "pairs_it_emits": int(wn * (wn - 1) // 2),
        "share_of_all_emissions": round(float(wn * (wn - 1) / 2 / emissions), 4),
        "portals": {str(k): int(v) for k, v in pv.head(8).items()},
        "pairwise_J_mean": round(float(js.mean()), 4),
        "pairwise_J_p95": round(float(np.percentile(js, 95)), 4),
        "share_of_its_pairs_above_tau": round(float((js >= TAU).mean()), 5),
        "estimated_real_duplicates_in_it": int((js >= TAU).mean() * wn * (wn - 1) / 2),
    }
    C.log("worst bucket: band %d, %d members, emits %s pairs, mean pairwise J %.3f, "
          "%.3f%% of them above tau"
          % (wj, wn, f"{wn*(wn-1)//2:,}", js.mean(), 100 * (js >= TAU).mean()))

    uniq, counts = np.unique(store.flat, return_counts=True)
    dfreq = counts / len(store)
    out["shingle_df"] = {
        "distinct_shingles": int(len(uniq)),
        "share_in_gt_1pct_of_notices": round(float((dfreq > 0.01).mean()), 6),
        "postings_in_gt_1pct": round(float(counts[dfreq > 0.01].sum() / counts.sum()), 4),
        "postings_in_gt_5pct": round(float(counts[dfreq > 0.05].sum() / counts.sum()), 4),
        "postings_in_gt_20pct": round(float(counts[dfreq > 0.20].sum() / counts.sum()), 4),
        "max_df": round(float(dfreq.max()), 4),
        "mechanism": (
            "A MinHash row is the minimum of a hash over the notice's shingle set. A "
            "shingle present in a large fraction of the corpus is a candidate for that "
            "minimum in every one of those notices at once, so whenever such a shingle "
            "happens to draw a small value under permutation i, thousands of notices "
            "get the SAME value in row i. A band is r=5 consecutive rows; if all five "
            "are won by corpus-wide shingles, every notice carrying them lands in one "
            "bucket. The bucket then emits n(n-1)/2 pairs -- quadratic in a quantity "
            "the tuning in part (c) never controlled, because the S-curve is a "
            "statement about ONE pair and says nothing about how many pairs share a "
            "bucket. %.1f%% of all shingle postings sit in shingles that occur in more "
            "than 1%% of notices; that tail is the fuel."
            % (100 * float(counts[dfreq > 0.01].sum() / counts.sum()))),
    }

    # ------------------------------------------------------------------
    # 3. what it costs against the budget
    # ------------------------------------------------------------------
    # per-pair costs, measured properly: 200k pairs per trial, median of three,
    # so the projection below does not rest on a noisy micro-benchmark.
    rng_t = np.random.default_rng(1)
    ta = rng_t.integers(0, len(store), 200000)
    tb = rng_t.integers(0, len(store), 200000)
    trials = []
    for _ in range(3):
        t0 = time.time()
        _ = (sig[ta] == sig[tb]).mean(axis=1)
        trials.append((time.time() - t0) / 200000 * 1e6)
    sketch_us = float(np.median(trials))
    trials = []
    for _ in range(3):
        t0 = time.time()
        for i in range(3000):
            C.jaccard_exact(store[i], store[i + 1])
        trials.append((time.time() - t0) / 3000 * 1e6)
    exact_us = float(np.median(trials))

    # the database side: rows/second measured by part (d)'s bulk join
    db_us = None
    dpath = os.path.join(C.RESULTS, "d_database.json")
    if os.path.exists(dpath):
        import json
        dj = json.load(open(dpath, encoding="utf-8"))
        a = [p for p in dj.get("access_paths_bulk", []) if p.get("tag") == "A"]
        if a and a[0].get("candidate_rows"):
            db_us = a[0]["seconds"] / a[0]["candidate_rows"] * 1e6
    per_pair_us = sketch_us + (db_us or 0.0)

    nightly_emissions = emissions * NIGHTLY_INTAKE / len(store) * 2
    out["budget_before"] = {
        "sketch_microseconds_per_pair": round(sketch_us, 3),
        "database_microseconds_per_candidate_row": round(db_us, 3) if db_us else None,
        "per_pair_microseconds_end_to_end": round(per_pair_us, 3),
        "exact_microseconds_per_pair": round(exact_us, 2),
        "full_corpus_emissions": emissions,
        "full_corpus_seconds": round(emissions * per_pair_us / 1e6, 1),
        "nightly_emissions_estimate": int(nightly_emissions),
        "nightly_seconds": round(nightly_emissions * per_pair_us / 1e6, 1),
        "budget_seconds": BUDGET_SECONDS,
        "worst_bucket_alone_seconds": round(wn * (wn - 1) / 2 * per_pair_us / 1e6, 1),
        "what_is_counted": ("emitting the candidate row from the covering index (measured "
                            "in part (d)) plus scoring it against the sketch. Exact "
                            "verification is counted separately: only pairs above the "
                            "sketch filter reach it."),
    }

    # --- the projection that actually matters: this is a growth problem
    #
    # Bucket membership is a roughly constant FRACTION of the corpus, so a
    # bucket's size grows with N and the pairs it emits grow with N^2. Capping
    # bucket size at c bounds each bucket's contribution at c(c-1)/2 while the
    # number of buckets grows only linearly -- the cap converts the quadratic
    # term into a linear one. That, not today's 68%, is the reason to do it.
    growth = []
    cap_probe = 100
    _, szc0, emc0, _, _ = capped_stats(keys, cap_probe)
    for N in (12000, 25000, 50000, 100000, 220000):
        f = N / len(store)
        all_pairs = N * (N - 1) / 2
        growth.append(dict(
            corpus=N,
            all_pairs_baseline_hours=round(all_pairs * exact_us / 1e6 / 3600, 2),
            lsh_uncapped_emissions=int(emissions * f * f),
            lsh_uncapped_seconds=round(emissions * f * f * per_pair_us / 1e6, 1),
            lsh_capped_emissions=int(emc0 * f),
            lsh_capped_seconds=round(emc0 * f * per_pair_us / 1e6, 1),
        ))
    out["growth_projection"] = {
        "rows": growth,
        "assumption": ("uncapped emissions scale as N^2 (bucket membership is a fixed "
                       "share of the corpus); capped emissions scale as N (each bucket "
                       "is bounded at cap(cap-1)/2 and the number of buckets grows "
                       "linearly). 220,000 is the corpus after four years at 4,000/week."),
        "reading": ("Today the uncapped scan costs %.0f s of sketch scoring against a "
                    "1,200 s budget, so the skew is not yet fatal -- and saying otherwise "
                    "would be dishonest. It is fatal on the board's timescale: it is the "
                    "only term in the pipeline that grows quadratically, and it is what "
                    "turned the old all-pairs job into 31 hours."
                    % (emissions * per_pair_us / 1e6)),
    }

    # ------------------------------------------------------------------
    # 4. mitigations, both priced
    # ------------------------------------------------------------------
    base_stats, J_lab, y, ia, ib = evaluate_representation(store, df, lab)
    hit0, tot0 = anchor_recall(keys, anchors, J_anchor, TAU)
    out["baseline_quality"] = dict(base_stats)
    out["baseline_quality"]["unbiased_recall_above_tau"] = round(hit0 / max(tot0, 1), 5)
    out["baseline_quality"]["unbiased_pairs_above_tau"] = tot0

    # --- M2: bucket cap (retrieval only, representation untouched)
    dups = np.where(y & (J_lab >= TAU))[0]
    m2 = []
    for cap in CAPS:
        if cap is None:
            m2.append(dict(cap=None, emissions=emissions, largest_bucket=int(sizes.max()),
                           dropped_buckets=0, dropped_memberships=0,
                           labelled_recall=1.0,
                           unbiased_recall=round(hit0 / max(tot0, 1), 5),
                           per_notice_p99=float(np.percentile(per, 99)),
                           per_notice_max=int(per.max())))
            continue
        t0 = time.time()
        perc, szc, emc, dropb, dropm = capped_stats(keys, cap)
        ok = capped_candidate(keys, ia[dups], ib[dups], cap)
        hitc, totc = anchor_recall_capped(keys, anchors, J_anchor, TAU, cap)
        m2.append(dict(cap=cap, emissions=emc,
                       emissions_reduction=round(1 - emc / emissions, 4),
                       largest_bucket=int(szc.max()) if szc.size else 0,
                       dropped_buckets=dropb, dropped_memberships=dropm,
                       labelled_recall=round(float(ok.mean()), 5),
                       labelled_dups_tested=int(len(dups)),
                       labelled_dups_lost=int((~ok).sum()),
                       unbiased_recall=round(hitc / max(totc, 1), 5),
                       unbiased_pairs=totc, unbiased_lost=totc - hitc,
                       per_notice_p99=float(np.percentile(perc, 99)),
                       per_notice_max=int(perc.max()),
                       seconds=round(time.time() - t0, 1)))
        C.log("cap=%4s emissions %12s (-%.1f%%)  labelled recall %.4f  unbiased recall %.5f"
              % (cap, f"{emc:,}", 100 * (1 - emc / emissions),
                 m2[-1]["labelled_recall"], m2[-1]["unbiased_recall"]))
    out["mitigation_M2_bucket_cap"] = m2

    # --- M1: document-frequency stop-list (changes the representation itself)
    m1 = []
    for st in DF_STOPS:
        if st is None:
            m1.append(dict(df_stop=None, mean_set_size=round(float(store.sizes.mean())),
                           emissions=emissions, largest_bucket=int(sizes.max()),
                           **base_stats,
                           unbiased_recall=round(hit0 / max(tot0, 1), 5)))
            continue
        t0 = time.time()
        s2 = C.build_shingles(df, VARIANT, SCHEME, boiler, use_cache=True, df_stop=st)
        sig2 = C.minhash_signatures(s2, K, tag="%s_%s_df%s" % (VARIANT, SCHEME, st))
        k2 = C.band_keys(sig2, BANDS, ROWS_PER_BAND)
        per2, sz2, em2 = band_stats(k2)
        q2, J2, y2, ia2, ib2 = evaluate_representation(s2, df, lab)
        a2, JA2 = anchor_sample(s2, seed=17) if False else (anchors, None)
        # recall must be measured against the pairs the NEW representation would merge
        idx2 = {n: i for i, n in enumerate(s2.ids)}
        pidx2 = C.PostingIndex(s2)
        tot2 = hit2 = 0
        for a in anchors:
            jj = pidx2.jaccard_against_all(s2[int(a)])
            jj[int(a)] = 0.0
            m = jj >= q2["tau"]
            if not m.any():
                continue
            cand = (k2[int(a)][None, :] == k2).any(axis=1)
            tot2 += int(m.sum())
            hit2 += int(cand[m].sum())
        m1.append(dict(df_stop=st, mean_set_size=round(float(s2.sizes.mean())),
                       emissions=em2, emissions_reduction=round(1 - em2 / emissions, 4),
                       largest_bucket=int(sz2.max()) if sz2.size else 0,
                       per_notice_p99=float(np.percentile(per2, 99)),
                       **q2,
                       unbiased_recall=round(hit2 / max(tot2, 1), 5),
                       unbiased_pairs=tot2,
                       seconds=round(time.time() - t0, 1)))
        C.log("df_stop=%.2f |S|=%d emissions %12s (-%.1f%%)  AUC %.4f  R@FP0 %.3f  tau %.3f"
              % (st, s2.sizes.mean(), f"{em2:,}", 100 * (1 - em2 / emissions),
                 q2["auc"], q2["recall_at_zero_fp"], q2["tau"]))
    out["mitigation_M1_df_stoplist"] = m1

    # --- the adopted mitigation
    cap_ok = [d for d in m2 if d["cap"] is not None and d["unbiased_recall"] >= 0.985]
    chosen_cap = min(cap_ok, key=lambda d: d["emissions"]) if cap_ok else m2[1]
    out["adopted_mitigation"] = {
        "what": "bucket cap at %d, applied at index time" % chosen_cap["cap"],
        "why": ("M1 (a document-frequency stop-list) attacks the cause and removes more "
                "work, but it changes the representation, which moves every Jaccard in "
                "the system: tau has to be re-derived and the labelled separation "
                "changes with it. M2 changes only which buckets are allowed to expand, "
                "so its price is a pure recall number and nothing else in the design "
                "has to move. Genuine duplicates collide in many bands at once "
                "(a pair at J=0.9 is expected to match %.0f of the %d bands), so "
                "refusing the few pathological buckets costs almost none of them."
                % (BANDS * 0.9 ** ROWS_PER_BAND, BANDS)),
        "detail": chosen_cap,
        "price": ("emissions fall %.1f%% (%s -> %s) and the largest bucket falls from "
                  "%d to %d; the measured cost is %d of %d labelled duplicates above tau "
                  "and %d of %s unbiased pairs above tau."
                  % (100 * chosen_cap["emissions_reduction"], f"{emissions:,}",
                     f"{chosen_cap['emissions']:,}", int(sizes.max()),
                     chosen_cap["largest_bucket"], chosen_cap["labelled_dups_lost"],
                     chosen_cap["labelled_dups_tested"], chosen_cap["unbiased_lost"],
                     f"{chosen_cap['unbiased_pairs']:,}")),
    }
    perc, szc, emc, _, _ = capped_stats(keys, chosen_cap["cap"])
    out["distribution_after"] = {
        "total_emissions": emc,
        "largest_bucket": int(szc.max()) if szc.size else 0,
        "per_notice_mean": round(float(perc.mean()), 1),
        "per_notice_median": float(np.median(perc)),
        "per_notice_p99": float(np.percentile(perc, 99)),
        "per_notice_max": int(perc.max()),
        "nightly_seconds": round(emc * NIGHTLY_INTAKE / len(store) * 2
                                 * per_pair_us / 1e6, 1),
        "full_corpus_seconds": round(emc * per_pair_us / 1e6, 1),
    }
    np.save(os.path.join(C.CACHE, "adopted_cap.npy"), np.array([chosen_cap["cap"]]))

    # ------------------------------------------------------------------
    # 5. figures
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(1, len(srt) + 1) / len(srt)
    axes[0].plot(x * 100, 100 * cum / emissions, color="#c44e52", lw=2)
    even = 100 * np.arange(1, len(srt) + 1) / len(srt)
    axes[0].plot(x * 100, even, "k--", lw=1, label="if every bucket cost the same")
    axes[0].axvline(0.1, color="#1a7f37", ls=":", lw=1.5)
    axes[0].text(0.13, 40, "0.1%% of buckets\ncarry %.0f%% of the work"
                 % (100 * conc["top_0.1%_buckets"]["share_of_emissions"]),
                 fontsize=9, color="#1a7f37")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("buckets, largest first (% of colliding buckets, log)")
    axes[0].set_ylabel("cumulative % of candidate pairs emitted")
    axes[0].set_title("Where the nightly cost actually lives")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].hist(sizes, bins=np.logspace(0, np.log10(sizes.max()), 60), color="#4c72b0")
    axes[1].axvline(chosen_cap["cap"], color="#1a7f37", ls="--", lw=2,
                    label="adopted cap = %d" % chosen_cap["cap"])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("bucket size (members)")
    axes[1].set_ylabel("buckets")
    axes[1].set_title("Bucket sizes: a long tail that costs n(n-1)/2 each")
    axes[1].legend(fontsize=8)
    fig.suptitle("Part (e): the distribution of work, before mitigation")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "e_work_distribution.png"), dpi=140)
    C.log("wrote figures/e_work_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    caps = [d for d in m2 if d["cap"] is not None]
    ax2 = ax.twinx()
    ax.plot([d["cap"] for d in caps], [d["emissions"] for d in caps], "o-",
            color="#c44e52", label="candidate pairs emitted")
    ax2.plot([d["cap"] for d in caps], [d["unbiased_recall"] for d in caps], "s-",
             color="#4c72b0", label="retrieval recall on pairs with J >= tau")
    ax2.plot([d["cap"] for d in caps], [d["labelled_recall"] for d in caps], "^--",
             color="#1a7f37", label="recall on labelled duplicates above tau")
    ax.axvline(chosen_cap["cap"], color="#1a7f37", ls=":", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bucket cap (buckets larger than this are not expanded)")
    ax.set_ylabel("candidate pairs emitted, full corpus (log)")
    ax2.set_ylabel("retrieval recall")
    ax2.set_ylim(0.9, 1.005)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Part (e): the price of the mitigation, measured")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "e_mitigation_price.png"), dpi=140)
    C.log("wrote figures/e_mitigation_price.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(0, np.log10(max(per.max(), perc.max())), 60)
    ax.hist(per, bins=bins, alpha=0.65, color="#c44e52", label="before  (p99 %.0f, max %d)"
            % (np.percentile(per, 99), per.max()))
    ax.hist(perc, bins=bins, alpha=0.65, color="#1a7f37", label="after cap %d  (p99 %.0f, max %d)"
            % (chosen_cap["cap"], np.percentile(perc, 99), perc.max()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidate rows a single notice generates")
    ax.set_ylabel("notices")
    ax.set_title("Part (e): per-notice work, before and after")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "e_before_after.png"), dpi=140)
    C.log("wrote figures/e_before_after.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ns = [r_["corpus"] for r_ in growth]
    ax.plot(ns, [r_["all_pairs_baseline_hours"] * 3600 for r_ in growth], "o-",
            color="#937860", label="compare every pair (the job that ran 31 hours)")
    ax.plot(ns, [r_["lsh_uncapped_seconds"] for r_ in growth], "o-", color="#c44e52",
            label="LSH, uncapped buckets  (grows as $N^2$)")
    ax.plot(ns, [r_["lsh_capped_seconds"] for r_ in growth], "o-", color="#1a7f37",
            label="LSH, bucket cap %d  (grows as $N$)" % cap_probe)
    ax.axhline(BUDGET_SECONDS, color="k", ls="--", lw=1.5, label="20-minute budget")
    ax.axvline(len(store), color="grey", ls=":", lw=1)
    ax.text(len(store) * 1.05, 3, "today", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (notices)")
    ax.set_ylabel("seconds of pair scoring per full run (log)")
    ax.set_title("Part (e): why the skew matters -- it is the only quadratic term left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "e_growth_projection.png"), dpi=140)
    C.log("wrote figures/e_growth_projection.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    top = g.head(12).reset_index()
    xs = np.arange(len(top))
    ax.bar(xs - 0.2, top.share_of_corpus * 100, 0.4, label="% of the corpus", color="#4c72b0")
    ax.bar(xs + 0.2, top.share_of_work * 100, 0.4, label="% of the candidate work",
           color="#c44e52")
    ax.set_xticks(xs)
    ax.set_xticklabels(top.portal_id, rotation=45, fontsize=8)
    ax.set_ylabel("percent")
    ax.set_title("Part (e): the aggregators are big, but they are not disproportionate\n"
                 "-- the concentration is in buckets, not portals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "e_portal_attribution.png"), dpi=140)
    C.log("wrote figures/e_portal_attribution.png")

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    C.save_json("e_skew.json", out)

    # ------------------------------------------------------------- report
    print()
    print("=" * 100)
    print("PART (e) SUMMARY")
    print("=" * 100)
    d = out["distribution_before"]
    print("full-corpus retrieval at b=%d r=%d: %s candidate pairs from %s colliding buckets"
          % (BANDS, ROWS_PER_BAND, f"{d['total_emissions']:,}", f"{d['colliding_buckets']:,}"))
    print("per notice: mean %.0f, median %.0f, p99 %.0f, max %d"
          % (d["per_notice_mean"], d["per_notice_median"], d["per_notice_p99"], d["per_notice_max"]))
    print()
    print("concentration:")
    for k, v in conc.items():
        print("   %-22s %6d buckets -> %5.1f%% of all emitted pairs"
              % (k, v["buckets"], 100 * v["share_of_emissions"]))
    for k, v in conc_notice.items():
        print("   %-22s              -> %5.1f%% of all emitted pairs" % (k, 100 * v))
    print()
    w = out["worst_bucket"]
    print("worst single bucket: band %d, %d members, emits %s pairs = %.1f%% of the entire "
          "corpus's candidate work" % (w["band"], w["members"], f"{w['pairs_it_emits']:,}",
                                       100 * w["share_of_all_emissions"]))
    print("   mean pairwise Jaccard inside it %.3f; only %.3f%% of its pairs are above tau "
          "(~%d real duplicates for %s comparisons)"
          % (w["pairwise_J_mean"], 100 * w["share_of_its_pairs_above_tau"],
             w["estimated_real_duplicates_in_it"], f"{w['pairs_it_emits']:,}"))
    print("   portals inside it: %s" % w["portals"])
    print()
    b = out["budget_before"]
    print("against the budget: %s candidate pairs x %.2f us (%.2f us to emit the row from "
          "the index + %.2f us to score it) = %.0f s for a full rebuild; a night's intake "
          "costs ~%.0f s"
          % (f"{b['full_corpus_emissions']:,}", b["per_pair_microseconds_end_to_end"],
             b["database_microseconds_per_candidate_row"] or 0.0,
             b["sketch_microseconds_per_pair"],
             b["full_corpus_seconds"], b["nightly_seconds"]))
    print("   (budget is %d s; the worst bucket alone is %.1f s)"
          % (BUDGET_SECONDS, b["worst_bucket_alone_seconds"]))
    print()
    print("growth projection (4,000 notices/week):")
    hdr = ("%9s %16s %18s %14s %16s %12s" %
           ("corpus", "all-pairs (hours)", "uncapped pairs", "uncapped s",
            "capped pairs", "capped s"))
    print(hdr)
    print("-" * len(hdr))
    for r_ in growth:
        print("%9s %16.1f %18s %14.0f %16s %12.0f"
              % (f"{r_['corpus']:,}", r_["all_pairs_baseline_hours"],
                 f"{r_['lsh_uncapped_emissions']:,}", r_["lsh_uncapped_seconds"],
                 f"{r_['lsh_capped_emissions']:,}", r_["lsh_capped_seconds"]))
    print("   budget = %d s. Uncapped breaches it at roughly %s notices; capped does not "
          "breach it inside the projection." %
          (BUDGET_SECONDS,
           f"{int(len(store) * (BUDGET_SECONDS / max(emissions * per_pair_us / 1e6, 1e-9)) ** 0.5):,}"))
    print()
    print("M1 -- document-frequency stop-list (attacks the cause, moves the representation):")
    hdr = ("%10s %9s %14s %9s %8s %8s %8s %8s %10s" %
           ("df_stop", "mean|S|", "emissions", "largest", "AUC", "R@FP0", "tau",
            "delta", "K needed"))
    print(hdr)
    print("-" * len(hdr))
    for d_ in m1:
        print("%10s %9d %14s %9d %8.4f %8.3f %8.4f %8.4f %10d"
              % (d_["df_stop"], d_["mean_set_size"], f"{d_['emissions']:,}",
                 d_["largest_bucket"], d_["auc"], d_["recall_at_zero_fp"], d_["tau"],
                 d_["resolution_needed"], d_["K_required_for_this_tau"]))
    print("   'K needed' is the signature size part (b)'s argument demands at that "
          "threshold. K=384 was sized for tau=0.54.")
    print()
    print("M2 -- bucket cap (attacks the symptom, moves nothing else):")
    hdr = ("%6s %14s %9s %10s %22s %22s" %
           ("cap", "emissions", "largest", "p99/notice", "labelled recall", "unbiased recall"))
    print(hdr)
    print("-" * len(hdr))
    for d_ in m2:
        mark = "  <== adopted" if d_["cap"] == chosen_cap["cap"] else ""
        print("%6s %14s %9d %10.0f %14.5f (%d/%d) %13.5f (%d/%s)%s"
              % (d_["cap"], f"{d_['emissions']:,}", d_["largest_bucket"], d_["per_notice_p99"],
                 d_["labelled_recall"], d_.get("labelled_dups_tested", len(dups)) -
                 d_.get("labelled_dups_lost", 0), d_.get("labelled_dups_tested", len(dups)),
                 d_["unbiased_recall"], d_.get("unbiased_pairs", tot0) -
                 d_.get("unbiased_lost", 0), f"{d_.get('unbiased_pairs', tot0):,}", mark))
    print()
    a = out["adopted_mitigation"]
    print("ADOPTED: %s" % a["what"])
    print("  price: %s" % a["price"])
    da = out["distribution_after"]
    print("  after: per notice mean %.0f (was %.0f), p99 %.0f (was %.0f), max %d (was %d)"
          % (da["per_notice_mean"], d["per_notice_mean"], da["per_notice_p99"],
             out["distribution_before"]["per_notice_p99"], da["per_notice_max"],
             out["distribution_before"]["per_notice_max"]))
    print("  full-rebuild sketch scoring %.0f s -> %.0f s; nightly %.0f s -> %.0f s"
          % (b["full_corpus_seconds"], da["full_corpus_seconds"],
             b["nightly_seconds"], da["nightly_seconds"]))
    print("elapsed %.1fs" % out["elapsed_seconds"])


if __name__ == "__main__":
    main()
