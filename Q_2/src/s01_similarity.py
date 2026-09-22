"""
s01_similarity.py  --  PART (a): define "similar", mechanically, and defend
the two representation decisions with evidence from THIS corpus.

What it does
------------
1. Reports how labelled_pairs.csv is skewed (the question tells us to find out
   before using it) and contrasts that with the corpus base rate.
2. Builds the shingle set of every notice appearing in a labelled pair under
   nine (normalisation x granularity) combinations.
3. Scores all 900 adjudicated pairs with exact Jaccard under each, and reports
   the separation achieved -- including an asymmetric-cost metric, because the
   head of product's two failure modes are not equally expensive.
4. Runs a paired test between the two granularity finalists, so the choice is
   made on a difference that is measured rather than eyeballed.
5. Prints the one-same-pair / one-different-pair walkthrough the question asks
   for, under the competing choices.

Writes results/a_similarity.json, figures/a_score_distributions.png,
figures/a_separation.png, figures/a_granularity_tradeoff.png.
"""

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

CONFIGS = [
    ("raw", "w3"), ("raw", "w5"), ("raw", "c5"),
    ("mask", "w5"), ("canon", "w3"), ("canon", "w5"),
    ("canon", "w7"), ("canon", "w9"), ("canon", "c5"),
]

LABELS = {
    ("raw", "w3"): "everything kept, word 3-grams",
    ("raw", "w5"): "everything kept, word 5-grams",
    ("raw", "c5"): "everything kept, char 5-grams",
    ("mask", "w5"): "boilerplate off, ALL numbers masked",
    ("canon", "w3"): "boilerplate off, money canonical, w3",
    ("canon", "w5"): "boilerplate off, money canonical, w5",
    ("canon", "w7"): "boilerplate off, money canonical, w7",
    ("canon", "w9"): "boilerplate off, money canonical, w9",
    ("canon", "c5"): "boilerplate off, money canonical, c5",
}

ADOPTED = ("canon", "w5")
COST_RATIO = 50.0        # see part (c): false merge : missed merge


def auc(pos, neg):
    """Rank AUC = P(score of a random same-pair > score of a random diff-pair)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    s = pd.Series(np.concatenate([pos, neg]))
    ranks = s.rank(method="average").values
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def recall_at_zero_fp(pos, neg):
    """Fraction of 'same' pairs scoring strictly above EVERY 'different' pair.

    The metric that matters under the stated asymmetry: a false merge is the
    expensive error, so what we care about is how many true duplicates we can
    still capture in the region no adjudicated 'different' pair can reach.
    """
    hi = float(np.max(neg))
    return float((np.asarray(pos) > hi).mean()), hi


def best_threshold(pos, neg, ratio):
    """Threshold minimising ratio*FPR + FNR on the labelled set (rates only;
    the count-level reweighting for the true base rate happens in part (c))."""
    cand = np.unique(np.concatenate([pos, neg]))
    best, bt = None, None
    for t in cand:
        cost = ratio * float((neg >= t).mean()) + float((pos < t).mean())
        if best is None or cost < best:
            best, bt = cost, float(t)
    return bt, best


def bootstrap_r0(pos, neg, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        out[i] = (p > q.max()).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    t_start = time.time()
    out = {}

    df = C.load_notices()
    lab = C.load_labels()
    C.log("corpus %d notices / %d portals; labels %d rows"
          % (len(df), df.portal_id.nunique(), len(lab)))

    # ---------------------------------------------------------------- skew
    n_same = int((lab.label == "same").sum())
    n_diff = int((lab.label == "different").sum())
    N = len(df)
    total_pairs = N * (N - 1) // 2
    out["label_skew"] = {
        "n_pairs": int(len(lab)),
        "n_same": n_same,
        "n_different": n_diff,
        "frac_same_in_labels": round(n_same / len(lab), 4),
        "corpus_total_pairs": int(total_pairs),
        "adjudicators": {k: int(v) for k, v in lab.adjudicated_by.value_counts().items()},
        "note": ("The labelled set is %.0f%% 'same'. The corpus cannot be: with %.2e "
                 "possible pairs, even ten thousand true duplicate pairs would be a "
                 "base rate near 1e-4. Conditional RATES (TPR, FPR) transfer from this "
                 "sample; COUNTS and precision do not, and are re-weighted in part (c). "
                 "The negatives are also 'hard': they were sent to a human precisely "
                 "because something made them look alike, so FPR measured here is "
                 "pessimistic relative to a random non-duplicate pair -- which is the "
                 "safe direction to be wrong in, given the cost asymmetry."
                 % (100 * n_same / len(lab), float(total_pairs))),
    }

    need = pd.unique(pd.concat([lab.notice_id_a, lab.notice_id_b]))
    sub = df[df.notice_id.isin(need)].reset_index(drop=True)
    C.log("labelled pairs touch %d distinct notices" % len(sub))

    # Boilerplate templates are learned from the WHOLE corpus -- that is where
    # the evidence for "this line is a template" lives -- then applied here.
    t0 = time.time()
    boiler = C.learn_boilerplate(df)
    tops = sorted(((p, len(v)) for p, v in boiler.items() if v), key=lambda kv: -kv[1])
    out["boilerplate"] = {
        "portals_with_templates": int(sum(1 for v in boiler.values() if v)),
        "top_portals": tops[:10],
        "learn_seconds": round(time.time() - t0, 2),
        "rule": ("a line repeating verbatim (case/punctuation-insensitive, digits kept) "
                 "in >60%% of a portal's notices, for portals with >=20 notices, "
                 "lines >=25 chars"),
    }

    # how much text the template rule actually removes, by portal
    strip = []
    for portal in [p for p, _ in tops[:8]]:
        rows = df[df.portal_id == portal].head(40)
        before = rows.body.str.len().mean()
        after = np.mean([len(C.normalise(b, "canon", boiler.get(portal))) for b in rows.body])
        strip.append(dict(portal=portal, mean_chars_before=round(float(before)),
                          mean_chars_after=round(float(after)),
                          removed_pct=round(100 * (1 - after / before), 1)))
    out["boilerplate"]["removal_effect"] = strip

    idx = {nid: i for i, nid in enumerate(sub.notice_id.values)}
    ia = lab.notice_id_a.map(idx).values
    ib = lab.notice_id_b.map(idx).values
    y = (lab.label.values == "same")

    # ------------------------------------------------- score every config
    scores, stats = {}, []
    for variant, scheme in CONFIGS:
        t0 = time.time()
        store = C.build_shingles(sub, variant, scheme, boiler, use_cache=False)
        build_s = time.time() - t0
        s = np.array([C.jaccard_exact(store[int(a)], store[int(b)])
                      for a, b in zip(ia, ib)])
        scores[(variant, scheme)] = s
        pos, neg = s[y], s[~y]
        a = auc(pos, neg)
        r0, ceiling = recall_at_zero_fp(pos, neg)
        lo, hi = bootstrap_r0(pos, neg)
        thr, _ = best_threshold(pos, neg, COST_RATIO)
        stats.append(dict(
            variant=variant, scheme=scheme, label=LABELS[(variant, scheme)],
            mean_set_size=round(float(store.sizes.mean())),
            build_seconds=round(build_s, 1),
            auc=round(float(a), 5),
            same_p01=round(float(np.percentile(pos, 1)), 4),
            same_p05=round(float(np.percentile(pos, 5)), 4),
            same_median=round(float(np.median(pos)), 4),
            diff_p99=round(float(np.percentile(neg, 99)), 4),
            diff_max=round(float(neg.max()), 4),
            recall_at_zero_fp=round(r0, 4),
            r0_ci95=[round(lo, 4), round(hi, 4)],
            highest_different_pair=round(ceiling, 4),
            thr_cost50=round(thr, 4),
            fpr_at_thr=round(float((neg >= thr).mean()), 5),
            fnr_at_thr=round(float((pos < thr).mean()), 5),
        ))
        C.log("%-10s AUC=%.4f R@FP0=%.3f [%.3f,%.3f] |S|=%4d  %.1fs"
              % (variant + "/" + scheme, a, r0, lo, hi, store.sizes.mean(), build_s))

    out["configs"] = stats
    out["adopted"] = {"variant": ADOPTED[0], "scheme": ADOPTED[1],
                      "score": "exact Jaccard of the two shingle sets"}

    # ------------------------------------- paired test, w5 vs w9 finalists
    d5, d9 = scores[("canon", "w5")], scores[("canon", "w9")]
    out["granularity_paired_test"] = {
        "mean_shift_same_pairs_w5_to_w9": round(float((d9 - d5)[y].mean()), 4),
        "mean_shift_diff_pairs_w5_to_w9": round(float((d9 - d5)[~y].mean()), 4),
        "reading": ("Going from 5-grams to 9-grams pushes different-pairs down four "
                    "times as far as same-pairs, so 9-grams genuinely separate better. "
                    "But the bootstrap intervals on recall-at-zero-false-merges overlap, "
                    "and 9-grams also drag the true-duplicate median down "
                    "(%.3f -> %.3f). Every point of true-duplicate similarity lost has "
                    "to be bought back with more LSH bands, and part (e) shows candidate "
                    "volume -- not scoring -- is what threatens the 20-minute budget."
                    % (np.median(d5[y]), np.median(d9[y]))),
    }

    # ------------------------------------------- the two-pair walkthrough
    raw_w5 = scores[("raw", "w5")]
    can_w5 = scores[("canon", "w5")]
    can_c5 = scores[("canon", "c5")]
    same_i = int(np.argmax(np.where(y, can_w5 - raw_w5, -9)))
    diff_i = int(np.argmax(np.where(~y, raw_w5, -9)))

    def pair_row(i):
        a_id, b_id = lab.notice_id_a.iloc[i], lab.notice_id_b.iloc[i]
        ra = df[df.notice_id == a_id].iloc[0]
        rb = df[df.notice_id == b_id].iloc[0]
        return dict(
            notice_a=a_id, notice_b=b_id, label=lab.label.iloc[i],
            portal_a=ra.portal_id, portal_b=rb.portal_id,
            title_a=str(ra.title)[:120], title_b=str(rb.title)[:120],
            value_a=int(ra.estimated_value), value_b=int(rb.estimated_value),
            body_len_a=int(len(ra.body)), body_len_b=int(len(rb.body)),
            J_raw_w5=round(float(raw_w5[i]), 4),
            J_canon_w5=round(float(can_w5[i]), 4),
            J_mask_w5=round(float(scores[("mask", "w5")][i]), 4),
            J_raw_c5=round(float(scores[("raw", "c5")][i]), 4),
            J_canon_c5=round(float(can_c5[i]), 4),
            J_canon_w9=round(float(scores[("canon", "w9")][i]), 4),
        )

    out["walkthrough"] = {"same_pair": pair_row(same_i),
                          "different_pair": pair_row(diff_i)}

    out["decision_evidence"] = {
        "decision_2_signal_vs_noise": {
            "same_mean_raw": round(float(raw_w5[y].mean()), 4),
            "same_mean_mask": round(float(scores[("mask", "w5")][y].mean()), 4),
            "same_mean_canon": round(float(can_w5[y].mean()), 4),
            "diff_mean_raw": round(float(raw_w5[~y].mean()), 4),
            "diff_mean_canon": round(float(can_w5[~y].mean()), 4),
            "same_pairs_lifted_by_canon": int(((can_w5 - raw_w5)[y] > 0.02).sum()),
            "same_pairs_hurt_by_canon": int(((can_w5 - raw_w5)[y] < -0.02).sum()),
        },
        "decision_1_granularity": {
            "word5_auc": round(float(auc(can_w5[y], can_w5[~y])), 5),
            "char5_auc": round(float(auc(can_c5[y], can_c5[~y])), 5),
            "char5_set_size_multiple": round(
                [s["mean_set_size"] for s in stats if s["variant"] == "canon" and s["scheme"] == "c5"][0] /
                [s["mean_set_size"] for s in stats if s["variant"] == "canon" and s["scheme"] == "w5"][0], 2),
        },
    }

    # -------------------------------------------------------- the figures
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    bins = np.linspace(0, 1, 51)
    for ax, (variant, scheme) in zip(axes.ravel(), CONFIGS):
        s = scores[(variant, scheme)]
        ax.hist(s[~y], bins=bins, alpha=0.7, label="different (%d)" % int((~y).sum()), color="#c44e52")
        ax.hist(s[y], bins=bins, alpha=0.7, label="same (%d)" % int(y.sum()), color="#4c72b0")
        st = [d for d in stats if d["variant"] == variant and d["scheme"] == scheme][0]
        ax.axvline(st["diff_max"], color="k", ls="--", lw=1)
        ttl = "%s/%s  AUC=%.4f  R@FP0=%.2f" % (variant, scheme, st["auc"], st["recall_at_zero_fp"])
        if (variant, scheme) == ADOPTED:
            ttl = "ADOPTED  " + ttl
            for sp in ax.spines.values():
                sp.set_color("#1a7f37"); sp.set_linewidth(2.5)
        ax.set_title(ttl, fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("exact Jaccard")
    fig.suptitle("Part (a): 900 adjudicated pairs under nine representation choices "
                 "(dashed line = highest-scoring 'different' pair)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "a_score_distributions.png"), dpi=140)
    C.log("wrote figures/a_score_distributions.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = ["%s/%s" % (d["variant"], d["scheme"]) for d in stats]
    r0s = [d["recall_at_zero_fp"] for d in stats]
    err = np.array([[d["recall_at_zero_fp"] - d["r0_ci95"][0] for d in stats],
                    [d["r0_ci95"][1] - d["recall_at_zero_fp"] for d in stats]])
    cols = ["#1a7f37" if (d["variant"], d["scheme"]) == ADOPTED else "#4c72b0" for d in stats]
    ax.barh(names, r0s, xerr=err, color=cols, capsize=3)
    for i, d in enumerate(stats):
        ax.text(0.012, i, "AUC %.4f    mean|S| %d" % (d["auc"], d["mean_set_size"]),
                va="center", fontsize=8, color="white")
    ax.set_xlabel("recall at zero false merges on the labelled set (95% bootstrap CI)")
    ax.set_title("Part (a): safe recall bought by each representation (green = adopted)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "a_separation.png"), dpi=140)
    C.log("wrote figures/a_separation.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    gran = [d for d in stats if d["variant"] == "canon" and d["scheme"].startswith("w")]
    ks = [int(d["scheme"][1:]) for d in gran]
    ax.plot(ks, [d["diff_p99"] for d in gran], "o-", color="#c44e52", label="99th pct of 'different' pairs")
    ax.plot(ks, [d["same_p05"] for d in gran], "o-", color="#4c72b0", label="5th pct of 'same' pairs")
    ax.plot(ks, [d["same_median"] for d in gran], "o--", color="#4c72b0", alpha=0.5,
            label="median 'same' pair")
    ax.fill_between(ks, [d["same_p05"] for d in gran], [d["diff_p99"] for d in gran],
                    color="grey", alpha=0.18, label="overlap band (where merges are decided)")
    ax.axvline(5, color="#1a7f37", ls=":", lw=2)
    ax.text(5.05, 0.95, "adopted k=5", color="#1a7f37", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xlabel("word n-gram length k")
    ax.set_ylabel("exact Jaccard")
    ax.set_title("Part (a), decision 1: longer n-grams separate better but sink the duplicates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "a_granularity_tradeoff.png"), dpi=140)
    C.log("wrote figures/a_granularity_tradeoff.png")

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    C.save_json("a_similarity.json", out)

    # ------------------------------------------------------------- report
    print()
    print("=" * 92)
    print("PART (a) SUMMARY   --   similarity = exact Jaccard over shingle sets")
    print("=" * 92)
    print("labelled set: %d same / %d different (%.0f%% same) against %.2e possible corpus pairs"
          % (n_same, n_diff, 100 * n_same / len(lab), float(total_pairs)))
    print("boilerplate rule fired on %d portals; on the nodal aggregators it removes:"
          % out["boilerplate"]["portals_with_templates"])
    for s in strip[:6]:
        print("    %s  %d -> %d chars  (-%.0f%%)"
              % (s["portal"], s["mean_chars_before"], s["mean_chars_after"], s["removed_pct"]))
    print()
    hdr = ("%-10s %-34s %7s %8s %7s %14s" %
           ("config", "what it does", "mean|S|", "AUC", "R@FP0", "R@FP0 95% CI"))
    print(hdr)
    print("-" * len(hdr))
    for d in stats:
        mark = "  <== ADOPTED" if (d["variant"], d["scheme"]) == ADOPTED else ""
        print("%-10s %-34s %7d %8.4f %7.3f  [%.3f, %.3f]%s"
              % (d["variant"] + "/" + d["scheme"], d["label"][:34], d["mean_set_size"],
                 d["auc"], d["recall_at_zero_fp"], d["r0_ci95"][0], d["r0_ci95"][1], mark))
    print()
    for name, key in (("SAME", "same_pair"), ("DIFFERENT", "different_pair")):
        w = out["walkthrough"][key]
        print("%s pair  %s (%s) / %s (%s)" % (name, w["notice_a"], w["portal_a"],
                                              w["notice_b"], w["portal_b"]))
        print("    title A: %s" % w["title_a"])
        print("    title B: %s" % w["title_b"])
        print("    value  : %d / %d      body chars: %d / %d"
              % (w["value_a"], w["value_b"], w["body_len_a"], w["body_len_b"]))
        print("    J(raw,w5)=%.3f   J(mask,w5)=%.3f   J(canon,w5)=%.3f   "
              "J(raw,c5)=%.3f   J(canon,c5)=%.3f"
              % (w["J_raw_w5"], w["J_mask_w5"], w["J_canon_w5"], w["J_raw_c5"], w["J_canon_c5"]))
    print()
    print("elapsed %.1fs" % out["elapsed_seconds"])


if __name__ == "__main__":
    main()
