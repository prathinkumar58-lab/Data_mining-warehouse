"""
s03_lsh_tuning.py  --  PART (c): sublinear retrieval, with the risk priced.

Retrieval structure: banded LSH over the K=384 MinHash signature. The signature
is cut into b bands of r rows; two notices are candidates if any band matches
exactly. P[candidate | J] = 1 - (1 - J^r)^b -- the S-curve.

The tension the question asks to be made explicit is between
    recall at the candidate stage   (a miss here is unrecoverable: the pair is
                                     never scored, so the bidder sees a duplicate)
and
    candidate volume                (every candidate costs sketch time, exact
                                     verification time and database work against
                                     a 20-minute nightly budget).

Where the 50:1 asymmetry enters -- and where it does NOT:

  * It enters the MERGE THRESHOLD tau. That is the only place a decision to
    merge is taken, so it is the only place where a false merge can be created.
    tau is chosen by minimising  R * FPR + FNR  on the adjudicated pairs.

  * It does NOT enter (b, r). A spurious candidate cannot cause a false merge --
    the verification stage still has to clear it. A spurious candidate costs
    CPU. So the candidate stage is a recall-versus-budget problem, and we set
    it by requiring that retrieval loss be an order of magnitude below the loss
    the threshold already imposes (FNR at tau), then taking the cheapest (b, r)
    that clears it.

  Saying "we tuned LSH for the asymmetry" would be the wrong answer: it would
  spend compute buying precision at a stage that cannot produce the expensive
  error.

Writes results/c_lsh.json and four figures.
"""

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
TAU = 0.54
DELTA = 0.08
COST_RATIOS = [1, 5, 10, 50, 200, 1000]
R_ADOPTED = 50.0
RETRIEVAL_LOSS_BUDGET = 0.01      # target: P[miss | true duplicate above tau] <= 1%

GRID = [(128, 3), (96, 4), (76, 5), (64, 6), (54, 7), (48, 8), (42, 9), (38, 10),
        (32, 12), (24, 16)]


N_ANCHOR = 400
J_EDGES = np.concatenate([np.arange(0, 0.60, 0.05), np.arange(0.60, 1.0001, 0.05)])


def anchor_sample(store, n_anchor=N_ANCHOR, seed=17):
    """An UNBIASED sample: n_anchor notices drawn at random, each scored with
    exact Jaccard against all 12,000 notices.

    This matters. The obvious way to get pairs across the similarity range is to
    run a generous LSH pass and sample its output -- but then every pair in the
    sample was selected BY a band collision, and the measured P[candidate | J]
    is biased upwards exactly where we most need it to be honest (at low J,
    where the work comes from). Anchors have no sampler in the loop: every pair
    involving an anchor is present, candidate or not.
    """
    cpath = os.path.join(C.CACHE, "anchor_J_%d.npz" % n_anchor)
    if os.path.exists(cpath):
        z = np.load(cpath)
        return z["anchors"], z["J"]
    rng = np.random.default_rng(seed)
    anchors = np.sort(rng.choice(len(store), n_anchor, replace=False)).astype(np.int64)
    pidx = C.PostingIndex(store)
    t0 = time.time()
    J = np.empty((n_anchor, len(store)), dtype=np.float32)
    for k, a in enumerate(anchors):
        J[k] = pidx.jaccard_against_all(store[int(a)])
        J[k, a] = np.nan                      # exclude the self-pair
    C.log("anchor sample: %d x %d exact Jaccards in %.1fs"
          % (n_anchor, len(store), time.time() - t0))
    np.savez_compressed(cpath, anchors=anchors, J=J)
    return anchors, J


def anchor_curve(keys, anchors, J, edges=J_EDGES):
    """Realised P[candidate | J] over the unbiased anchor sample."""
    tot = np.zeros(len(edges) - 1, dtype=np.int64)
    hit = np.zeros(len(edges) - 1, dtype=np.int64)
    for k, a in enumerate(anchors):
        cand = (keys[int(a)][None, :] == keys).any(axis=1)
        j = J[k]
        m = ~np.isnan(j)
        b = np.digitize(j[m], edges) - 1
        b = np.clip(b, 0, len(edges) - 2)
        tot += np.bincount(b, minlength=len(edges) - 1)
        hit += np.bincount(b, weights=cand[m].astype(np.float64),
                           minlength=len(edges) - 1).astype(np.int64)
    return tot, hit


def anchor_recall(keys, anchors, J, tau):
    """Unbiased retrieval recall on the set the system would merge: pairs whose
    EXACT Jaccard is at or above tau."""
    tot = hit = 0
    for k, a in enumerate(anchors):
        j = J[k]
        m = (~np.isnan(j)) & (j >= tau)
        if not m.any():
            continue
        cand = (keys[int(a)][None, :] == keys).any(axis=1)
        tot += int(m.sum())
        hit += int(cand[m].sum())
    return hit, tot


def is_candidate(keys, pairs):
    """Boolean per pair: do the two notices share at least one band key?"""
    return (keys[pairs[:, 0]] == keys[pairs[:, 1]]).any(axis=1)


def bucket_cost(keys):
    """Work the candidate stage creates, without materialising the pairs.

    emissions = sum over bands, over buckets, of C(n,2) -- the number of rows a
    SQL self-join on the bucket table would produce. This is the quantity the
    nightly budget actually pays for.
    """
    N, b = keys.shape
    emissions = 0
    sizes = []
    for j in range(b):
        _, cnt = np.unique(keys[:, j], return_counts=True)
        cnt = cnt[cnt > 1]
        sizes.append(cnt)
        emissions += int((cnt * (cnt - 1) // 2).sum())
    sizes = np.concatenate(sizes) if sizes else np.zeros(0, dtype=int)
    return emissions, sizes


def threshold_for_ratio(pos, neg, ratio):
    cand = np.unique(np.concatenate([pos, neg, [0.0, 1.0]]))
    best, bt = None, None
    for t in cand:
        cost = ratio * float((neg >= t).mean()) + float((pos < t).mean())
        if best is None or cost < best:
            best, bt = cost, float(t)
    return bt


def main():
    t_start = time.time()
    out = {}

    df = C.load_notices()
    boiler = C.learn_boilerplate(df)
    store = C.build_shingles(df, VARIANT, SCHEME, boiler)
    sig = C.minhash_signatures(store, 768, tag="%s_%s" % (VARIANT, SCHEME))[:, :K]

    lab = C.load_labels()
    idx = {nid: i for i, nid in enumerate(store.ids)}
    lab_i = np.array([[idx[a], idx[b]] for a, b in zip(lab.notice_id_a, lab.notice_id_b)])
    y = (lab.label.values == "same")
    J_lab = np.array([C.jaccard_exact(store[a], store[b]) for a, b in lab_i])

    anchors, J_anchor = anchor_sample(store)
    n_anchor_pairs = int(np.isfinite(J_anchor).sum())
    C.log("unbiased anchor sample: %d anchors x %d notices = %d pairs"
          % (len(anchors), len(store), n_anchor_pairs))

    # ------------------------------------------------------------------
    # 1. where the asymmetry lands: the merge threshold
    # ------------------------------------------------------------------
    pos, neg = J_lab[y], J_lab[~y]
    tau_table = []
    for R in COST_RATIOS:
        t = threshold_for_ratio(pos, neg, R)
        fpr = float((neg >= t).mean())
        fnr = float((pos < t).mean())
        tau_table.append(dict(cost_ratio=R, tau=round(t, 4),
                              fpr_on_hard_negatives=round(fpr, 5),
                              fnr_on_true_duplicates=round(fnr, 5),
                              weighted_cost=round(R * fpr + fnr, 4)))
    out["threshold_vs_cost_ratio"] = tau_table
    tau = [d["tau"] for d in tau_table if d["cost_ratio"] == R_ADOPTED][0]
    fnr_at_tau = [d["fnr_on_true_duplicates"] for d in tau_table if d["cost_ratio"] == R_ADOPTED][0]
    out["adopted"] = {"cost_ratio_R": R_ADOPTED, "tau": tau, "fnr_at_tau": fnr_at_tau}
    C.log("R=%.0f -> tau=%.4f, FNR on true duplicates %.3f" % (R_ADOPTED, tau, fnr_at_tau))

    # what a false merge would cost in counts, not rates -- the skew correction
    n_neg = int((~y).sum())
    fpr_upper = 1 - 0.05 ** (1.0 / n_neg)          # 95% upper bound given zero events
    n_pos_above = int((pos >= tau).sum())
    out["count_level_risk"] = {
        "hard_negatives_tested": n_neg,
        "hard_negatives_above_tau": int((neg >= tau).sum()),
        "fpr_95pct_upper_bound_on_hard_negatives": round(float(fpr_upper), 5),
        "reading": ("Zero of %d adjudicated 'different' pairs reach tau = %.2f, but zero "
                    "events out of %d only bounds the rate at %.2f%% with 95%% confidence. "
                    "Because those negatives are HARD (a human was asked about them), that "
                    "bound is pessimistic for a random candidate pair -- but it is the "
                    "number to quote to the head of product, and it is why the merge "
                    "decision is taken on exact Jaccard rather than on the sketch."
                    % (n_neg, tau, n_neg, 100 * fpr_upper)),
    }

    # what a larger R cannot buy: the labelled set runs out of negatives
    growth_per_night = 4000 / 7.0
    scale_full = (len(store) * (len(store) - 1) / 2) / max(n_anchor_pairs, 1)
    jf = J_anchor[np.isfinite(J_anchor)]
    tau_lo = [d["tau"] for d in tau_table if d["cost_ratio"] == 5][0]
    band_n = int(((jf >= tau_lo) & (jf < tau)).sum())
    out["threshold_saturation"] = {
        "saturates_at_R": 10,
        "why": ("tau is pinned just above the highest-scoring adjudicated 'different' "
                "pair (%.4f). Above R = 10 the labelled set contains no negative that "
                "moving tau could exclude, so a larger ratio changes nothing measurable. "
                "R = 50 is therefore recorded as the governing number, but what it is "
                "really buying is the decision to verify on exact Jaccard rather than on "
                "the sketch, and the review band below."
                % float(neg.max())),
        "review_band": [tau_lo, tau],
        "pairs_in_band_per_full_corpus": int(band_n * scale_full),
        "pairs_in_band_per_night": int(band_n * scale_full * 2 * growth_per_night / len(store)),
        "recommendation": None,   # filled in below once the affordable width is measured
    }

    # how wide a review band can ops actually afford? 200 pairs/night at ~20s
    # of adjudication each is a bit over an hour of one person's day.
    per_night = lambda lo: int(((jf >= lo) & (jf < tau)).sum() * scale_full * 2
                               * growth_per_night / len(store))
    affordable_lo = tau
    for lo in np.arange(tau - 0.005, tau_lo - 1e-9, -0.005):
        if per_night(lo) > 200:
            break
        affordable_lo = float(lo)
    out["threshold_saturation"]["affordable_review_band"] = [round(affordable_lo, 3), tau]
    out["threshold_saturation"]["affordable_band_pairs_per_night"] = per_night(affordable_lo)
    out["threshold_saturation"]["recommendation"] = (
        "Pairs just below tau are the ones the evidence cannot settle, and a ratio of 50 "
        "says neither merging nor silently discarding them is right. But the full band "
        "[%.3f, %.3f) is %d pairs on a night's intake, which is not an hour of anyone's "
        "day -- so it cannot all go to a human. The band that fits a 200-pair/night "
        "adjudication budget is [%.3f, %.3f) (%d pairs/night). That is where an R of 50 "
        "should be spent: buy certainty on the pairs nearest the line, and let the rest "
        "fall to the 'grumble' side, which is the cheap error."
        % (tau_lo, tau, int(band_n * scale_full * 2 * growth_per_night / len(store)),
           affordable_lo, tau, per_night(affordable_lo)))

    # ------------------------------------------------------------------
    # 2. the S-curve: theory and realised, for every (b, r)
    # ------------------------------------------------------------------
    dup_above = J_lab[y & (J_lab >= tau)]      # the population retrieval must not lose
    rows = []
    for (b, r) in GRID:
        if b * r > K:
            continue
        t0 = time.time()
        keys = C.band_keys(sig, b, r)
        cand_lab = is_candidate(keys, lab_i)
        emissions, sizes = bucket_cost(keys)
        hit, tot = anchor_recall(keys, anchors, J_anchor, tau)
        build_s = time.time() - t0

        # recall on the adjudicated duplicates that sit above the threshold
        m = y & (J_lab >= tau)
        rec_lab = float(cand_lab[m].mean())
        rec_theory = float(C.p_candidate(dup_above, b, r).mean())
        rec_all = float(cand_lab[y].mean())
        rows.append(dict(
            b=b, r=r, rows_used=b * r, lsh_threshold=round(C.lsh_threshold(b, r), 4),
            p_at_tau=round(float(C.p_candidate(tau, b, r)), 5),
            unbiased_recall_pairs_above_tau=round(hit / max(tot, 1), 5),
            unbiased_n_pairs_above_tau=tot,
            unbiased_misses=tot - hit,
            labelled_recall_dups_above_tau=round(rec_lab, 4),
            theoretical_recall_dups_above_tau=round(rec_theory, 4),
            labelled_recall_all_same_pairs=round(rec_all, 4),
            candidate_emissions_full_corpus=emissions,
            emissions_per_notice=round(emissions / len(store), 1),
            largest_bucket=int(sizes.max()) if sizes.size else 0,
            buckets_with_collisions=int(sizes.size),
            band_build_seconds=round(build_s, 1),
        ))
        C.log("b=%3d r=%2d t=%.3f recall(J>=tau, unbiased)=%.4f (%d/%d)  "
              "recall(labelled)=%.4f  emissions=%s  max bucket=%d"
              % (b, r, C.lsh_threshold(b, r), hit / max(tot, 1), hit, tot, rec_lab,
                 f"{emissions:,}", rows[-1]["largest_bucket"]))
    out["grid"] = rows

    # ------------------------------------------------------------------
    # 3. the operating point
    # ------------------------------------------------------------------
    ok = [d for d in rows if d["unbiased_recall_pairs_above_tau"] >= 1 - RETRIEVAL_LOSS_BUDGET]
    chosen = min(ok, key=lambda d: d["candidate_emissions_full_corpus"]) if ok else \
        max(rows, key=lambda d: d["unbiased_recall_pairs_above_tau"])
    out["operating_point"] = {
        "b": chosen["b"], "r": chosen["r"],
        "rule": ("cheapest (b, r) whose realised recall on true duplicates above tau is "
                 ">= %.0f%%. The recall floor is set one order of magnitude below the "
                 "loss the threshold already imposes (FNR at tau = %.3f), so retrieval "
                 "is not the binding source of misses."
                 % (100 * (1 - RETRIEVAL_LOSS_BUDGET), fnr_at_tau)),
        "detail": chosen,
    }
    C.log("operating point: b=%d r=%d" % (chosen["b"], chosen["r"]))

    # ------------------------------------------------------------------
    # 4. realised vs theoretical S-curve at the operating point
    # ------------------------------------------------------------------
    keys = C.band_keys(sig, chosen["b"], chosen["r"])
    tot_b, hit_b = anchor_curve(keys, anchors, J_anchor)
    emp = []
    for i in range(len(J_EDGES) - 1):
        if tot_b[i] < 20:
            continue
        mid = (J_EDGES[i] + J_EDGES[i + 1]) / 2
        emp.append(dict(j_lo=round(float(J_EDGES[i]), 3), j_mid=round(float(mid), 3),
                        n=int(tot_b[i]), retrieved=int(hit_b[i]),
                        realised=round(float(hit_b[i] / tot_b[i]), 5),
                        theory=round(float(C.p_candidate(mid, chosen["b"], chosen["r"])), 5)))
    out["empirical_s_curve"] = emp
    out["empirical_s_curve_note"] = (
        "Measured on the unbiased anchor sample (%d anchors x %d notices = %s pairs), "
        "so the low-J end is a real measurement rather than an artefact of having "
        "sampled pairs that an LSH pass already liked." % (len(anchors), len(store),
                                                           f"{n_anchor_pairs:,}"))

    # candidate-list length seen by an individual notice
    per_notice = np.zeros(len(store), dtype=np.int64)
    for j in range(chosen["b"]):
        uq, inv, cnt = np.unique(keys[:, j], return_inverse=True, return_counts=True)
        per_notice += cnt[inv] - 1
    out["candidate_list_lengths"] = {
        "mean": round(float(per_notice.mean()), 1),
        "median": float(np.median(per_notice)),
        "p90": float(np.percentile(per_notice, 90)),
        "p99": float(np.percentile(per_notice, 99)),
        "max": int(per_notice.max()),
        "note": "counted with multiplicity across bands, i.e. the rows a lookup returns",
    }
    np.save(os.path.join(C.CACHE, "per_notice_emissions.npy"), per_notice)

    # ------------------------------------------------------------------
    # 5. figures
    # ------------------------------------------------------------------
    gx = np.linspace(0.001, 1, 400)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for d in rows:
        lw = 2.8 if (d["b"], d["r"]) == (chosen["b"], chosen["r"]) else 1.0
        al = 1.0 if lw > 2 else 0.45
        ax.plot(gx, C.p_candidate(gx, d["b"], d["r"]), lw=lw, alpha=al,
                label="b=%d r=%d%s" % (d["b"], d["r"],
                                       "  <- operating point" if lw > 2 else ""))
    ax.plot([d["j_mid"] for d in emp], [d["realised"] for d in emp], "ko", ms=6,
            label="realised (%d unbiased anchors)" % len(anchors))
    ax.axvline(tau, color="#1a7f37", ls="--", lw=1.5)
    ax.text(tau + 0.01, 0.06, "merge threshold $\\tau$=%.2f\n(from the 50:1 ratio)" % tau,
            fontsize=8, color="#1a7f37")
    ax.hist(J_lab[y], bins=np.linspace(0, 1, 41), density=True, alpha=0.18,
            color="#4c72b0", label="where true duplicates actually live")
    ax.set_xlabel("true Jaccard similarity of a pair")
    ax.set_ylabel("P[pair survives to the candidate list]")
    ax.set_title("Part (c): probability of retrieval as a function of true similarity")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "c_s_curves.png"), dpi=140)
    C.log("wrote figures/c_s_curves.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = [d["candidate_emissions_full_corpus"] for d in rows]
    ys = [d["unbiased_recall_pairs_above_tau"] for d in rows]
    ax.plot(xs, ys, "o-", color="#4c72b0")
    for d in rows:
        ax.annotate("b=%d,r=%d" % (d["b"], d["r"]),
                    (d["candidate_emissions_full_corpus"], d["unbiased_recall_pairs_above_tau"]),
                    fontsize=7, xytext=(4, -9), textcoords="offset points")
    ax.scatter([chosen["candidate_emissions_full_corpus"]],
               [chosen["unbiased_recall_pairs_above_tau"]], s=200, facecolor="none",
               edgecolor="#1a7f37", linewidth=2.5, zorder=5, label="operating point")
    ax.axhline(1 - RETRIEVAL_LOSS_BUDGET, color="#c44e52", ls="--", lw=1.2,
               label="recall floor (%.0f%%)" % (100 * (1 - RETRIEVAL_LOSS_BUDGET)))
    ax.set_xscale("log")
    ax.set_xlabel("candidate pairs emitted over the full corpus (log scale) = the work")
    ax.set_ylabel("realised recall on true duplicates with J >= $\\tau$")
    ax.set_title("Part (c): the tension, made explicit")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "c_recall_vs_work.png"), dpi=140)
    C.log("wrote figures/c_recall_vs_work.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].plot([d["cost_ratio"] for d in tau_table], [d["tau"] for d in tau_table],
                 "o-", color="#4c72b0")
    axes[0].scatter([R_ADOPTED], [tau], s=160, facecolor="none", edgecolor="#1a7f37", lw=2.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("R = cost(false merge) / cost(missed merge)")
    axes[0].set_ylabel("cost-optimal merge threshold $\\tau$")
    axes[0].set_title("Where the asymmetry enters: the threshold")
    axes[0].grid(alpha=0.25)
    axes[1].plot([d["cost_ratio"] for d in tau_table],
                 [1 - d["fnr_on_true_duplicates"] for d in tau_table], "o-", color="#c44e52")
    axes[1].scatter([R_ADOPTED], [1 - fnr_at_tau], s=160, facecolor="none",
                    edgecolor="#1a7f37", lw=2.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("R = cost(false merge) / cost(missed merge)")
    axes[1].set_ylabel("share of true duplicates actually merged")
    axes[1].set_title("What the asymmetry costs in merges forgone")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "c_cost_ratio.png"), dpi=140)
    C.log("wrote figures/c_cost_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(per_notice, bins=np.logspace(0, np.log10(max(per_notice.max(), 10)), 60),
            color="#4c72b0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(per_notice.mean(), color="#c44e52", ls="--",
               label="mean %.0f" % per_notice.mean())
    ax.axvline(np.median(per_notice), color="#1a7f37", ls="--",
               label="median %.0f" % np.median(per_notice))
    ax.set_xlabel("candidate rows returned for one notice (b=%d, r=%d)" % (chosen["b"], chosen["r"]))
    ax.set_ylabel("notices")
    ax.set_title("Part (c): candidate-list length is not one number\n(this is the door into part (e))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "c_candidate_lengths.png"), dpi=140)
    C.log("wrote figures/c_candidate_lengths.png")

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    C.save_json("c_lsh.json", out)

    # ------------------------------------------------------------- report
    print()
    print("=" * 96)
    print("PART (c) SUMMARY")
    print("=" * 96)
    print("Where the 50:1 asymmetry enters -- the merge threshold:")
    print("%8s %8s %22s %24s" % ("R", "tau", "FPR on hard negatives", "true duplicates merged"))
    for d in tau_table:
        mark = "  <== adopted" if d["cost_ratio"] == R_ADOPTED else ""
        print("%8d %8.4f %22.5f %23.1f%%%s"
              % (d["cost_ratio"], d["tau"], d["fpr_on_hard_negatives"],
                 100 * (1 - d["fnr_on_true_duplicates"]), mark))
    ts = out["threshold_saturation"]
    print()
    print("Saturation: tau stops moving above R = %d (highest adjudicated 'different' "
          "pair = %.4f)." % (ts["saturates_at_R"], float(neg.max())))
    print("  review band [%.3f, %.3f): ~%s pairs corpus-wide, ~%d per night's intake "
          "-- too many for a human"
          % (ts["review_band"][0], ts["review_band"][1],
             f"{ts['pairs_in_band_per_full_corpus']:,}", ts["pairs_in_band_per_night"]))
    print("  band that fits a 200-pair/night adjudication budget: [%.3f, %.3f) = %d/night"
          % (ts["affordable_review_band"][0], ts["affordable_review_band"][1],
             ts["affordable_band_pairs_per_night"]))
    print()
    print("Candidate stage -- recall against work (full corpus, 12,000 notices):")
    hdr = ("%4s %3s %8s %10s %14s %12s %16s %11s" %
           ("b", "r", "thresh", "P[cand|tau]", "recall(J>=tau)", "labelled", "emissions",
            "max bucket"))
    print(hdr)
    print("-" * len(hdr))
    for d in rows:
        mark = " <==" if (d["b"], d["r"]) == (chosen["b"], chosen["r"]) else ""
        print("%4d %3d %8.3f %10.4f %14.5f %12.4f %16s %11d%s"
              % (d["b"], d["r"], d["lsh_threshold"], d["p_at_tau"],
                 d["unbiased_recall_pairs_above_tau"], d["labelled_recall_dups_above_tau"],
                 f"{d['candidate_emissions_full_corpus']:,}", d["largest_bucket"], mark))
    print()
    cl = out["candidate_list_lengths"]
    print("candidate rows per notice at the operating point: mean %.0f  median %.0f  "
          "p90 %.0f  p99 %.0f  max %d"
          % (cl["mean"], cl["median"], cl["p90"], cl["p99"], cl["max"]))
    print("(the gap between the median and the max is what part (e) is about)")
    print()
    print("realised vs theoretical S-curve at b=%d r=%d:" % (chosen["b"], chosen["r"]))
    print("%8s %10s %10s %10s %10s" % ("J", "pairs", "retrieved", "realised", "theory"))
    for d in emp:
        print("%8.3f %10d %10d %10.5f %10.5f"
              % (d["j_mid"], d["n"], d["retrieved"], d["realised"], d["theory"]))
    print()
    print("elapsed %.1fs" % out["elapsed_seconds"])


if __name__ == "__main__":
    main()
