"""Paper figures from audited result JSON only (results/*.json -> paper-overleaf/figs/*.pdf).

    python3 make_figs.py [--res DIR] [--out DIR]   # draws every figure whose input JSON exists
Figures: fig_e3p.pdf (RQ3 FWER vs number of looks), fig_rq2.pdf (RQ2 coverage), fig_c2.pdf (certified contrasts).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"R1": "#1f5fa8", "R1prime": "#0b8a6f", "SPRT0.05": "#c2641a", "SPRT0.01": "#e0a24a",
       "McNemar_peek": "#9b3d8f", "Wilson_peek": "#7a7a7a"}
LABEL = {"R1": "Comparison (ours)", "R1prime": "Sign certificate (ours)", "SPRT0.05": "SPRT α=0.05",
         "SPRT0.01": "SPRT α=0.01", "McNemar_peek": "McNemar, peeking", "Wilson_peek": "Wilson, peeking"}


def fig_e3p(res, out):
    """Null setting only: on real pools every method had FWER 0 (no discriminating power, appendix sentence)."""
    fig, ax0 = plt.subplots(1, 1, figsize=(4.6, 2.4))
    axes = [ax0]
    for ax, setting in zip(axes, ("null",)):
        byL = res["results"][setting]
        Ls = sorted(byL, key=int)
        for m in LABEL:
            ax.plot([int(L) for L in Ls], [byL[L][m]["fwer"] for L in Ls], marker="o", ms=3, lw=1.4,
                    color=COL[m], label=LABEL[m], zorder=3 if m.startswith("R1") else 2)
        ax.axhline(0.05, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([int(L) for L in Ls], [str(L) for L in Ls])
        ax.set_xlabel("number of looks")
    axes[0].set_ylabel("FWER ↓")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_rq2(res, out):
    benches = list(res)
    methods = ("RC-mix", "plug-in", "Wilson-peek", "CP-final")
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        ys = [res[b]["within_session_A"][m]["simultaneous_miss"] for b in benches]
        ax.bar([i + j * w for i in range(len(benches))], ys, w, label=m + (" (ours)" if m in ("RC-mix", "plug-in") else ""))
    ax.axhline(0.05, color="black", lw=0.8, ls="--")
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(benches))], benches, fontsize=8)
    ax.set_ylabel("simultaneous miss ↓")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_c2(res, out):
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.bar(["R1", "R1′ (ours)"], [res["certified_r1"], res["certified_r1prime"]], color=[COL["R1"], COL["R1prime"]])
    ax.set_ylabel(f"certified contrasts of {res['K']} ↑")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_rq1(res, out):
    """Per-model attack success rate per channel in session B, with the pooled rate as a marker (c2.json rows)."""
    p, n = {}, {}
    for r in res["rows"]:
        p.setdefault(r["model"], {})[r["a"]] = r["p_a"]
        p.setdefault(r["model"], {})[r["b"]] = r["p_b"]
    short = {"gpt-4o-mini-2024-07-18": "4o-mini", "gpt-4.1-mini-2025-04-14": "4.1-mini", "gpt-5.4-nano-2026-03-17": "5.4-nano",
             "qwen25-7b-local": "Qwen2.5-7B", "qwen3-8b-local": "Qwen3-8B", "qwen3-32b-local": "Qwen3-32B",
             "llama31-8b-local": "Llama-3.1-8B", "deepseek-v4.1-flash": "DeepSeek", "glm-5.3": "GLM-5.3"}
    models = [m for m in short if m in p]
    ch = ("tool_return", "config", "tool_desc")
    lab = {"tool_return": "tool return", "config": "configuration", "tool_desc": "tool description"}
    col = {"tool_return": "#1f5fa8", "config": "#c2641a", "tool_desc": "#0b8a6f"}
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    w = 0.26
    for j, c in enumerate(ch):
        ax.bar([i + (j - 1) * w for i in range(len(models))], [p[m][c] for m in models], w, color=col[c], label=lab[c])
    ax.scatter(range(len(models)), [sum(p[m].values()) / 3 for m in models], marker="_", s=260, color="black", zorder=3,
               label="pooled rate")
    ax.set_xticks(range(len(models)), [short[m] for m in models], fontsize=7, rotation=20)
    ax.set_ylabel("attack success rate ↓")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


C3NAME = {"qwen3-8b-local": "Qwen3-8B", "glm-5.3": "GLM-5.3", "gpt-5.4-nano-2026-03-17": "GPT-5.4-nano",
          "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
C3COL = {"qwen3-8b-local": "#c2641a", "glm-5.3": "#1f5fa8", "gpt-5.4-nano-2026-03-17": "#0b8a6f", "deepseek-v4.1-flash": "#7a7a7a"}


def fig_c3(res, out):
    """RQ1 audit: sign interval for 2q-1 (tool return minus configuration) after every round, per model."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m, v in res["models"].items():
        t = v["trace"]
        n = [r[0] for r in t]
        ax.fill_between(n, [r[1] for r in t], [r[2] for r in t], color=C3COL[m], alpha=0.12, lw=0, step="post")
        ax.step(n, [r[1] for r in t], where="post", color=C3COL[m], lw=1.1, label=C3NAME[m])
        ax.step(n, [r[2] for r in t], where="post", color=C3COL[m], lw=1.1)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Round (log scale)")
    ax.set_ylabel("Sign interval for $2q-1$")
    ax.set_ylim(-1, 1)
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=C3COL[m], lw=2) for m in res["models"]]
    ax.legend(hs, [C3NAME[m] for m in res["models"]], fontsize=6.5, frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


ARMCOL = {"cold": "#7a7a7a", "prev": "#1f5fa8", "oracle": "#0b8a6f", "adversary": "#c2641a"}
ARMLAB = {"cold": "Cold", "prev": "Previous table", "oracle": "Correct side", "adversary": "Wrong side"}
SHORT = {"gpt-4o-mini-2024-07-18": "4o-mini", "gpt-4.1-mini-2025-04-14": "4.1-mini", "gpt-5.4-nano-2026-03-17": "5.4-nano",
         "qwen25-7b-local": "Qwen2.5-7B", "qwen3-8b-local": "Qwen3-8B", "qwen3-32b-local": "Qwen3-32B"}


def fig_e4(parts, out):
    """RQ4: distribution of channel runs per replay for each hedge, one group per release transition."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    arms = ("cold", "prev", "oracle", "adversary")
    for i, t in enumerate(parts):
        for j, a in enumerate(arms):
            b = ax.boxplot(t["arms"][a]["cost"], positions=[i * 5 + j], widths=0.7, patch_artist=True, showfliers=False)
            for k in ("boxes",):
                for x in b[k]:
                    x.set_facecolor(ARMCOL[a])
                    x.set_alpha(0.7)
            for x in b["medians"]:
                x.set_color("k")
    ax.set_xticks([i * 5 + 1.5 for i in range(len(parts))])
    ax.set_xticklabels([f"{SHORT[t['prev']]}\n→ {SHORT[t['new']]}" for t in parts], fontsize=6.5)
    ax.set_ylabel("Channel runs to close table")
    ax.set_ylim(0, 3000)
    for a in arms:
        ax.bar(0, 0, color=ARMCOL[a], label=ARMLAB[a])
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center", handlelength=1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_e5(out):
    """RQ4: distinct violation species against runs per cell, with the stop of the discovery rule."""
    import m3b_analyze as M
    import e5_run
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cols = ["#1f5fa8", "#0b8a6f", "#c2641a", "#9b3d8f", "#7a7a7a", "#e0a24a"]
    for i, c in enumerate(e5_run.CELLS):
        seq = M._seq(i)
        a = M.analyse_cell(seq)
        seen, ys = set(), []
        for sp in seq:
            if sp is not None:
                seen.add(sp)
            ys.append(len(seen))
        lab = f"{SHORT.get(c[0], c[0])}" if c[0] in SHORT else {"glm-5.3": "GLM-5.3", "deepseek-v4.1-flash": "DeepSeek"}.get(c[0], c[0])
        ax.plot(range(1, len(ys) + 1), ys, color=cols[i], lw=1.4, label=lab)
        if a["bound"] is not None and a["bound"] != float("inf") and a["stop_n"]:
            ax.plot(a["stop_n"], ys[a["stop_n"] - 1], "o", color=cols[i], ms=4)
    ax.set_xlabel("Run")
    ax.set_ylabel("Distinct violation species")
    ax.set_ylim(0, 52)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left", handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(res_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for name, fn in (("e3p", fig_e3p), ("c2", fig_c2), ("c2", fig_rq1), ("c3", fig_c3)):  # RQ2 is a generated table (coverage + width together)
        p = os.path.join(res_dir, f"{name}.json")
        if os.path.exists(p):
            tag = "rq1" if fn is fig_rq1 else "c3" if fn is fig_c3 else name
            fn(json.load(open(p)), os.path.join(out_dir, f"fig_{tag}.pdf"))
            done.append(tag)
    e4p = [os.path.join(res_dir, f"e4_replay_{i}.json") for i in range(4)]
    if all(os.path.exists(p) for p in e4p):
        fig_e4([t for p in e4p for t in json.load(open(p))], os.path.join(out_dir, "fig_e4.pdf"))
        done.append("e4")
    if os.path.exists(os.path.join(res_dir, "e5.json")):
        fig_e5(os.path.join(out_dir, "fig_e5.pdf"))
        done.append("e5")
    print("figures:", done)
    return done


if __name__ == "__main__":
    a = sys.argv
    res_dir = a[a.index("--res") + 1] if "--res" in a else os.path.join(HERE, "..", "results")
    out_dir = a[a.index("--out") + 1] if "--out" in a else os.path.join(HERE, "..", "paper_artifacts", "figs")
    main(res_dir, out_dir)
