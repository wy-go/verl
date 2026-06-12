#!/usr/bin/env python3
"""Figures: the SAME verl GRPO job under two launch methods — direct vs `ray job submit`."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(direct="#4C72B0", job="#DD8452", core="#55A868", grey="#7f7f7f",
         light="#EAEAF2", warn="#C44E52", ok="#55A868", bg="#FFFFFF", ink="#222")

def box(ax, x, y, w, h, text, fc, tc="white", fs=10.5, bold=True, alpha=1.0, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.4, edgecolor=ec or fc, facecolor=fc, alpha=alpha, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)

def arrow(ax, p1, p2, color="#444", lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16, lw=lw,
                 color=color, zorder=1, connectionstyle=f"arc3,rad={rad}"))

def lab(ax, x, y, t, fs=11, color="#222", bold=True, ha="left"):
    ax.text(x, y, t, fontsize=fs, color=color, fontweight="bold" if bold else "normal", ha=ha, va="center")

# ================= Figure 1: launch-path comparison =================
fig, ax = plt.subplots(figsize=(13.4, 8.2)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 8.2); ax.axis("off")
ax.text(6.7, 7.9, "Same verl GRPO job — two ways to launch it onto the SAME Ray cluster",
        ha="center", fontsize=15.5, fontweight="bold")

# divider
ax.plot([6.7, 6.7], [0.7, 7.4], ls=(0,(4,4)), color=C["grey"], lw=1.2)
ax.text(3.35, 7.35, "A.  DIRECT  (what worked before)", ha="center", fontsize=13, color=C["direct"], fontweight="bold")
ax.text(10.05, 7.35, "B.  RAY JOB SUBMIT  (records in Ray UI)", ha="center", fontsize=13, color=C["job"], fontweight="bold")

# ---- column A : direct ----
box(ax, 1.0, 6.25, 4.7, 0.72, "devbox  — ssh to the WORKER pod", C["direct"], fs=10.5)
box(ax, 1.0, 5.30, 4.7, 0.72, "bash run_qwen3_8b_grpo.sh", C["ink"], fs=10.5)
box(ax, 1.0, 4.05, 4.7, 1.0, "verl.trainer.main_ppo\nray.init(address=\"auto\")\nCREATES the session  +  runtime_env\n(LD_LIBRARY_PATH -> sglang libcudart)", C["core"], fs=9.3)
box(ax, 1.0, 2.95, 4.7, 0.8, "driver runs in your shell (on worker)\nactors placed on the 8 GPUs", C["direct"], fs=9.6)
box(ax, 1.0, 2.05, 4.7, 0.66, "X  no Ray Jobs UI record", C["warn"], fs=10.5)
for a,b in [((3.35,6.25),(3.35,6.02)),((3.35,5.30),(3.35,5.05)),((3.35,4.05),(3.35,3.75)),((3.35,2.95),(3.35,2.71))]:
    arrow(ax, a, b, color=C["grey"], lw=1.7)

# ---- column B : ray job submit ----
box(ax, 8.0, 6.25, 4.7, 0.72, "devbox — ssh to the HEAD pod", C["job"], fs=10.5)
box(ax, 8.0, 5.12, 4.7, 0.9, "ray job submit  (token auto-auth)\n--entrypoint-resources '{\"worker\":1}'\n--  bash run_qwen3_8b_grpo.sh", C["ink"], fs=9.3)
box(ax, 8.0, 4.05, 4.7, 0.82, "JobSupervisor actor pinned to WORKER\n(entrypoint-resources -> has env + GPUs)", C["job"], fs=9.4)
box(ax, 8.0, 2.95, 4.7, 0.82, "verl.trainer.main_ppo\nray.init() ATTACHES to the job session\n(runtime_env comes from the JOB)", C["core"], fs=9.3)
box(ax, 8.0, 2.05, 4.7, 0.66, "OK  Ray Jobs UI record: qwen3_8b_grpo_*", C["ok"], fs=10.3)
for a,b in [((10.35,6.25),(10.35,6.02)),((10.35,5.12),(10.35,4.87)),((10.35,4.05),(10.35,3.77)),((10.35,2.95),(10.35,2.71))]:
    arrow(ax, a, b, color=C["grey"], lw=1.7)

# ---- shared bottom : same cluster / same env / same result ----
box(ax, 1.0, 0.75, 11.7, 1.05,
    "SAME debug Ray cluster (1 head CPU  +  1 worker 8xH100)  ·  SAME image + preprocess.sh env  ·  "
    "SAME result:  gsm8k val acc 0.82",
    C["light"], tc="#222", fs=11, bold=True, ec=C["grey"])
arrow(ax, (3.35, 2.05), (4.6, 1.80), color=C["grey"], lw=1.6, rad=-0.15)
arrow(ax, (10.35, 2.05), (9.1, 1.80), color=C["grey"], lw=1.6, rad=0.15)

# gotcha callout
ax.text(6.7, 0.45,
        "key gotcha: verl's ray.init already sets runtime_env (LD_LIBRARY_PATH). Under a job, don't ALSO set the same keys "
        "via --runtime-env-json (merge conflict). Just submit clean; retry the known-flaky sglang launch.",
        ha="center", fontsize=8.8, color="#555", style="italic")

plt.tight_layout(); plt.savefig("figures/verl_launch_1_compare.png", dpi=160, bbox_inches="tight"); plt.close()

# ================= Figure 2: identical outcome =================
fig, ax = plt.subplots(figsize=(9.2, 5.6))
steps   = [0, 2, 4, 6, 7]
job_acc = [0.317, 0.524, 0.748, 0.797, 0.821]   # this run, via ray job submit
dir_acc = [0.320, 0.535, 0.760, 0.807, 0.817]   # RESULTS.md direct-launch baseline
ax.plot(steps, dir_acc, "o--", color=C["direct"], lw=2.4, ms=8, label="direct launch (RESULTS.md baseline)")
ax.plot(steps, job_acc, "s-",  color=C["job"],    lw=2.4, ms=8, label="ray job submit (this run, 101145)")
for x, y in zip(steps, job_acc):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8.5, color=C["job"])
ax.set_title("Same job, same outcome — Qwen3-8B GRPO on gsm8k (8xH100, sglang)", fontsize=13, fontweight="bold")
ax.set_xlabel("training step", fontsize=11); ax.set_ylabel("val acc@1 (gsm8k)", fontsize=11)
ax.set_xticks(steps); ax.set_ylim(0.25, 0.88); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
ax.text(0.02, 0.96, "launch method does NOT change the result —\nonly whether the run is recorded in the Ray Jobs UI",
        transform=ax.transAxes, fontsize=9.2, va="top", color="#555",
        bbox=dict(boxstyle="round,pad=0.4", fc=C["light"], ec=C["grey"], lw=1))
plt.tight_layout(); plt.savefig("figures/verl_launch_2_outcome.png", dpi=160, bbox_inches="tight"); plt.close()

print("wrote figures/verl_launch_1_compare.png and figures/verl_launch_2_outcome.png")
