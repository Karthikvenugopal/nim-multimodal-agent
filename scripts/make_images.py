"""Regenerate the sample corpus images in corpus/images/.

The numeric facts rendered here intentionally do NOT appear in any text
document, so benchmark questions about them can only be answered by the
vision path of the agent.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "corpus" / "images"


def fig_latency_benchmark() -> None:
    """Bar chart: p95 inference latency per VoltEdge variant."""
    fig, ax = plt.subplots(figsize=(6, 4))
    products = ["VoltEdge Nano", "VoltEdge Pro", "VoltEdge Max"]
    latency = [42, 18, 9]
    bars = ax.bar(products, latency, color=["#76b900", "#4a7a00", "#2d4a00"])
    ax.bar_label(bars, fmt="%d ms")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("VoltEdge p95 Inference Latency — ResNet-50 INT8, batch 1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_benchmark.png", dpi=120)
    plt.close(fig)


def fig_revenue_mix() -> None:
    """Pie chart: Q3 2025 revenue mix by product line."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = ["VoltEdge Pro", "VoltEdge Max", "VoltEdge Nano", "Services"]
    shares = [52, 31, 12, 5]
    ax.pie(shares, labels=labels, autopct="%d%%", startangle=90,
           colors=["#76b900", "#9ad13a", "#c2e87e", "#e6f4c7"])
    ax.set_title("Kestrel Systems Q3 2025 Revenue Mix")
    fig.tight_layout()
    fig.savefig(OUT / "fig_revenue_mix.png", dpi=120)
    plt.close(fig)


def fig_gpu_utilization() -> None:
    """Line chart: fleet GPU utilization over a 24-hour window."""
    hours = list(range(24))
    util = [31, 28, 25, 24, 23, 26, 35, 48, 62, 71, 78, 84,
            88, 90, 91, 89, 85, 80, 72, 63, 55, 46, 39, 34]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hours, util, marker="o", markersize=3, color="#76b900")
    ax.annotate("peak 91% @ 14:00", xy=(14, 91), xytext=(15.5, 95),
                arrowprops={"arrowstyle": "->"})
    ax.annotate("trough 23% @ 04:00", xy=(4, 23), xytext=(5.5, 15),
                arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("fleet GPU utilization (%)")
    ax.set_ylim(0, 105)
    ax.set_title("VoltEdge Fleet GPU Utilization — 24h sample")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gpu_utilization.png", dpi=120)
    plt.close(fig)


def fig_architecture() -> None:
    """Block diagram of the VoltEdge streaming inference pipeline."""
    stages = ["Camera\nArray", "Ingest\nGateway", "Triton\nInference\nServer",
              "Post-\nprocess", "MQTT\nBroker", "Ops\nDashboard"]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    for i, stage in enumerate(stages):
        x = i * 1.55
        ax.add_patch(plt.Rectangle((x, 0.25), 1.2, 0.9, fill=True,
                                   facecolor="#e6f4c7", edgecolor="#4a7a00"))
        ax.text(x + 0.6, 0.7, stage, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.5, 0.7), xytext=(x + 1.25, 0.7),
                        arrowprops={"arrowstyle": "->", "color": "#333"})
    ax.set_xlim(-0.2, 9.0)
    ax.set_ylim(0, 1.4)
    ax.set_title("VoltEdge Streaming Inference Pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture.png", dpi=130)
    plt.close(fig)


def fig_error_rates() -> None:
    """Grouped bar chart: stream error rate by firmware version."""
    products = ["Nano", "Pro", "Max"]
    fw_23 = [2.4, 1.8, 1.1]
    fw_24 = [0.9, 0.6, 0.3]
    x = range(len(products))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar([i - width / 2 for i in x], fw_23, width,
                label="firmware 2.3", color="#bbbbbb")
    b2 = ax.bar([i + width / 2 for i in x], fw_24, width,
                label="firmware 2.4", color="#76b900")
    ax.bar_label(b1, fmt="%.1f%%")
    ax.bar_label(b2, fmt="%.1f%%")
    ax.set_xticks(list(x), [f"VoltEdge {p}" for p in products])
    ax.set_ylabel("stream error rate (%)")
    ax.set_title("Stream Error Rate by Firmware Version")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_rates.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_latency_benchmark, fig_revenue_mix, fig_gpu_utilization,
               fig_architecture, fig_error_rates):
        fn()
        print(f"wrote {fn.__name__}")
