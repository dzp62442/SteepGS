"""Pure formatting helpers for OmniScene metric reports."""

from typing import Any, Mapping


METRIC_NAMES = ("psnr", "ssim", "lpips")
METRIC_GROUPS = (("all_18", "ALL_18"), ("novel_12", "NOVEL_12"))


def format_metrics_text(metrics: Mapping[str, Any]) -> str:
    """Format both required camera groups without changing metric values."""
    lines = []
    for group, label in METRIC_GROUPS:
        values = metrics[group]
        lines.append(f"[{label}]")
        for name in METRIC_NAMES:
            lines.append(f"{name.upper()}: {float(values[name]):.10f}")
    return "\n".join(lines) + "\n"
