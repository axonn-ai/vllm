"""Parse and diff Prometheus text exposition format for vLLM histogram metrics."""

import re
from collections import defaultdict


def parse_prometheus_text(text: str) -> dict:
    """Parse Prometheus text exposition format, extracting histogram metrics.

    Returns a dict keyed by metric name, each containing:
      - "buckets": list of {"le": str, "cumulative_count": float}
      - "count": float
      - "sum": float
    """
    histograms: dict[str, dict] = defaultdict(
        lambda: {"buckets": [], "count": 0.0, "sum": 0.0}
    )

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match: metric_name{labels} value
        # or:   metric_name value
        match = re.match(r'^(\S+?)(\{[^}]*\})?\s+(\S+)$', line)
        if not match:
            continue

        full_name = match.group(1)
        labels_str = match.group(2) or ""
        value_str = match.group(3)

        try:
            value = float(value_str)
        except ValueError:
            continue

        if full_name.endswith("_bucket"):
            base = full_name[:-len("_bucket")]
            le_match = re.search(r'le="([^"]+)"', labels_str)
            if le_match:
                histograms[base]["buckets"].append({
                    "le": le_match.group(1),
                    "cumulative_count": value,
                })
        elif full_name.endswith("_count"):
            base = full_name[:-len("_count")]
            if base in histograms or _looks_like_histogram(base, text):
                histograms[base]["count"] = value
        elif full_name.endswith("_sum"):
            base = full_name[:-len("_sum")]
            if base in histograms or _looks_like_histogram(base, text):
                histograms[base]["sum"] = value

    return dict(histograms)


def _looks_like_histogram(base: str, text: str) -> bool:
    """Check if a metric base name has corresponding _bucket lines."""
    return f"{base}_bucket{{" in text


def diff_prometheus_metrics(before: str, after: str) -> dict:
    """Diff two Prometheus text snapshots and return per-metric histogram diffs.

    Returns a dict keyed by metric name with:
      - "count": differential count
      - "sum": differential sum
      - "mean": sum / count (or 0)
      - "buckets": list of {"le": str, "cumulative_count": float} (differential)
    """
    before_metrics = parse_prometheus_text(before)
    after_metrics = parse_prometheus_text(after)

    result = {}
    for name, after_data in after_metrics.items():
        before_data = before_metrics.get(name, {"buckets": [], "count": 0.0, "sum": 0.0})

        diff_count = after_data["count"] - before_data["count"]
        diff_sum = after_data["sum"] - before_data["sum"]

        # Build bucket lookup from before
        before_buckets = {b["le"]: b["cumulative_count"] for b in before_data["buckets"]}

        diff_buckets = []
        for bucket in after_data["buckets"]:
            le = bucket["le"]
            before_val = before_buckets.get(le, 0.0)
            diff_buckets.append({
                "le": le,
                "cumulative_count": bucket["cumulative_count"] - before_val,
            })

        result[name] = {
            "count": diff_count,
            "sum": diff_sum,
            "mean": diff_sum / diff_count if diff_count > 0 else 0.0,
            "buckets": diff_buckets,
        }

    return result
