from __future__ import annotations
import collections
import threading
from taut.core.models import PipelineMetrics
import abc
import logging

class MetricsExporter(abc.ABC):
    """Interface for exporting metrics to external systems."""
    @abc.abstractmethod
    def export(self, metrics_collector: MetricsCollector) -> None:
        pass

class LoggingExporter(MetricsExporter):
    """Simple exporter that logs the metrics summary."""
    def __init__(self, level: int = logging.INFO):
        self.level = level
        
    def export(self, metrics_collector: MetricsCollector) -> None:
        logger = logging.getLogger("taut.metrics")
        logger.log(self.level, "Taut Metrics Export:\n%s", metrics_collector.summary())


class MetricsCollector:
    """Thread-safe aggregator for pipeline metrics."""
    
    def export(self, exporter: MetricsExporter) -> None:
        """Export metrics using the provided exporter."""
        exporter.export(self)
        
    def __init__(self, pricing: dict[str, float] | None = None):
        self._requests = collections.deque(maxlen=10_000)
        self._lock = threading.Lock()
        self.pricing = pricing or {}
    
    def record(self, metrics: PipelineMetrics) -> None:
        with self._lock:
            self._requests.append(metrics)
            
    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
    
    @property
    def total_requests(self) -> int:
        with self._lock:
            return len(self._requests)
    
    @property
    def cache_hit_rate(self) -> float:
        with self._lock:
            if not self._requests:
                return 0.0
            hits = sum(1 for r in self._requests if r.cache_hit)
            return hits / len(self._requests)
    
    @property
    def total_tokens_saved(self) -> int:
        with self._lock:
            return sum(r.total_tokens_saved for r in self._requests)
    
    @property
    def total_cost_saved(self) -> float:
        with self._lock:
            return sum(r.estimated_cost_saved for r in self._requests)
    
    def model_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        with self._lock:
            for r in self._requests:
                if r.model_used:
                    dist[r.model_used] = dist.get(r.model_used, 0) + 1
        return dist
    
    def summary(self) -> str:
        with self._lock:
            if not self._requests:
                return "No requests recorded yet."
            total_reqs = len(self._requests)
            hits = sum(1 for r in self._requests if r.cache_hit)
            hit_rate = hits / total_reqs if total_reqs else 0.0
            tokens_saved = sum(r.total_tokens_saved for r in self._requests)
            cost_saved = sum(r.estimated_cost_saved for r in self._requests)
            
            dist: dict[str, int] = {}
            for r in self._requests:
                if r.model_used:
                    dist[r.model_used] = dist.get(r.model_used, 0) + 1
                    
        lines = [
            "┌─────────────────────────┬──────────────┐",
            f"│ Total Requests          │ {total_reqs:>12,} │",
            f"│ Cache Hit Rate          │ {hit_rate:>11.1%} │",
            f"│ Total Tokens Saved      │ {tokens_saved:>12,} │",
            f"│ Estimated Cost Saved    │ ${cost_saved:>11.2f} │",
        ]
        if dist:
            lines.append("│ Model Distribution      │              │")
            for model, count in sorted(dist.items(), key=lambda x: -x[1]):
                pct = count / total_reqs
                label = f"  {model[:22]}"
                lines.append(f"│ {label:<23} │ {pct:>11.1%} │")
        lines.append("└─────────────────────────┴──────────────┘")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cache_hit_rate": self.cache_hit_rate,
            "total_tokens_saved": self.total_tokens_saved,
            "total_cost_saved": self.total_cost_saved,
            "model_distribution": self.model_distribution(),
        }
