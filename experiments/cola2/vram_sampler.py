"""Sample device-wide VRAM from AMD SMI while a Cola2 workload runs."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import subprocess
import threading
import time


COMMAND = ("amd-smi", "metric", "--mem-usage", "--json")


class VramSampler:
    """Each instance captures one run. Mark phases from the workload thread."""

    def __init__(self, gpu_index=0, interval_seconds=0.25, timeout_seconds=3.0):
        if not all(math.isfinite(value) and value > 0 for value in (interval_seconds, timeout_seconds)):
            raise ValueError("sampling interval and timeout must be finite and positive")
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._phase = "startup"
        self._phase_revision = 0
        self._phase_changes = []
        self._samples = []

    def _read_vram(self, sample):
        completed = subprocess.run(
            COMMAND, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=self.timeout_seconds, check=False,
        )
        sample["exit_code"] = completed.returncode
        if completed.stderr:
            sample["stderr"] = completed.stderr
        if completed.returncode:
            sample["stdout"] = completed.stdout
            raise ValueError(f"AMD SMI exited with status {completed.returncode}")
        try:
            sample["raw"] = json.loads(completed.stdout)
        except ValueError:
            sample["stdout"] = completed.stdout
            raise

        devices = [row for row in sample["raw"]["gpu_data"] if row["gpu"] == self.gpu_index]
        if len(devices) != 1:
            raise ValueError(f"expected one AMD SMI entry for GPU index {self.gpu_index}")
        usage = devices[0]["mem_usage"]
        used, total = usage["used_vram"], usage["total_vram"]
        used_value, total_value = used["value"], total["value"]
        unit = used["unit"]
        if (
            type(used_value) not in (int, float) or type(total_value) not in (int, float)
            or not math.isfinite(used_value) or not math.isfinite(total_value)
            or not 0 <= used_value <= total_value or total_value == 0
            or not isinstance(unit, str) or not unit or unit != total["unit"]
        ):
            raise ValueError("invalid or incompatible AMD SMI VRAM reading")
        sample.update(used_vram=used_value, total_vram=total_value, unit=unit)

    def _sample(self, kind):
        with self._lock:
            phase, revision = self._phase, self._phase_revision
            sample = {
                "kind": kind,
                "phase": phase if kind == "periodic" else kind,
                "query_start_utc": datetime.now(timezone.utc).isoformat(),
                "query_start_monotonic_ns": time.monotonic_ns(),
            }
        try:
            self._read_vram(sample)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, OverflowError) as error:
            sample["error"] = f"{type(error).__name__}: {error}"
        with self._lock:
            sample["query_end_monotonic_ns"] = time.monotonic_ns()
            sample["query_end_utc"] = datetime.now(timezone.utc).isoformat()
            sample["phase_changed_during_query"] = kind == "periodic" and revision != self._phase_revision
            self._samples.append(sample)

    def _poll(self):
        deadline = time.monotonic() + self.interval_seconds
        while not self._stop_requested.wait(max(0, deadline - time.monotonic())):
            self._sample("periodic")
            # Maintain the target start-to-start interval when queries are fast.
            deadline = max(deadline + self.interval_seconds, time.monotonic())

    def start(self):
        """Capture the baseline synchronously, then begin background sampling."""
        if self._thread is not None:
            raise RuntimeError("create a new sampler for each run")
        self._sample("baseline")
        self._thread = threading.Thread(target=self._poll, name="amd-smi-vram-sampler", daemon=True)
        self._thread.start()

    def phase(self, name):
        """Label subsequent queries. Queries spanning phase changes remain in the overall peak."""
        if not isinstance(name, str) or not name:
            raise ValueError("phase name must be a nonempty string")
        with self._lock:
            if self._phase != name:
                self._phase = name
                self._phase_revision += 1
                self._phase_changes.append({
                    "phase": name,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ns": time.monotonic_ns(),
                })

    def stop(self):
        """Stop background queries and capture the current post-workload reading."""
        if self._thread is None or self._stop_requested.is_set():
            return
        self._stop_requested.set()
        self._thread.join()
        self._sample("final")

    def report(self):
        """Report observed maxima in AMD SMI's raw units and retain query evidence."""
        with self._lock:
            samples = deepcopy(self._samples)
            phase_changes = deepcopy(self._phase_changes)
        valid = [sample for sample in samples if "used_vram" in sample]
        compatible = len({(sample["total_vram"], sample["unit"]) for sample in valid}) == 1
        periodic = [sample for sample in valid if sample["kind"] == "periodic"]
        phase_samples = {}
        for sample in periodic:
            if not sample["phase_changed_during_query"]:
                phase_samples.setdefault(sample["phase"], []).append(sample)
        phases = {sample["phase"] for sample in samples if sample["kind"] == "periodic"}
        phases.update(change["phase"] for change in phase_changes)
        phase_peaks = {
            phase: self._peak(phase_samples.get(phase, [])) if compatible else None
            for phase in sorted(phases)
        }
        return {
            "command": list(COMMAND),
            "gpu_index": self.gpu_index,
            "scope": "device-wide; includes other GPU clients; brief peaks may be missed",
            "target_interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "unit": valid[0]["unit"] if compatible else None,
            "successful_sample_count": len(valid),
            "successful_periodic_sample_count": len(periodic),
            "error_count": len(samples) - len(valid),
            "consistent_total_and_unit": compatible,
            "baseline": next((sample for sample in samples if sample["kind"] == "baseline"), None),
            "final": next((sample for sample in samples if sample["kind"] == "final"), None),
            "overall_sampled_peak": self._peak(valid) if compatible else None,
            "workload_sampled_peak": self._peak(periodic) if compatible else None,
            "phase_sampled_peaks": phase_peaks,
            "phase_changes": phase_changes,
            "samples": samples,
        }

    @staticmethod
    def _peak(samples):
        if not samples:
            return None
        sample = max(samples, key=lambda item: item["used_vram"])
        return {key: sample[key] for key in (
            "used_vram", "total_vram", "unit", "kind", "phase",
            "phase_changed_during_query", "query_start_utc", "query_end_utc",
            "query_start_monotonic_ns", "query_end_monotonic_ns",
        )}
