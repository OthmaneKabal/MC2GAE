"""Low-overhead resource profiling for isolated GSSL worker processes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path


GIB = 1024.0 ** 3


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _maximum(values):
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def _combined_maximum(*values):
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def _std(values):
    values = [float(value) for value in values if value is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _number(value):
    value = value.strip().replace("MiB", "").replace("W", "").replace("%", "").strip()
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class EfficiencyProfiler:
    """Sample one worker without changing its model or optimization steps."""

    def __init__(self, run_dir: Path, spec: dict, config: dict):
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.config = config
        self.interval = max(float(spec.get("profile_sample_seconds", 1.0)), 0.25)
        self.samples = []
        self.started_at = None
        self._stop_event = threading.Event()
        self._thread = None
        self._torch = None
        self._device = None

    def start(self):
        import torch

        self._torch = torch
        self.started_at = time.perf_counter()
        self.config["_efficiency_profile_t0"] = self.started_at
        self.config["_efficiency_phase"] = "startup"
        self.config["_efficiency_epoch_timings"] = []
        self.config["_efficiency_phase_timings"] = []
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(self._device)
        self._sample_once()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gssl-efficiency-profiler",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self):
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def _cpu_rss_gb(self):
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / GIB
        except (ImportError, OSError):
            try:
                import resource

                value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # Linux reports KiB while macOS reports bytes.
                return value / (GIB if platform.system() == "Darwin" else 1024.0 ** 2)
            except (ImportError, OSError):
                return None

    def _gpu_snapshot(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            line = next(line for line in result.stdout.splitlines() if line.strip())
            values = [_number(value) for value in line.split(",")]
            if len(values) != 4:
                return {}
            return {
                "gpu_utilization_percent": values[0],
                "nvml_global_memory_gb": values[1] / 1024.0 if values[1] is not None else None,
                "nvml_total_memory_gb": values[2] / 1024.0 if values[2] is not None else None,
                "gpu_power_w": values[3],
            }
        except (FileNotFoundError, subprocess.SubprocessError, StopIteration, ValueError):
            return {}

    def _process_gpu_memory_gb(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            process_mib = 0.0
            found = False
            for line in result.stdout.splitlines():
                values = [value.strip() for value in line.split(",", 1)]
                if len(values) != 2 or values[0] != str(os.getpid()):
                    continue
                memory_mib = _number(values[1])
                if memory_mib is not None:
                    process_mib += memory_mib
                    found = True
            return process_mib / 1024.0 if found else None
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            return None

    def _sample_once(self):
        if self.started_at is None:
            return
        sample = {
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "phase": self.config.get("_efficiency_phase", "unknown"),
            "cpu_rss_gb": self._cpu_rss_gb(),
        }
        if self._device is not None:
            sample.update({
                "torch_allocated_gb": self._torch.cuda.memory_allocated(self._device) / GIB,
                "torch_reserved_gb": self._torch.cuda.memory_reserved(self._device) / GIB,
            })
        sample.update(self._gpu_snapshot())
        sample["nvml_process_memory_gb"] = self._process_gpu_memory_gb()
        self.samples.append(sample)

    def _environment(self):
        torch = self._torch
        environment = {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "torch_geometric_version": _package_version("torch-geometric"),
            "pylibcugraphops_version": (
                _package_version("pylibcugraphops-cu12")
                or _package_version("pylibcugraphops")
            ),
        }
        if self._device is not None:
            environment["gpu_name"] = torch.cuda.get_device_name(self._device)
            properties = torch.cuda.get_device_properties(self._device)
            environment["gpu_total_memory_gb"] = properties.total_memory / GIB
        return environment

    def stop(self, status: str, error: str | None = None):
        if self.started_at is None:
            return None
        self.config["_efficiency_phase"] = "finished"
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5.0)
        self._sample_once()

        torch_peak_allocated = None
        torch_peak_reserved = None
        if self._device is not None:
            torch_peak_allocated = self._torch.cuda.max_memory_allocated(self._device) / GIB
            torch_peak_reserved = self._torch.cuda.max_memory_reserved(self._device) / GIB

        epochs = list(self.config.get("_efficiency_epoch_timings", []))
        epoch_total = [entry.get("epoch_total_seconds") for entry in epochs]
        epoch_train = [entry.get("train_seconds") for entry in epochs]
        epoch_evaluation = [entry.get("evaluation_seconds") for entry in epochs]
        steady_epochs = epochs[1:] if len(epochs) > 1 else epochs
        steady_total = [entry.get("epoch_total_seconds") for entry in steady_epochs]
        steady_train = [entry.get("train_seconds") for entry in steady_epochs]
        steady_steps = sum(int(entry.get("steps", 0)) for entry in steady_epochs)
        steady_train_seconds = sum(
            float(entry.get("train_seconds", 0.0)) for entry in steady_epochs
        )

        sampled_torch_allocated = [sample.get("torch_allocated_gb") for sample in self.samples]
        sampled_torch_reserved = [sample.get("torch_reserved_gb") for sample in self.samples]
        train_samples = [sample for sample in self.samples if sample.get("phase") == "train"]
        profile = {
            "status": status,
            "error": error,
            "run_id": self.spec.get("run_id"),
            "graph": self.spec.get("graph"),
            "encoder": self.spec.get("encoder"),
            "decoder": self.spec.get("decoder"),
            "task": self.spec.get("task"),
            "hidden_size": int(self.spec.get("channels", [0])[0]),
            "num_bases": self.spec.get("num_bases"),
            "seed": self.spec.get("seed"),
            "batch_size": self.spec.get("batch_size"),
            "linear_probe_batch_size": self.spec.get("linear_probe_batch_size"),
            "num_neighbors": self.spec.get("num_neighbors"),
            "num_parameters": self.config.get("_efficiency_num_parameters"),
            "num_nodes": self.config.get("_efficiency_num_nodes"),
            "num_edges": self.config.get("_efficiency_num_edges"),
            "num_relations": self.config.get("_efficiency_num_relations"),
            "sample_interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "total_wall_time_seconds": time.perf_counter() - self.started_at,
            "startup_time_seconds": epochs[0].get("started_elapsed_seconds") if epochs else None,
            "epoch_count": len(epochs),
            "epoch_timings": epochs,
            "epoch_times_seconds": epoch_total,
            "train_epoch_times_seconds": epoch_train,
            "evaluation_epoch_times_seconds": epoch_evaluation,
            "steady_epoch_mean_seconds": _mean(steady_total),
            "steady_epoch_std_seconds": _std(steady_total),
            "steady_train_mean_seconds": _mean(steady_train),
            "steady_train_std_seconds": _std(steady_train),
            "steady_optimizer_steps_per_second": (
                steady_steps / steady_train_seconds if steady_train_seconds > 0 else None
            ),
            "estimated_target_nodes_per_second": (
                steady_steps * int(self.spec.get("batch_size", 0)) / steady_train_seconds
                if steady_train_seconds > 0 else None
            ),
            "mean_torch_allocated_gb": _mean(sampled_torch_allocated),
            "peak_torch_allocated_gb": _combined_maximum(
                torch_peak_allocated, _maximum(sampled_torch_allocated)
            ),
            "mean_torch_reserved_gb": _mean(sampled_torch_reserved),
            "peak_torch_reserved_gb": _combined_maximum(
                torch_peak_reserved, _maximum(sampled_torch_reserved)
            ),
            "mean_train_torch_allocated_gb": _mean(
                sample.get("torch_allocated_gb") for sample in train_samples
            ),
            "peak_train_torch_allocated_gb": _maximum(
                sample.get("torch_allocated_gb") for sample in train_samples
            ),
            "mean_train_torch_reserved_gb": _mean(
                sample.get("torch_reserved_gb") for sample in train_samples
            ),
            "peak_train_torch_reserved_gb": _maximum(
                sample.get("torch_reserved_gb") for sample in train_samples
            ),
            "mean_nvml_process_memory_gb": _mean(
                sample.get("nvml_process_memory_gb") for sample in self.samples
            ),
            "peak_nvml_process_memory_gb": _maximum(
                sample.get("nvml_process_memory_gb") for sample in self.samples
            ),
            "mean_train_nvml_process_memory_gb": _mean(
                sample.get("nvml_process_memory_gb") for sample in train_samples
            ),
            "peak_train_nvml_process_memory_gb": _maximum(
                sample.get("nvml_process_memory_gb") for sample in train_samples
            ),
            "mean_nvml_global_memory_gb": _mean(
                sample.get("nvml_global_memory_gb") for sample in self.samples
            ),
            "peak_nvml_global_memory_gb": _maximum(
                sample.get("nvml_global_memory_gb") for sample in self.samples
            ),
            "nvml_total_memory_gb": _maximum(
                sample.get("nvml_total_memory_gb") for sample in self.samples
            ),
            "mean_gpu_utilization_percent": _mean(
                sample.get("gpu_utilization_percent") for sample in self.samples
            ),
            "peak_gpu_utilization_percent": _maximum(
                sample.get("gpu_utilization_percent") for sample in self.samples
            ),
            "mean_train_gpu_utilization_percent": _mean(
                sample.get("gpu_utilization_percent") for sample in train_samples
            ),
            "peak_train_gpu_utilization_percent": _maximum(
                sample.get("gpu_utilization_percent") for sample in train_samples
            ),
            "mean_gpu_power_w": _mean(sample.get("gpu_power_w") for sample in self.samples),
            "peak_gpu_power_w": _maximum(sample.get("gpu_power_w") for sample in self.samples),
            "mean_cpu_rss_gb": _mean(sample.get("cpu_rss_gb") for sample in self.samples),
            "peak_cpu_rss_gb": _maximum(sample.get("cpu_rss_gb") for sample in self.samples),
            "phase_timings": list(self.config.get("_efficiency_phase_timings", [])),
            "result_metrics": dict(self.config.get("_efficiency_result_metrics", {})),
            **self._environment(),
        }
        if profile["mean_gpu_power_w"] is not None:
            profile["estimated_energy_wh"] = (
                profile["mean_gpu_power_w"] * profile["total_wall_time_seconds"] / 3600.0
            )
        else:
            profile["estimated_energy_wh"] = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        samples_path = self.run_dir / "efficiency_samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
        (self.run_dir / "efficiency_profile.json").write_text(
            json.dumps(profile, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return profile
