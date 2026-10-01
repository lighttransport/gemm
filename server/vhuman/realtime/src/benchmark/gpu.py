"""Low-rate NVML telemetry, including native allocations outside Torch's allocator."""
import os
import time


class GpuSampler:
    def __init__(self, metrics, tts_pid=None, device=0):
        self.metrics, self.tts_pid, self.next = metrics, tts_pid, 0
        try:
            import pynvml
        except ImportError:
            self.nvml = None
            return
        try:
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.device = pynvml.nvmlDeviceGetHandleByIndex(device)
        except pynvml.NVMLError:
            self.nvml = None

    def poll(self):
        if not self.nvml or time.monotonic() < self.next: return
        self.next = time.monotonic() + 1
        nv = self.nvml
        try:
            utilization = nv.nvmlDeviceGetUtilizationRates(self.device).gpu
            processes = nv.nvmlDeviceGetComputeRunningProcesses(self.device)
        except nv.NVMLError:
            self.metrics.add("nvml_sample_errors", 1)
            return
        self.metrics.add("gpu_utilization_global_pct", utilization)
        for process in processes:
            if process.pid == os.getpid(): name = "avatar_process_vram_mib"
            elif process.pid == self.tts_pid: name = "tts_process_vram_mib"
            else: continue
            if process.usedGpuMemory is not None and process.usedGpuMemory < 2**63:
                self.metrics.add(name, process.usedGpuMemory / 2**20)

    def close(self):
        if self.nvml: self.nvml.nvmlShutdown()
