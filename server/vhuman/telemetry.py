"""Whole-device portrait telemetry: NVML on CUDA, PCI-matched sysfs on ROCm."""
from pathlib import Path
import threading
import time

from . import gpu
from .rig.benchmark import Telemetry


def pci_address(value):
    domain, bus, slot = value.lower().rsplit(':', 2)
    return f'{int(domain, 16):04x}:{int(bus, 16):02x}:{slot}'


class ROCmTelemetry(Telemetry):
    def __init__(self, device=0, *, status=None, drm_root=Path('/sys/class/drm')):
        status = status or gpu.gpu_status(device, 'rocm')
        if not status or not status.get('pci_bus_id'):
            raise RuntimeError('ROCm PCI identity unavailable; rebuild rdna4/rocew_probe')
        wanted = pci_address(status['pci_bus_id'])
        matches = []
        for card in Path(drm_root).glob('card[0-9]*'):
            if '-' in card.name or not (card/'device').exists():
                continue
            try:
                address = pci_address((card/'device').resolve().name)
            except ValueError:  # Platform DRM devices need not have a PCI address.
                continue
            if address == wanted:
                matches.append(card/'device')
        if len(matches) != 1:
            raise RuntimeError('cannot uniquely map HIP device to AMD VRAM telemetry')
        self.path = matches[0]
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        def read(name):
            return int((self.path/name).read_text())
        return dict(time=time.time(), memory_used_bytes=read('mem_info_vram_used'),
                    memory_total_bytes=read('mem_info_vram_total'), memory_reserved_bytes=0,
                    gpu_percent=read('gpu_busy_percent'))

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.sample())

    def report(self):
        result = super().report()
        result['scope'] = 'AMD sysfs whole-device VRAM, PCI-matched HIP ordinal; sampled at 100 ms; no separate reserved-memory estimate'
        result['provider'] = 'amdgpu-sysfs'
        return result


def telemetry(device=0, backend='cuda'):
    return ROCmTelemetry(device) if backend == 'rocm' else Telemetry(device)


def physical_report(monitor):
    report = monitor.report()
    report['physical_peak_bytes'] = max(s['memory_used_bytes']+s['memory_reserved_bytes'] for s in monitor.samples)
    report['physical_scope'] = ('whole-device AMD sysfs VRAM usage' if isinstance(monitor, ROCmTelemetry)
                                else 'whole-device NVML used plus driver/firmware reserved memory')
    report.pop('samples', None)
    return report
