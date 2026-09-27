"""
Per-repetition telemetry and per-session environment capture.

Sampler — one NVML/psutil thread for the duration of a single request:
  VRAM, power, temperature, SM clock, PCIe link gen/width, clock-event
  (throttle) reasons, process RSS; plus the hardware energy counter read at
  start/stop for exact per-request energy. A second thread samples PCIe TX/RX
  throughput (each NVML call blocks ~30 ms, so it can't share the fast loop).

environment() — the reproducibility manifest written once per session (M5).
"""

import json
import platform
import socket
import subprocess
import sys
import threading
import time

from config import GGUF_DIR, LLAMACPP_BIN, OLLAMA_BIN, ROOT

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

try:
    import psutil
except ImportError:
    psutil = None

# clock-event reasons that indicate the measurement was throttled
_THROTTLE_BITS = {
    "sw_power_cap":     0x0000000000000004,
    "hw_slowdown":      0x0000000000000008,
    "sw_thermal":       0x0000000000000020,
    "hw_thermal":       0x0000000000000040,
    "hw_power_brake":   0x0000000000000080,
}

def gpu_count():
    return pynvml.nvmlDeviceGetCount() if _NVML else 0

def _handles():
    return {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count())}

def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def vram_used_mib():
    """Instant per-GPU VRAM use — taken before a server starts, as the baseline
    for attributing VRAM to the model (observed placement, M8)."""
    if not _NVML:
        return {}
    return {i: pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2 for i, h in _handles().items()}

def gpu_temps():
    if not _NVML:
        return {}
    return {i: _try(lambda h=h: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            for i, h in _handles().items()}

def wait_cool(max_temp_c, timeout_s=300):
    """Block until every GPU is at or below max_temp_c (or timeout). Returns temps."""
    t0 = time.time()
    while True:
        temps = gpu_temps()
        if not temps or all(t is not None and t <= max_temp_c for t in temps.values()):
            return temps
        if time.time() - t0 > timeout_s:
            print(f"    [thermal] still {temps} after {timeout_s}s — continuing", flush=True)
            return temps
        time.sleep(5)

# ── Sampler ───────────────────────────────────────────────────────────────────
class Sampler:
    def __init__(self, pid=None, proc_name=None, interval=0.1, pcie=True):
        self._pid, self._proc_name = pid, proc_name
        self._interval, self._pcie = interval, pcie
        self._stop = threading.Event()
        self._threads = []
        self.n_gpus = gpu_count()
        g = range(self.n_gpus)
        self.vram_peak   = {i: 0.0 for i in g}
        self.power_samples = []                      # (t, total W over supported GPUs)
        self.power_gpus  = set()
        self.temp_max    = {i: None for i in g}
        self.sm_clk      = {i: [] for i in g}
        self.link_gen    = {i: 0 for i in g}
        self.link_width  = {i: 0 for i in g}
        self.throttle    = set()
        self.pcie_rx     = {i: [] for i in g}        # KB/s samples
        self.pcie_tx     = {i: [] for i in g}
        self.rss_peak_mib = 0.0
        self._energy0 = self._energy1 = None
        self.t0 = self.t1 = None

    def _energy(self):
        if not _NVML:
            return None
        vals = {i: _try(lambda h=h: pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
                for i, h in _handles().items()}
        return vals

    def _rss(self):
        if psutil is None:
            return 0.0
        if self._pid is not None:
            pids = [self._pid]
        elif self._proc_name:
            pids = [p.info["pid"] for p in psutil.process_iter(["pid", "name"])
                    if self._proc_name.lower() in (p.info["name"] or "").lower()]
        else:
            return 0.0
        best = 0.0
        for pid in pids:
            try:
                best = max(best, psutil.Process(pid).memory_info().rss / 1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return best

    def _fast_loop(self):
        hs = _handles() if _NVML else {}
        while not self._stop.is_set():
            total_w = 0.0
            for i, h in hs.items():
                mem = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2, 0.0)
                self.vram_peak[i] = max(self.vram_peak[i], mem)
                w = _try(lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000)
                if w is not None:
                    total_w += w
                    self.power_gpus.add(i)
                t = _try(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                if t is not None:
                    self.temp_max[i] = max(self.temp_max[i] or 0, t)
                c = _try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                if c:
                    self.sm_clk[i].append(c)
                self.link_gen[i]   = max(self.link_gen[i],
                                         _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h), 0))
                self.link_width[i] = max(self.link_width[i],
                                         _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(h), 0))
                reasons = _try(lambda: pynvml.nvmlDeviceGetCurrentClocksEventReasons(h), 0)
                for name, bit in _THROTTLE_BITS.items():
                    if reasons & bit:
                        self.throttle.add(f"gpu{i}:{name}")
            self.power_samples.append((time.perf_counter(), total_w))
            self.rss_peak_mib = max(self.rss_peak_mib, self._rss())
            self._stop.wait(self._interval)

    def _pcie_loop(self):
        hs = _handles() if _NVML else {}
        while not self._stop.is_set():
            for i, h in hs.items():
                rx = _try(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES))
                tx = _try(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES))
                if rx is not None: self.pcie_rx[i].append(rx)
                if tx is not None: self.pcie_tx[i].append(tx)
            self._stop.wait(0.05)

    def start(self):
        self._energy0 = self._energy()
        self.t0 = time.perf_counter()
        loops = [self._fast_loop] + ([self._pcie_loop] if self._pcie else [])
        for fn in loops:
            th = threading.Thread(target=fn, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
        self.t1 = time.perf_counter()
        self._energy1 = self._energy()
        return self

    # ── summaries ──
    def energy_j(self):
        """Hardware energy counter delta (mJ → J), per GPU and total; None if unsupported."""
        if not self._energy0 or not self._energy1:
            return None, {}
        per = {}
        for i in self._energy0:
            a, b = self._energy0.get(i), self._energy1.get(i)
            if a is not None and b is not None:
                per[i] = (b - a) / 1000
        return (sum(per.values()) if per else None), per

    def avg_watts(self):
        ws = [w for _, w in self.power_samples]
        return sum(ws) / len(ws) if ws else None

    def peak_watts(self):
        return max((w for _, w in self.power_samples), default=None)

    def tail_watts(self, seconds):
        """Mean power over the final `seconds` of the request — decode is the last
        phase of a non-streaming request, so this isolates decode power from prefill."""
        if not seconds or not self.t1:
            return None
        ws = [w for t, w in self.power_samples if t >= self.t1 - seconds]
        return sum(ws) / len(ws) if ws else None

    def summary(self, baseline_vram=None):
        baseline_vram = baseline_vram or {}
        e_total, e_per = self.energy_j()
        out = {
            "wall_s":        (self.t1 - self.t0) if self.t1 else None,
            "energy_j":      e_total,
            "avg_watts":     self.avg_watts(),
            "peak_watts":    self.peak_watts(),
            "power_gpus":    ",".join(map(str, sorted(self.power_gpus))),
            "throttle":      ";".join(sorted(self.throttle)),
            "peak_rss_mib":  self.rss_peak_mib or None,
        }
        observed = []
        for i in range(self.n_gpus):
            delta = self.vram_peak[i] - baseline_vram.get(i, 0.0)
            if delta > 512:
                observed.append(i)
            clk = self.sm_clk[i]
            out |= {
                f"gpu{i}_vram_mib":       self.vram_peak[i],
                f"gpu{i}_vram_delta_mib": delta,
                f"gpu{i}_energy_j":       e_per.get(i),
                f"gpu{i}_temp_max_c":     self.temp_max[i],
                f"gpu{i}_sm_clk_mhz":     sum(clk) / len(clk) if clk else None,
                f"gpu{i}_pcie_gen":       self.link_gen[i] or None,
                f"gpu{i}_pcie_width":     self.link_width[i] or None,
                f"gpu{i}_pcie_rx_mbs":    (sum(self.pcie_rx[i]) / len(self.pcie_rx[i]) / 1024)
                                          if self.pcie_rx[i] else None,
                f"gpu{i}_pcie_tx_mbs":    (sum(self.pcie_tx[i]) / len(self.pcie_tx[i]) / 1024)
                                          if self.pcie_tx[i] else None,
            }
        out["observed_gpus"] = ",".join(map(str, observed))
        return out

def gpu_fields(n_gpus):
    per = ["vram_mib", "vram_delta_mib", "energy_j", "temp_max_c", "sm_clk_mhz",
           "pcie_gen", "pcie_width", "pcie_rx_mbs", "pcie_tx_mbs"]
    return [f"gpu{i}_{k}" for i in range(n_gpus) for k in per]

# ── Environment manifest ──────────────────────────────────────────────────────
def _cmd_out(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"unavailable ({e.__class__.__name__})"

def _gpu_env():
    if not _NVML:
        return []
    out = []
    for i, h in _handles().items():
        pci = _try(lambda: pynvml.nvmlDeviceGetPciInfo(h))
        out.append({
            "index":             i,
            "name":              _try(lambda: pynvml.nvmlDeviceGetName(h)),
            "uuid":              _try(lambda: pynvml.nvmlDeviceGetUUID(h)),
            "pci_bus_id":        _try(lambda: pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId),
            "vbios":             _try(lambda: pynvml.nvmlDeviceGetVbiosVersion(h)),
            "vram_total_mib":    _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024**2),
            "power_limit_w":     _try(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000),
            "pcie_gen_max":      _try(lambda: pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)),
            "pcie_gen_gpu_max":  _try(lambda: pynvml.nvmlDeviceGetGpuMaxPcieLinkGeneration(h)),
            "pcie_width_max":    _try(lambda: pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)),
            "pcie_gen_idle":     _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)),
            "pcie_width_idle":   _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)),
            "power_readable":    _try(lambda: pynvml.nvmlDeviceGetPowerUsage(h)) is not None,
            "energy_readable":   _try(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h)) is not None,
        })
    return out

def _git():
    head  = _cmd_out(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = _cmd_out(["git", "-C", str(ROOT), "status", "--porcelain"])
    # nearest release tag, e.g. "v0.2.0" or "v0.2.0-3-gabc1234" (3 commits past the tag)
    version = _cmd_out(["git", "-C", str(ROOT), "describe", "--tags", "--always"])
    return {"version": version, "commit": head, "dirty": bool(dirty.strip())}

def environment(args=None):
    mem = psutil.virtual_memory().total / 1024**3 if psutil else None
    driver = _try(lambda: pynvml.nvmlSystemGetDriverVersion()) if _NVML else None
    cuda   = _try(lambda: pynvml.nvmlSystemGetCudaDriverVersion()) if _NVML else None
    return {
        "captured_at":     time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host":            socket.gethostname(),
        "os":              platform.platform(),
        "python":          sys.version.split()[0],
        "cpu":             platform.processor() or platform.machine(),
        "cpu_count":       psutil.cpu_count(logical=True) if psutil else None,
        "ram_total_gib":   mem,
        "nvidia_driver":   driver,
        "cuda_driver_api": cuda,
        "gpus":            _gpu_env(),
        "llamacpp_bin":    str(LLAMACPP_BIN),
        "llamacpp_version": _cmd_out([str(LLAMACPP_BIN), "--version"]) if LLAMACPP_BIN.exists() else None,
        "ollama_version":  _cmd_out([OLLAMA_BIN, "--version"]) if OLLAMA_BIN else None,
        "gguf_dir":        str(GGUF_DIR),
        "harness_git":     _git(),
        "argv":            sys.argv,
        "args":            {k: v for k, v in vars(args).items() if k != "func"} if args else None,
    }

def write_environment(path, args=None):
    env = environment(args)
    path.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    return env
