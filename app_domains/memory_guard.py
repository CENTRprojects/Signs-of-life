"""Memory guard: monitor RAM usage during a run and adapt the parallelism to it.

The values set in .env / config.py (MAX_PROCESSES, MAX_WORKERS, WORKERS_POST_PROCESSING) are used as MAXIMUM values.
While running, when memory usage goes above MEM_HIGH_PCT the effective values are reduced (e.g. 5 -> 4 processes,
50 -> 40 async workers), and when memory goes back below MEM_LOW_PCT they are slowly increased back to the maximum.

Every change, every new memory peak and a summary per file chunk are written to logging/memory_<container>.log

Usage:
    from memory_guard import GOVERNOR, run_adaptive, adaptive_parallel
    GOVERNOR.start_monitor()                      # main process only: background sampling thread
    n_jobs = GOVERNOR.get("WORKERS_POST_PROCESSING", phase="dns")
    results = adaptive_parallel(func, items, phase="page_processing")
"""
import os
import gc
import time
import asyncio
import threading
from os.path import join
from datetime import datetime
from concurrent import futures

import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.executor import get_memmapping_executor
from joblib.externals.loky import reusable_executor

from config import RUN_CONFIG

MEM_HIGH_PCT = RUN_CONFIG.get("MEM_HIGH_PCT", 80)          # above: reduce parallelism by MEM_STEP_PCT
MEM_CRITICAL_PCT = RUN_CONFIG.get("MEM_CRITICAL_PCT", 90)  # above: reduce parallelism by MEM_CRITICAL_STEP_PCT
MEM_LOW_PCT = RUN_CONFIG.get("MEM_LOW_PCT", 65)            # below: increase parallelism back toward the max
MEM_STEP_PCT = RUN_CONFIG.get("MEM_STEP_PCT", 20)
MEM_CRITICAL_STEP_PCT = RUN_CONFIG.get("MEM_CRITICAL_STEP_PCT", 40)
MEM_RECOVERY_COOLDOWN_S = RUN_CONFIG.get("MEM_RECOVERY_COOLDOWN_S", 60)  # wait after a decrease before increasing
# wait between two decreases, so the running tasks have time to finish and free their memory
MEM_DECREASE_COOLDOWN_S = RUN_CONFIG.get("MEM_DECREASE_COOLDOWN_S", 30)
MEM_CRITICAL_COOLDOWN_S = RUN_CONFIG.get("MEM_CRITICAL_COOLDOWN_S", 10)
MEM_MONITOR_INTERVAL_S = RUN_CONFIG.get("MEM_MONITOR_INTERVAL_S", 5)
MEM_PEAK_LOG_STEP_PCT = RUN_CONFIG.get("MEM_PEAK_LOG_STEP_PCT", 2)  # log a new peak only if it grew by this much
MEM_SLICE_SIZE = RUN_CONFIG.get("MEM_SLICE_SIZE", 1000)   # items per Parallel call in adaptive_parallel

# minimum value for each adaptive setting
MIN_VALUES = {
    "MAX_PROCESSES": RUN_CONFIG.get("MIN_PROCESSES", 1),
    "MAX_WORKERS": RUN_CONFIG.get("MIN_WORKERS", 5),
    "WORKERS_POST_PROCESSING": RUN_CONFIG.get("MIN_WORKERS_POST_PROCESSING", 1),
}

LOG_PATH = join(RUN_CONFIG["MAIN_DIR"], "logging", f"memory_{RUN_CONFIG['CONTAINER_ID']}.log")

MB = 1024 * 1024


# ---------------------------------------------------------------------------------------------
# Memory reading
# ---------------------------------------------------------------------------------------------
def _read_int(path):
    try:
        with open(path) as f:
            value = f.read().strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _read_stat(path, key):
    try:
        with open(path) as f:
            for line in f:
                name, value = line.split()
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _read_cgroup():
    """Return (used_bytes, limit_bytes) of the container cgroup, or None when there is no memory limit.
    used = usage - inactive_file, which is what the OOM killer looks at (page cache can be reclaimed)"""
    limit = _read_int("/sys/fs/cgroup/memory.max")  # cgroup v2
    if limit is None:
        return None
    usage = _read_int("/sys/fs/cgroup/memory.current") or 0
    return usage - _read_stat("/sys/fs/cgroup/memory.stat", "inactive_file"), limit


def read_memory():
    """Return the current memory state: the tightest of the container limit and the host RAM"""
    vm = psutil.virtual_memory()
    host_used = vm.total - vm.available
    state = {"used_mb": host_used / MB, "limit_mb": vm.total / MB, "pct": 100.0 * host_used / vm.total,
             "source": "host"}
    cgroup = _read_cgroup()
    if cgroup is not None:
        used, limit = cgroup
        pct = 100.0 * used / limit
        if pct > state["pct"]:
            state = {"used_mb": used / MB, "limit_mb": limit / MB, "pct": pct, "source": "cgroup"}
    return state


def process_tree_rss_mb():
    """Return (main process RSS, sum of children RSS) in MB - children = joblib workers + chrome"""
    proc = psutil.Process()
    main_rss = proc.memory_info().rss / MB
    children_rss = 0.0
    for child in proc.children(recursive=True):
        try:
            children_rss += child.memory_info().rss / MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return main_rss, children_rss


def mem_log(msg):
    """Append one line to the memory log (safe from several processes: one small write per line)"""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} {msg}"
    print(f"[MEM] {line}")
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _fmt(state):
    return f"mem={state['pct']:.1f}% ({state['used_mb']:.0f}/{state['limit_mb']:.0f}MB {state['source']})"


def _decrease(value, minimum, pct):
    return max(minimum, min(value - 1, int(value * (100 - pct) / 100)))


def _increase(value, maximum, pct):
    return min(maximum, max(value + 1, int(value * (100 + pct) / 100)))


def _decide(pct, last_decrease):
    """Return (adjust function, step %, reason) for the memory usage pct, or None when nothing should change"""
    since_decrease = time.time() - last_decrease
    if pct >= MEM_CRITICAL_PCT and since_decrease > MEM_CRITICAL_COOLDOWN_S:
        return _decrease, MEM_CRITICAL_STEP_PCT, "CRITICAL"
    if MEM_HIGH_PCT <= pct and since_decrease > MEM_DECREASE_COOLDOWN_S:
        return _decrease, MEM_STEP_PCT, "HIGH"
    if pct < MEM_LOW_PCT and since_decrease > MEM_RECOVERY_COOLDOWN_S:
        return _increase, MEM_STEP_PCT, "RECOVERY"
    return None


# ---------------------------------------------------------------------------------------------
# Governor (main process)
# ---------------------------------------------------------------------------------------------
class MemoryGovernor:
    """Keep the effective parallelism values, adjusted to the memory usage.
    The values from RUN_CONFIG are the maximum ones"""

    def __init__(self):
        self.lock = threading.RLock()
        self.max_values = {k: max(MIN_VALUES[k], int(RUN_CONFIG[k])) for k in MIN_VALUES}
        self.current = dict(self.max_values)
        self.last_decrease = 0.0
        self.phase = "init"
        self.monitor_thread = None
        self.reset_stats()

    def reset_stats(self):
        with self.lock:
            self.peak = {"pct": 0.0}
            self.peak_logged_pct = 0.0
            self.n_adjustments = 0
            self.min_reached = dict(self.current)

    def set_phase(self, phase):
        self.phase = phase

    def get(self, key, phase=None):
        """Check the memory and return the current effective value for key"""
        self.check(phase)
        with self.lock:
            return self.current[key]

    def check(self, phase=None):
        """Read memory, record the peak and adjust the effective values"""
        if phase is not None:
            self.phase = phase
        state = read_memory()
        with self.lock:
            self._record_peak(state)
            decision = _decide(state["pct"], self.last_decrease)
            if decision is not None:
                self._adjust(state, *decision)
        if state["pct"] >= MEM_CRITICAL_PCT:
            gc.collect()
        return state

    def _adjust(self, state, func, step_pct, reason):
        changes = []
        for key, value in self.current.items():
            if func is _decrease:
                new_value = func(value, MIN_VALUES[key], step_pct)
            else:
                new_value = func(value, self.max_values[key], step_pct)
            if new_value != value:
                changes.append(f"{key} {value}->{new_value}")
                self.current[key] = new_value
                self.min_reached[key] = min(self.min_reached[key], new_value)
        if not changes:
            return
        self.n_adjustments += 1
        if func is _decrease:
            self.last_decrease = time.time()
        main_rss, children_rss = process_tree_rss_mb()
        mem_log(f"ADJUST {reason} phase={self.phase} {_fmt(state)} main_rss={main_rss:.0f}MB "
                f"children_rss={children_rss:.0f}MB | {', '.join(changes)}")

    def _record_peak(self, state):
        if state["pct"] > self.peak["pct"]:
            self.peak = dict(state, phase=self.phase, time=datetime.now().strftime("%H:%M:%S"))
            if state["pct"] >= max(MEM_LOW_PCT, self.peak_logged_pct + MEM_PEAK_LOG_STEP_PCT):
                self.peak_logged_pct = state["pct"]
                main_rss, children_rss = process_tree_rss_mb()
                mem_log(f"PEAK phase={self.phase} {_fmt(state)} main_rss={main_rss:.0f}MB "
                        f"children_rss={children_rss:.0f}MB | current={self.current}")

    def summary(self, label):
        """Log the peak / adjustments since the last summary, then reset them"""
        with self.lock:
            peak = self.peak
            mem_log(f"SUMMARY {label} | peak={peak['pct']:.1f}% ({peak.get('used_mb', 0):.0f}MB) "
                    f"at {peak.get('time')} phase={peak.get('phase')} | adjustments={self.n_adjustments} "
                    f"| min_reached={self.min_reached} | current={self.current} | max={self.max_values}")
        self.reset_stats()

    def start_monitor(self):
        """Background thread sampling memory every MEM_MONITOR_INTERVAL_S seconds (main process only)"""
        if self.monitor_thread is not None:
            return
        mem_log(f"START max={self.max_values} min={MIN_VALUES} thresholds low={MEM_LOW_PCT}% "
                f"high={MEM_HIGH_PCT}% critical={MEM_CRITICAL_PCT}% | {_fmt(read_memory())}")

        def _loop():
            while True:
                try:
                    self.check()
                except Exception as e:
                    print(f"[MEM] monitor error: {e}")
                time.sleep(MEM_MONITOR_INTERVAL_S)

        self.monitor_thread = threading.Thread(target=_loop, name="memory_monitor", daemon=True)
        self.monitor_thread.start()


GOVERNOR = MemoryGovernor()


# ---------------------------------------------------------------------------------------------
# Adaptive execution helpers
# ---------------------------------------------------------------------------------------------
def run_adaptive(func, items, key="MAX_PROCESSES", phase=None, dynamic_kwargs=None, **kwargs):
    """Run func(item, **kwargs) for each item in worker processes, keeping at most GOVERNOR.get(key) running.
    The limit is re-read each time a task finishes, so it can go down/up in the middle of the run.
    dynamic_kwargs = {argument name: governor key}: argument evaluated with the current value at each submission"""
    dynamic_kwargs = dynamic_kwargs or {}
    items = list(items)
    results = [None] * len(items)
    if not items:
        return results
    # same executor type as joblib Parallel (they share the global loky executor)
    executor = get_memmapping_executor(GOVERNOR.max_values[key])
    pending = list(range(len(items)))
    running = {}
    while pending or running:
        limit = GOVERNOR.get(key, phase)
        while pending and len(running) < limit:
            idx = pending.pop(0)
            task_kwargs = dict(kwargs)
            for arg, gov_key in dynamic_kwargs.items():
                task_kwargs[arg] = GOVERNOR.current[gov_key]
            running[executor.submit(func, items[idx], **task_kwargs)] = idx
        done, _ = futures.wait(list(running), timeout=MEM_MONITOR_INTERVAL_S, return_when=futures.FIRST_COMPLETED)
        for fut in done:
            results[running.pop(fut)] = fut.result()
    return results


def adaptive_parallel(func, items, phase=None, key="WORKERS_POST_PROCESSING", max_jobs=None):
    """Same as Parallel(n_jobs=RUN_CONFIG[key])(delayed(func)(x) for x in items), but the items are processed in
    slices of MEM_SLICE_SIZE and n_jobs is re-evaluated against the memory before each slice"""
    items = list(items)
    results = []
    with tqdm(total=len(items)) as progress:
        for start in range(0, len(items), MEM_SLICE_SIZE):
            n_jobs = GOVERNOR.get(key, phase)
            if max_jobs is not None:
                n_jobs = max(1, min(n_jobs, max_jobs))
            results += Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in items[start:start + MEM_SLICE_SIZE])
            progress.update(len(items[start:start + MEM_SLICE_SIZE]))
    return results


def release_workers(reason=""):
    """Kill the idle joblib worker processes (they keep the modules/models they loaded in memory)"""
    try:
        executor = getattr(reusable_executor, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"[MEM] release_workers error: {e}")
    gc.collect()
    mem_log(f"RELEASE workers ({reason}) | {_fmt(read_memory())}")


class AdaptiveAsyncLimiter:
    """Concurrency limit for the async requests of one worker process (MAX_WORKERS).
    Adjusted from inside the worker process, as each process has its own event loop"""

    def __init__(self, max_limit, min_limit=MIN_VALUES["MAX_WORKERS"]):
        self.max_limit = max(min_limit, max_limit)
        self.min_limit = min(min_limit, self.max_limit)
        self.limit = self.max_limit
        self.active = 0
        self.last_decrease = 0.0

    async def acquire(self):
        # single threaded event loop: no race between the check and the increment
        while self.active >= self.limit:
            await asyncio.sleep(0.2)
        self.active += 1

    def release(self):
        self.active -= 1

    async def monitor(self):
        while True:  # cancelled at the end of the batch
            state = read_memory()
            old = self.limit
            decision = _decide(state["pct"], self.last_decrease)
            if decision is not None:
                func, step_pct, reason = decision
                bound = self.min_limit if func is _decrease else self.max_limit
                self.limit = func(self.limit, bound, step_pct)
            if self.limit != old:
                if self.limit < old:
                    self.last_decrease = time.time()
                mem_log(f"ADJUST {reason} phase=requests {_fmt(state)} rss={psutil.Process().memory_info().rss / MB:.0f}MB "
                        f"active={self.active} | MAX_WORKERS {old}->{self.limit}")
            await asyncio.sleep(MEM_MONITOR_INTERVAL_S)
