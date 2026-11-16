#!/usr/bin/env python
"""
run_local.py -- run the "big" jobs on ONE machine (laptop/desktop) instead of Slurm.

  python run_local.py --status                       # what is done / pending, no work started
  python run_local.py --only smoke                   # 1) environment smoke test (~10 min)
  python run_local.py --only testbed --seeds 0-9     # 2) a small slice first, to measure
  python run_local.py --only testbed --seeds 0-99    # 3) the full Stage-1 run (resumable)
  python run_local.py --only smoothness              # 4) the two smoothness-recovery fits (slow)

Design:
* A job = one command line. Jobs declare how many CPU cores and how much RAM they need; the scheduler
  starts a job only if the budget (--cpus, --mem-gb) allows. Defaults leave 1 core and ~25% RAM free.
* Checkpointing: a job writes to <out>.partial and is renamed on success, and a marker file is written to
  figures/done/. Re-running skips finished jobs; a job killed halfway restarts from scratch with no
  duplicate rows. Ctrl-C is safe.
* Each job runs single-threaded numerics (OMP/OPENBLAS/MKL = 1); parallelism comes from running jobs side by side.
* Logs: logs/local/<job>.out and .err
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
DONE = os.path.join(FIG, "done")
LOGS = os.path.join(ROOT, "logs", "local")
PY = sys.executable


@dataclass
class Job:
    name: str
    cmd: list
    cpus: int
    mem_gb: float
    out: str | None = None            # final result file (a .partial is written first)
    env: dict = field(default_factory=dict)

    @property
    def marker(self):
        return os.path.join(DONE, self.name + ".done")


def parse_range(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def build_jobs(a):
    jobs = []
    if "smoke" in a.only:
        jobs.append(Job("smoke", [PY, "-m", "models.bayesian_pop_trf"], cpus=2, mem_gb=3))
    if "testbed" in a.only:
        extra = a.testbed_args.split() if a.testbed_args else []
        for seed in parse_range(a.seeds):
            for snr in [x for x in a.snrs.split(",")]:
                out = os.path.join(FIG, f"testbed_snr{snr}_task{seed}.jsonl")
                cmd = [PY, "erp_testbed.py", "--modes", "null", "gain", "latency", "width", "--reps", "1",
                       "--seed", str(seed), "--n_patients", "8", "--n_electrodes", "4", "--n_times", "720000",
                       "--snr", snr, "--noise", "pink", "--common_noise", "0.3", "--effect_frac", "0.7",
                       "--shared_stimulus", "--boot", "200"] + extra
                jobs.append(Job(f"testbed_snr{snr}_seed{seed}", cmd, cpus=1, mem_gb=a.testbed_mem_gb, out=out))
    if "ieeg1" in a.only:
        extra = a.ieeg_args.split() if a.ieeg_args else []
        for seed in parse_range(a.seeds):
            for band in a.bands.split(","):
                for snr in a.snrs.split(","):
                    name = f"ieeg1{a.tag}_{band}_snr{snr}_seed{seed}"
                    out = os.path.join(FIG, name + ".jsonl")
                    cmd = [PY, "ieeg_stage1.py", "--band", band, "--snr", snr, "--seed", str(seed)] + extra
                    jobs.append(Job(name, cmd, cpus=1, mem_gb=a.ieeg_mem_gb, out=out))
    if "smoothness" in a.only:
        for h in ("amplitude", "hyperparameter"):
            out = os.path.join(FIG, f"smooth_{h}.jsonl")
            cmd = [PY, "smoothness_recovery_test.py", "--hierarchy", h, "--chains", "4"] + (
                a.smooth_args.split() if a.smooth_args else [])
            jobs.append(Job(f"smoothness_{h}", cmd, cpus=4, mem_gb=6, out=out))
    return jobs


def detect_resources():
    cpus = os.cpu_count() or 1
    mem = None
    try:
        import psutil
        mem = psutil.virtual_memory().total / 1e9
    except Exception:
        try:                                              # Linux / macOS fallback
            if sys.platform == "darwin":
                mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1e9
            else:
                mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except Exception:
            mem = None
    return cpus, mem


def launch(job):
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", **job.env)
    cmd = list(job.cmd)
    if job.out:
        part = job.out + ".partial"
        if os.path.exists(part):
            os.remove(part)
        cmd += ["--out", part]
    fo = open(os.path.join(LOGS, job.name + ".out"), "w")
    fe = open(os.path.join(LOGS, job.name + ".err"), "w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fo, stderr=fe)
    return p, fo, fe


def finalize(job, rc):
    if rc != 0:
        return False
    if job.out:
        part = job.out + ".partial"
        if not os.path.exists(part):
            return False
        os.replace(part, job.out)
    os.makedirs(DONE, exist_ok=True)
    open(job.marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=["testbed"], choices=["smoke", "testbed", "smoothness", "ieeg1"])
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--snrs", default="0.005,0.03,0.2")
    ap.add_argument("--cpus", type=int, default=None, help="cores to use (default: all but one)")
    ap.add_argument("--mem-gb", type=float, default=None, help="RAM to use (default: 75%% of total)")
    ap.add_argument("--testbed-mem-gb", type=float, default=6.0, help="declared RAM per testbed job")
    ap.add_argument("--testbed-args", default="", help="extra args appended to every testbed job (e.g. to shrink it)")
    ap.add_argument("--tag", default="", help="label for sensitivity variants (e.g. _e256) so results never overwrite the main run")
    ap.add_argument("--bands", default="lfp,hg", help="ieeg1: which signal bands to simulate (lfp = broad noise correlation, hg = local)")
    ap.add_argument("--ieeg-mem-gb", type=float, default=3.0, help="declared RAM per ieeg1 job (256 electrodes needs ~2x the 128-electrode value)")
    ap.add_argument("--ieeg-args", default="", help="extra args for ieeg1 jobs (e.g. --electrodes 256)")
    ap.add_argument("--smooth-args", default="", help="extra args for smoothness jobs")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=None, help="stop after launching this many jobs (for trials)")
    a = ap.parse_args()

    jobs = build_jobs(a)
    done = [j for j in jobs if os.path.exists(j.marker)]
    pending = [j for j in jobs if not os.path.exists(j.marker)]
    hc, hm = detect_resources()
    cpus = a.cpus or max(1, hc - 1)
    mem = a.mem_gb or ((hm * 0.75) if hm else 8.0)
    print(f"machine: {hc} cores, {('%.0f GB' % hm) if hm else 'unknown RAM'}  |  budget: {cpus} cores, {mem:.0f} GB")
    print(f"jobs: {len(jobs)} total, {len(done)} done, {len(pending)} pending")
    for j in pending:
        if j.cpus > cpus or j.mem_gb > mem:
            print(f"WARNING: job {j.name} needs {j.cpus} cores / {j.mem_gb} GB, more than your budget; it will never start.")
    if a.status or a.dry_run:
        if a.dry_run:
            for j in pending[:10]:
                print(" ", " ".join(j.cmd))
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
        return
    if a.max_jobs:
        pending = pending[: a.max_jobs]

    running, cpu_used, mem_used = [], 0, 0.0
    times, n_total, failed = [], len(pending), []
    t_start = time.time()
    try:
        while pending or running:
            for r in running[:]:
                job, p, fo, fe, t0 = r
                rc = p.poll()
                if rc is None:
                    continue
                fo.close(); fe.close()
                running.remove(r); cpu_used -= job.cpus; mem_used -= job.mem_gb
                ok = finalize(job, rc)
                times.append(time.time() - t0)
                if not ok:
                    failed.append(job.name)
                finished = n_total - len(pending) - len(running)
                avg = sum(times) / len(times)
                left = len(pending) + len(running)
                eta = avg * left / max(1, min(len(running) + 1, cpus // max(1, job.cpus)))
                print(f"[{time.strftime('%H:%M:%S')}] {'DONE ' if ok else 'FAIL '} {job.name}  ({times[-1] / 60:.1f} min)  "
                      f"finished {finished}/{n_total}  running {len(running)}  rough ETA {eta / 3600:.1f} h", flush=True)
            while pending and cpu_used + pending[0].cpus <= cpus and mem_used + pending[0].mem_gb <= mem:
                job = pending.pop(0)
                p, fo, fe = launch(job)
                running.append((job, p, fo, fe, time.time()))
                cpu_used += job.cpus; mem_used += job.mem_gb
                print(f"[{time.strftime('%H:%M:%S')}] start {job.name}", flush=True)
            if pending and not running:
                print("Nothing can start with this budget; stopping."); break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nInterrupted: stopping running jobs (unfinished ones restart next time).")
        for job, p, fo, fe, t0 in running:
            p.terminate()
        time.sleep(2)
    print(f"finished in {(time.time() - t_start) / 3600:.2f} h; failed: {failed if failed else 'none'}")
    if failed:
        print("Look at logs/local/<job>.err for the failed ones; re-run the same command to retry only those.")


if __name__ == "__main__":
    main()