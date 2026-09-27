"""Fixed-rate HTTP load test: the p99 latency SLO and the degraded-response rate under overload.

Usage:
  uv run python bench/bench_load.py                 # train, serve, load at 50, 100 and 200 RPS
  uv run python bench/bench_load.py --check         # also enforce bench/thresholds.json
  uv run python bench/bench_load.py --check --relax-latency   # CI: invariants only, no p99 limit
  uv run python bench/bench_load.py --bundle-dir var/bundle   # reuse a bundle instead of training

Three things about the method, each of which changed the numbers more than any code tweak did:

*Fixed rate, not as-fast-as-possible.* A closed-loop client that waits for each response before
sending the next measures throughput, and its latency flatters the server, because a slow response
delays the next request instead of queueing behind it. Requests here are scheduled at
``t0 + i / rate`` regardless of what has come back.

*The server is a separate process* - the repo's own ``serve`` entry point, one uvicorn worker, one
faiss thread. In-process, the load generator's event loop would share a GIL with request handling.

*The load generator is several processes.* This one is easy to get wrong. Measured on this machine,
one asyncio client process reported 52 RPS as the server's ceiling; three client processes against
the same server reached 119 RPS. A load test that is itself the bottleneck reports the client's
limit as the server's.

Phases: a warm-up at the overload rate (long enough for every worker thread to parse its booster
and for the cost estimators to settle - without it the *nominal* phase inherits a cold-start tail),
then the nominal rate, then 2x, then 4x. The check that matters is not "does it degrade at 2x" -
if the process keeps up at 2x then not degrading is the correct answer - but the invariant behind
the whole design: **whenever throughput falls behind the offered rate, a real share of the responses
must be degraded** - at least ``min_degraded_rate_when_behind`` (10%) of them. Falling behind while
still planning full work is the failure mode a budget controller exists to prevent. Errors must be
zero in every phase.

*Whose queue is it, though.* That invariant can only be asserted when the **server** is what fell
behind, and at high multiples on a small box it often is not: the clients' connection pools
saturate, requests wait inside httpx for a free connection, and the server never sees the offered
concurrency. A run like that reports seconds of client-observed latency while the server's own view
of the same requests - the queue wait it measured plus the time its stages took - stays in the low
milliseconds, and the budget controller is right not to degrade, because from inside the process
nothing is late. Every response carries that server-side view, so the two are compared directly:
when client-observed latency exceeds the server's own by more than ``HARNESS_LIMIT_RATIO`` at
either the median or the 99th percentile - the tail is where a connection pool saturates first - the
phase is reported as harness-limited and the shed-quality invariant is not asserted for it: the load
test is measuring itself. Nothing about the invariant itself is relaxed; a phase where the server is
what fell behind carries the wait in its own numbers, so both ratios stay near 1. Errors, throughput
at the nominal rate and the latency SLO are still enforced.

Nothing leaves the machine: the server binds a loopback port and the clients only talk to it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from retrieval_ranking_cascade.core.config import Settings
from retrieval_ranking_cascade.core.security import hash_api_key
from retrieval_ranking_cascade.pipeline import TrainingConfig, train_bundle
from retrieval_ranking_cascade.serving.bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "load.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"
API_KEY = "bench-load-key-" + "l" * 24
# (name, multiple of the nominal rate, seconds) - warm-up runs at the highest rate so every
# worker thread is touched before anything is measured.
PHASES: tuple[tuple[str, float], ...] = (
    ("warmup", 4.0),
    ("nominal", 1.0),
    ("overload_2x", 2.0),
    ("saturate_4x", 4.0),
)
MEASURED = ("nominal", "overload_2x", "saturate_4x")
# A phase whose client-observed p50 or p99 exceeds the server's own by more than this factor was
# bottlenecked outside the server; see the module docstring.
HARNESS_LIMIT_RATIO = 5.0
# The server here runs on its defaults, and stage 2's cost is linear in this one, so the run records
# it. Read from the field default rather than from Settings() so the value in the report is the one
# the repo ships and not whatever this shell has in its environment.
RERANK_DEPTH = int(Settings.model_fields["rerank_depth"].default)


# --------------------------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Latency and degradation for one load phase, aggregated over every client process."""

    name: str
    target_rps: float
    seconds: float
    requests: int
    errors: int
    degraded: int
    reranked: int
    latency_ms: dict[str, float]
    schedule_lag_ms: dict[str, float]
    server_ms: dict[str, float]
    max_in_flight: int
    max_contention: float
    degraded_reasons: dict[str, int]

    @property
    def achieved_rps(self) -> float:
        return self.requests / self.seconds if self.seconds > 0 else 0.0

    @property
    def harness_ratio(self) -> float:
        """How far client-observed latency runs above the server's own view of the same requests.

        The median alone is not enough. A client connection pool saturates at the tail first: some
        requests sit in httpx waiting for a free connection for seconds while the median still gets
        one quickly, which is how a phase can show a client p95 in the seconds against a server-side
        p95 in the tens of milliseconds and still look nearly healthy at p50. Both percentiles are
        compared and the larger ratio wins: either one far above 1 means the queue that produced the
        latency was not the server's.

        The asymmetry is safe: when the *server* is the bottleneck its own numbers include the wait
        (``queued_seconds`` is stamped on the event loop, before a worker thread picks the request
        up), so both ratios stay near 1 and the phase is judged as server-limited, which is what the
        shed-quality invariant is asserted against.
        """
        ratios = [
            self.latency_ms.get(percentile, 0.0) / self.server_ms[percentile]
            for percentile in ("p50", "p99")
            if self.server_ms.get(percentile, 0.0) > 0.0
        ]
        return max(ratios) if ratios else 1.0

    @property
    def harness_limited(self) -> bool:
        """True when the queue that produced this phase's latency was outside the server."""
        return self.harness_ratio > HARNESS_LIMIT_RATIO

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def degraded_rate(self) -> float:
        return self.degraded / self.requests if self.requests else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_rps": self.target_rps,
            "seconds": round(self.seconds, 3),
            "requests": self.requests,
            "achieved_rps": round(self.achieved_rps, 2),
            "errors": self.errors,
            "error_rate": round(self.error_rate, 6),
            "degraded": self.degraded,
            "degraded_rate": round(self.degraded_rate, 6),
            "reranked": self.reranked,
            "latency_ms": self.latency_ms,
            "schedule_lag_ms": self.schedule_lag_ms,
            "server_ms": self.server_ms,
            "harness_ratio": round(self.harness_ratio, 2),
            "harness_limited": self.harness_limited,
            "max_in_flight": self.max_in_flight,
            "max_contention": round(self.max_contention, 2),
            "degraded_reasons": self.degraded_reasons,
        }


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
        "max": round(float(values.max()), 3),
        "mean": round(float(values.mean()), 3),
    }


def _merge(name: str, target_rps: float, reports: list[dict[str, Any]]) -> PhaseResult:
    latencies: list[float] = []
    lags: list[float] = []
    server: list[float] = []
    reasons: dict[str, int] = {}
    for report in reports:
        latencies.extend(report["latencies"])
        lags.extend(report["lags"])
        server.extend(report["server_ms"])
        for reason, count in report["reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return PhaseResult(
        name=name,
        target_rps=target_rps,
        seconds=max((report["wall"] for report in reports), default=0.0),
        requests=sum(report["requests"] for report in reports),
        errors=sum(report["errors"] for report in reports),
        degraded=sum(report["degraded"] for report in reports),
        reranked=sum(report["reranked"] for report in reports),
        latency_ms=_percentiles(latencies),
        schedule_lag_ms=_percentiles(lags),
        server_ms=_percentiles(server),
        max_in_flight=max((report["max_in_flight"] for report in reports), default=0),
        max_contention=max((report["max_contention"] for report in reports), default=1.0),
        degraded_reasons=reasons,
    )


# --------------------------------------------------------------------------------------------- #
# The client worker (this file, re-invoked with --worker)
# --------------------------------------------------------------------------------------------- #


async def _phase(
    client: httpx.AsyncClient,
    *,
    rate: float,
    seconds: float,
    users: int,
    k: int,
    seed: int,
    offset: float,
) -> dict[str, Any]:
    """Fire ``rate`` requests per second for ``seconds``, recording every response."""
    rng = np.random.default_rng(seed)
    total = max(1, int(rate * seconds))
    ids = rng.integers(0, users, size=total)
    latencies: list[float] = []
    lags: list[float] = []
    server_ms: list[float] = []
    reasons: dict[str, int] = {}
    errors = degraded = reranked = 0
    max_in_flight = 0
    max_contention = 1.0
    lock = asyncio.Lock()

    async def one(index: int, start_at: float) -> None:
        nonlocal errors, degraded, reranked, max_in_flight, max_contention
        delay = start_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        lag = (time.perf_counter() - start_at) * 1000.0
        began = time.perf_counter()
        status = 0
        payload: dict[str, Any] = {}
        try:
            response = await client.post("/v1/recommend", json={"user_id": int(ids[index]), "k": k})
            status = response.status_code
            if status == 200:
                payload = response.json()
        except httpx.HTTPError:
            status = 0
        elapsed = (time.perf_counter() - began) * 1000.0
        # What the server thinks this request cost: the queue wait it measured plus its
        # stages. Anything the client saw beyond that was spent before the request reached
        # the event loop - in the accept backlog, or in this client's own connection pool.
        inside = float(payload.get("queued_ms", 0.0)) + sum(
            float(value) for value in payload.get("stage_ms", {}).values()
        )
        async with lock:
            latencies.append(elapsed)
            lags.append(lag)
            if status != 200:
                errors += 1
                return
            server_ms.append(inside)
            if payload.get("degraded"):
                degraded += 1
            for reason in payload.get("degraded_reasons", []):
                reasons[reason] = reasons.get(reason, 0) + 1
            reranked += bool(payload.get("reranked"))
            max_in_flight = max(max_in_flight, int(payload.get("in_flight", 0)))
            max_contention = max(max_contention, float(payload.get("contention", 1.0)))

    started = time.perf_counter() + offset
    await asyncio.gather(*(asyncio.create_task(one(i, started + i / rate)) for i in range(total)))
    return {
        "requests": total,
        "wall": time.perf_counter() - started,
        "errors": errors,
        "degraded": degraded,
        "reranked": reranked,
        "latencies": latencies,
        "lags": lags,
        "server_ms": server_ms,
        "reasons": reasons,
        "max_in_flight": max_in_flight,
        "max_contention": max_contention,
    }


async def _worker(args: argparse.Namespace) -> int:
    """Run every phase at this worker's share of the rate and print one JSON object."""
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=128)
    out: dict[str, Any] = {}
    async with httpx.AsyncClient(
        base_url=args.base_url, headers={"X-API-Key": API_KEY}, timeout=120.0, limits=limits
    ) as client:
        for index, (name, multiple) in enumerate(PHASES):
            out[name] = await _phase(
                client,
                rate=multiple * args.rate,
                seconds=args.warmup_seconds if name == "warmup" else args.seconds,
                users=args.users,
                k=args.k,
                seed=args.seed + index,
                offset=args.offset,
            )
    print(json.dumps(out))
    return 0


# --------------------------------------------------------------------------------------------- #
# The server and the orchestration
# --------------------------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env(bundle_dir: Path, audit_dir: Path, port: int) -> dict[str, str]:
    """Environment for the child server: which bundle to load and which limits to relax."""
    return {
        **os.environ,
        "APP_ENVIRONMENT": "test",
        "APP_HOST": "127.0.0.1",
        "APP_PORT": str(port),
        "APP_LOG_LEVEL": "WARNING",
        "APP_API_KEY_SHA256": hash_api_key(API_KEY),
        "APP_BUNDLE_DIR": str(bundle_dir),
        "APP_AUDIT_LOG_PATH": str(audit_dir / "admin.jsonl"),
        # The load generators share one API key, so the per-client token bucket would throttle them
        # long before the cascade does. The limiter is exercised in tests/integration/test_api.py.
        "APP_RATE_LIMIT_PER_SECOND": "1000000",
        "APP_RATE_LIMIT_BURST": "1000000",
        "APP_FAISS_THREADS": "1",
    }


class _ServerProcess:
    """Runs ``python -m retrieval_ranking_cascade serve`` and waits until /healthz answers."""

    def __init__(self, env: dict[str, str], port: int) -> None:
        self._env = env
        self.base_url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> _ServerProcess:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "retrieval_ranking_cascade", "serve"],
            cwd=ROOT,
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"the load-test server exited early with code {self._process.returncode}"
                )
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{self.base_url}/healthz", timeout=1.0).status_code == 200:
                    return self
            time.sleep(0.1)
        self.__exit__()
        raise RuntimeError("the load-test server did not become healthy in 180 s")

    def __exit__(self, *exc: object) -> None:
        process = self._process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)


def _run_clients(
    base_url: str,
    *,
    clients: int,
    rate: float,
    seconds: float,
    warmup_seconds: float,
    users: int,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Spawn the client processes, each taking ``rate / clients``, and collect their reports."""
    share = rate / clients
    processes = [
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--base-url",
                base_url,
                "--rate",
                str(share),
                "--seconds",
                str(seconds),
                "--warmup-seconds",
                str(warmup_seconds),
                "--users",
                str(users),
                "--k",
                str(k),
                "--seed",
                str(seed + 100 * index),
                # Stagger the schedules so arrivals interleave instead of pulsing together.
                "--offset",
                str(index / rate),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            text=True,
        )
        for index in range(clients)
    ]
    reports: list[dict[str, Any]] = []
    for process in processes:
        stdout, _ = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"a load client exited with code {process.returncode}")
        reports.append(json.loads(stdout.strip().splitlines()[-1]))
    return reports


def machine(clients: int) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
        "uvicorn_workers": 1,
        "faiss_threads": 1,
        "load_client_processes": clients,
    }


def run(
    *,
    bundle_dir: Path | None,
    scale: str,
    rate: float,
    seconds: float,
    warmup_seconds: float,
    k: int,
    seed: int,
    threads: int,
    clients: int,
) -> dict[str, Any]:
    if bundle_dir is None:
        target = ROOT / "var" / "bench-load" / "bundle"
        result = train_bundle(TrainingConfig(scale=scale, threads=threads), repo_root=ROOT)
        result.write(target)
        bundle_dir = target
    port = _free_port()
    bundle = load_bundle(bundle_dir)
    env = _server_env(bundle_dir, bundle_dir.parent / "audit", port)
    with _ServerProcess(env, port) as server:
        reports = _run_clients(
            server.base_url,
            clients=clients,
            rate=rate,
            seconds=seconds,
            warmup_seconds=warmup_seconds,
            users=bundle.n_users,
            k=k,
            seed=seed,
        )
        info = httpx.get(
            f"{server.base_url}/v1/bundle", headers={"X-API-Key": API_KEY}, timeout=30.0
        ).json()
    multiples = dict(PHASES)
    phases = [
        _merge(name, multiples[name] * rate, [report[name] for report in reports])
        for name in MEASURED
    ]
    return {
        "bundle_id": bundle.manifest.bundle_id,
        "scale": bundle.manifest.scale,
        "n_items": bundle.n_items,
        "candidate_k": info["candidate_k"],
        "ef_search": int(info["hnsw"]["ef_search"]),
        "rerank_depth": RERANK_DEPTH,
        "k": k,
        "clients": clients,
        "request_budget_ms": 50.0,
        "phases": [phase.as_dict() for phase in phases],
    }


def check_thresholds(report: dict[str, Any], *, relax_latency: bool = False) -> list[str]:
    """Enforce the SLO at the nominal rate, and the shed-quality invariant at every rate.

    ``relax_latency`` drops the absolute p50/p99 limits and the nominal degraded-rate limit, because
    those are properties of the documented reference CPU rather than of the code. A shared CI runner
    is slower, shares its cores with other tenants, and has no fixed clock; asserting 50 ms there
    would either be flaky or be a limit so loose it asserts nothing. The machine-independent
    invariants are still enforced in both modes: zero errors, throughput held at the nominal rate,
    and a degraded rate of at least ``min_degraded_rate_when_behind`` whenever the server falls
    behind the offered rate.
    """
    limits = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["load"]
    problems: list[str] = []
    phases = {phase["name"]: phase for phase in report["phases"]}
    nominal = phases["nominal"]
    if not relax_latency and nominal["latency_ms"]["p99"] > limits["max_p99_ms"]:
        problems.append(
            f"nominal p99 = {nominal['latency_ms']['p99']} ms exceeds {limits['max_p99_ms']} ms"
        )
    if not relax_latency and nominal["latency_ms"]["p50"] > limits["max_p50_ms"]:
        problems.append(
            f"nominal p50 = {nominal['latency_ms']['p50']} ms exceeds {limits['max_p50_ms']} ms"
        )
    if nominal["achieved_rps"] < limits["min_achieved_rps_fraction"] * nominal["target_rps"]:
        problems.append(
            f"nominal achieved only {nominal['achieved_rps']} of {nominal['target_rps']} RPS"
        )
    if not relax_latency and nominal["degraded_rate"] > limits["max_degraded_rate_nominal"]:
        problems.append(
            f"nominal degraded rate = {nominal['degraded_rate']} exceeds "
            f"{limits['max_degraded_rate_nominal']}"
        )
    keep_up = limits["keeps_up_rps_fraction"]
    for name, phase in phases.items():
        if phase["error_rate"] > limits["max_error_rate"]:
            problems.append(f"{name} error rate = {phase['error_rate']} above the limit")
        if phase["achieved_rps"] >= keep_up * phase["target_rps"]:
            continue
        if phase["harness_limited"]:
            # The queue was in the load generator, not in the server: client-observed latency
            # is far above the server's own view of the same requests, so the budget
            # controller saw nothing late to shed. Reported, not asserted.
            print(
                f"NOTE: {name} fell behind ({phase['achieved_rps']:.1f} of "
                f"{phase['target_rps']:.0f} RPS) but is harness-limited (client "
                f"p50/p99 {phase['latency_ms']['p50']:.1f}/{phase['latency_ms']['p99']:.1f} ms "
                f"against a server-side {phase['server_ms']['p50']:.1f}/"
                f"{phase['server_ms']['p99']:.1f} ms, worst ratio "
                f"{phase['harness_ratio']:.1f}x): the shed-quality invariant is not "
                f"asserted for this phase."
            )
            continue
        if phase["degraded_rate"] < limits["min_degraded_rate_when_behind"]:
            problems.append(
                f"{name} fell behind ({phase['achieved_rps']:.1f} of "
                f"{phase['target_rps']:.0f} RPS) while only "
                f"{phase['degraded_rate'] * 100:.1f}% of responses degraded: the budget "
                f"controller is not shedding quality"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fixed-rate HTTP load test")
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--scale", default="small", choices=["tiny", "smoke", "small", "full"])
    parser.add_argument("--rate", type=float, default=50.0, help="nominal requests per second")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=6.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--clients", type=int, default=3, help="load generator processes")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument(
        "--relax-latency",
        action="store_true",
        help="skip the absolute p50/p99 limits (for CI runners); keep the invariants",
    )
    # Worker mode: this file re-invokes itself as a load client.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", default="", help=argparse.SUPPRESS)
    parser.add_argument("--users", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--offset", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return asyncio.run(_worker(args))

    report = run(
        bundle_dir=args.bundle_dir,
        scale=args.scale,
        rate=args.rate,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
        k=args.k,
        seed=args.seed,
        threads=args.threads,
        clients=args.clients,
    )
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(args.clients), "report": report}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\nbundle {report['bundle_id'][:12]} ({report['scale']}, {report['n_items']} items), "
        f"candidate_k={report['candidate_k']}, efSearch={report['ef_search']}, "
        f"rerankDepth={report['rerank_depth']}, k={report['k']}, "
        f"budget {report['request_budget_ms']} ms, {report['clients']} client processes\n"
    )
    print(
        "| phase | target RPS | achieved | p50 ms | p95 ms | p99 ms | server p50 ms "
        "| errors | degraded | peak in-flight |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for phase in report["phases"]:
        latency = phase["latency_ms"]
        tag = " (harness-limited)" if phase["harness_limited"] else ""
        print(
            f"| {phase['name']}{tag} | {phase['target_rps']:.0f} | "
            f"{phase['achieved_rps']:.1f} | {latency['p50']} | {latency['p95']} | "
            f"{latency['p99']} | {phase['server_ms']['p50']} | {phase['errors']} | "
            f"{phase['degraded_rate'] * 100:.1f}% | {phase['max_in_flight']} |"
        )
    for phase in report["phases"]:
        if phase["degraded_reasons"]:
            print(f"\n{phase['name']} degradation reasons: {phase['degraded_reasons']}")
    if args.check:
        problems = check_thresholds(report, relax_latency=args.relax_latency)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("\nload thresholds met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
