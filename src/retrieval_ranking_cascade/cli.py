"""Command line entry point.

Installed as ``retrieval-ranking-cascade``; also runnable as
``python -m retrieval_ranking_cascade``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from retrieval_ranking_cascade.core.config import get_settings  # noqa: PLC0415
    from retrieval_ranking_cascade.core.logging import configure_logging  # noqa: PLC0415
    from retrieval_ranking_cascade.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )
    from retrieval_ranking_cascade.serving.api import create_app_from_settings  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_app_from_settings(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _train(scale: str, out: Path, seed: int) -> int:
    from retrieval_ranking_cascade.core.logging import configure_logging  # noqa: PLC0415
    from retrieval_ranking_cascade.pipeline import TrainingConfig, train_bundle  # noqa: PLC0415

    configure_logging("INFO", json_output=True, stream=sys.stderr)
    result = train_bundle(
        TrainingConfig(scale=scale, seed=seed), repo_root=Path(__file__).resolve().parents[2]
    )
    result.write(out)
    print(
        json.dumps(
            {
                "bundle_dir": str(out),
                "bundle_id": result.manifest.bundle_id,
                "metric_hash": result.metric_hash,
                "metrics": result.metrics,
                "durations": {k: round(v, 3) for k, v in result.durations.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _demo(workdir: Path, scale: str) -> int:
    from retrieval_ranking_cascade.core.logging import configure_logging  # noqa: PLC0415
    from retrieval_ranking_cascade.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_demo(workdir, scale=scale)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


def _audit_verify(path: Path, expect_head: str | None) -> int:
    from retrieval_ranking_cascade.serving.audit import AuditError, AuditLog  # noqa: PLC0415

    log = AuditLog(path)
    try:
        entries = log.verify()
    except AuditError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    head = log.head
    if expect_head is not None and head != expect_head:
        print(json.dumps({"ok": False, "error": "head mismatch", "head": head}, indent=2))
        return 1
    print(json.dumps({"ok": True, "entries": entries, "head": head}, indent=2))
    return 0


def _bundle_info(directory: Path) -> int:
    from retrieval_ranking_cascade.serving.bundle import (  # noqa: PLC0415
        BundleIntegrityError,
        load_bundle,
    )

    try:
        bundle = load_bundle(directory)
    except BundleIntegrityError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    manifest = bundle.manifest
    print(
        json.dumps(
            {
                "ok": True,
                "bundle_id": manifest.bundle_id,
                "scale": manifest.scale,
                "created_at": manifest.created_at,
                "n_items": manifest.n_items,
                "n_users": manifest.n_users,
                "data_sha256": manifest.data_sha256,
                "metrics": manifest.metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _json_default(value: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-ranking-cascade")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service from APP_BUNDLE_DIR")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    train = sub.add_parser("train", help="train a serving bundle and write it to disk")
    train.add_argument("--scale", default="small", choices=["tiny", "smoke", "small", "full"])
    train.add_argument("--out", type=Path, default=Path("var/bundle"))
    train.add_argument("--seed", type=int, default=7)

    bundle = sub.add_parser("bundle", help="verify a bundle on disk and print its manifest")
    bundle.add_argument("directory", type=Path)

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))
    demo.add_argument("--scale", default="small", choices=["tiny", "smoke", "small"])

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from retrieval_ranking_cascade.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "train":
        return _train(args.scale, args.out, args.seed)
    if args.command == "bundle":
        return _bundle_info(args.directory)
    if args.command == "audit":
        return _audit_verify(args.path, args.expect_head)
    return _demo(args.workdir, args.scale)


if __name__ == "__main__":
    raise SystemExit(main())
