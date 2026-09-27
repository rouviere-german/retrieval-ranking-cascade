"""The offline demo and the command line, both end to end with no network and no API key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retrieval_ranking_cascade.cli import main
from retrieval_ranking_cascade.demo import run_demo

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo_summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One demo run at ``tiny`` scale, shared by the assertions below."""
    return run_demo(tmp_path_factory.mktemp("demo"), scale="tiny")


# --- the demo's happy path ---


def test_the_demo_trains_a_bundle_and_serves_it(demo_summary: dict[str, Any]) -> None:
    assert demo_summary["scale"] == "tiny"
    assert len(str(demo_summary["bundle_id"])) == 12
    assert demo_summary["readyz"] == "ready"
    assert int(demo_summary["impressions"]) > 0
    assert float(demo_summary["recall_at_500"]) > 0.9
    assert int(demo_summary["items"]) == 10
    assert demo_summary["reranked"] is True
    assert demo_summary["degraded"] is False
    assert set(demo_summary["stage_ms"]) == {"retrieval", "ranking", "rules"}
    assert int(demo_summary["distinct_categories"]) >= 2


def test_the_demo_answers_an_unknown_user_from_popularity(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["cold_start_flagged"] is True
    assert int(demo_summary["cold_start_items"]) == 10


# --- the demo's failure paths, which are the point of it ---


def test_the_demo_shows_every_access_control_refusing_a_request(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["unauthenticated_status"] == 401
    assert demo_summary["admin_without_scope_status"] == 403
    assert demo_summary["cost_cap_status"] == 422
    assert demo_summary["cost_cap_code"] == "query_too_expensive"
    assert demo_summary["feedback_contract_status"] == 422


def test_the_demo_shows_a_tight_budget_degrading_rather_than_failing(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["tight_budget_degraded"] is True
    assert "skip_rerank" in demo_summary["tight_budget_reasons"]
    assert int(demo_summary["tight_budget_items"]) > 0


def test_the_demo_shows_the_canary_catching_a_corrupted_index_and_a_reindex_fixing_it(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["canary_alert_healthy"] is False
    assert float(demo_summary["canary_recall_healthy"]) > 0.9
    assert int(demo_summary["corrupted_vectors"]) > 0
    assert demo_summary["canary_alert_corrupted"] is True
    assert float(demo_summary["canary_recall_corrupted"]) < float(
        demo_summary["canary_recall_healthy"]
    )
    assert demo_summary["canary_alert_after_reindex"] is False
    assert float(demo_summary["canary_recall_after_reindex"]) > 0.9


def test_the_demo_shows_a_tampered_bundle_being_refused(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["tampered_bundle_status"] == 422
    assert demo_summary["tampered_bundle_code"] == "bundle_rejected"
    assert demo_summary["bundle_swap_status"] == 200


def test_the_demo_verifies_the_audit_chain_and_then_breaks_it(
    demo_summary: dict[str, Any],
) -> None:
    assert int(demo_summary["audit_entries"]) >= 3
    actions = demo_summary["audit_actions"]
    assert "bundle_swap_rejected" in actions
    assert "bundle_swap" in actions
    assert "reindex" in actions
    assert len(str(demo_summary["audit_head"])) == 12
    assert demo_summary["tampered_audit_detected"] is True


def test_the_demo_closes_the_feedback_loop_and_reads_the_monitors(
    demo_summary: dict[str, Any],
) -> None:
    assert int(demo_summary["feedback_accepted"]) > 0
    assert 0.0 < float(demo_summary["simulated_ctr"]) < 1.0
    assert float(demo_summary["catalog_coverage"]) > 0.0
    assert float(demo_summary["embedding_norm_psi"]) == pytest.approx(0.0, abs=1e-6)
    assert isinstance(demo_summary["drifted_features"], list)


def test_the_demo_recovers_the_examination_exponent_from_the_randomized_slice(
    demo_summary: dict[str, Any],
) -> None:
    estimated = float(demo_summary["estimated_eta"])
    truth = float(demo_summary["true_eta"])
    assert abs(estimated - truth) < 0.35


def test_the_demo_exposes_the_metric_families_a_dashboard_needs(
    demo_summary: dict[str, Any],
) -> None:
    exposed = demo_summary["metrics_exposed"]
    assert exposed == [
        "cascade_ann_recall_canary",
        "cascade_catalog_coverage",
        "cascade_degraded_responses_total",
        "cascade_embedding_norm_drift",
        "cascade_feature_psi",
        "cascade_stage_latency_seconds",
    ]


def test_the_whole_summary_is_json_serializable(demo_summary: dict[str, Any]) -> None:
    assert json.loads(json.dumps(demo_summary))["scale"] == "tiny"


# --- the command line ---


def test_help_and_a_missing_subcommand_are_handled(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as help_info:
        main(["--help"])
    assert help_info.value.code == 0
    assert "retrieval-ranking-cascade" in capsys.readouterr().out


def test_generate_api_key_prints_a_key_and_only_its_digest_is_to_be_stored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["generate-api-key"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert len(payload["api_key_sha256"]) == 64
    assert payload["api_key"] not in payload["api_key_sha256"]
    assert "APP_API_KEY_SHA256" in captured.err


@pytest.mark.usefixtures("restore_logging")
def test_train_writes_a_bundle_and_reports_its_metric_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``train`` reconfigures global logging to stderr, so this test restores it afterwards."""
    out = tmp_path / "bundle"
    assert main(["train", "--scale", "tiny", "--out", str(out), "--seed", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_dir"] == str(out)
    assert len(payload["bundle_id"]) == 64
    assert len(payload["metric_hash"]) == 64
    assert payload["metrics"]["impressions"] > 0
    assert (out / "manifest.json").is_file()


def test_bundle_verifies_a_bundle_on_disk(
    tiny_bundle_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bundle", str(tiny_bundle_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["scale"] == "tiny"
    assert len(payload["bundle_id"]) == 64


def test_bundle_reports_a_tampered_bundle_as_a_failure(
    tiny_bundle_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tiny_bundle_dir, broken)
    path = broken / "snapshot.npz"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    assert main(["bundle", str(broken)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "digest" in payload["error"]


def test_bundle_reports_a_missing_directory_as_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bundle", str(tmp_path / "nowhere")]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_audit_verify_accepts_a_good_chain_and_can_pin_the_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from retrieval_ranking_cascade.serving.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("key-1", "reindex")
    log.append("key-1", "bundle_swap")
    assert main(["audit", "verify", str(log.path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "entries": 2, "head": log.head}
    assert main(["audit", "verify", str(log.path), "--expect-head", log.head]) == 0
    capsys.readouterr()
    assert main(["audit", "verify", str(log.path), "--expect-head", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "head mismatch"


def test_audit_verify_reports_a_broken_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from retrieval_ranking_cascade.serving.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("key-1", "reindex")
    log.append("key-1", "bundle_swap")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    body = json.loads(lines[1])
    body["actor"] = "attacker"
    lines[1] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["audit", "verify", str(log.path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "entry 1" in payload["error"]


@pytest.mark.usefixtures("restore_logging")
def test_the_demo_subcommand_prints_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["demo", "--workdir", str(tmp_path / "demo"), "--scale", "tiny"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scale"] == "tiny"
    assert payload["tampered_audit_detected"] is True
