"""Runtime fault injections at declared nodes (lane-A mutators, runtime part).

F05_POST_PLANNER_TARGET_DRIFT
    Declared node: ``planner`` when it plans ``model_building``.  After the original
    planner returns, the update additionally rewrites
    ``target_spec.target_columns`` to a pre-registered wrong column W, leaving every
    other contract field (task family, metric, submission spec) unchanged.  This is
    the manuscript's own drift definition: the workflow consumes mutually
    inconsistent versions of C after intake approval.

F10_ID_PREDICTION_PERMUTATION
    Declared node: ``build_submission`` inside ``model_building``.  After the
    original serializer returns, the non-ID columns are row-permuted with a seeded
    derangement; ID set, ID order, row count and column schema are unchanged.

File-level mutations (B01, F01, F03) are applied to the task package before the
run by ``cases/mutation_specs/mutators.py``; nothing here touches the data files.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lane_a.common import pairing_digest, sha256_json
from lane_a.context import ctx


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= 1:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def planner_drift_wrapper(spec: dict[str, Any]):
    wrong = list(spec["wrong_target_columns"])

    def outer(fn):
        async def planner_with_drift(state):
            updates = await fn(state)
            c = ctx()
            already = any(item.get("template_id") == "F05" for item in c.injections_applied)
            if already or not isinstance(updates, dict) or updates.get("current_phase") != "model_building":
                return updates
            before = state.target_spec.model_dump(mode="python")
            after = dict(before)
            after["target_columns"] = list(wrong)
            updates["target_spec"] = after
            record = {
                "template_id": "F05",
                "node": "planner->model_building",
                "before_target_columns": list(before.get("target_columns") or []),
                "after_target_columns": list(wrong),
                "before_sha256": sha256_json(before),
                "after_sha256": sha256_json(after),
            }
            c.injections_applied.append(record)
            c.logger.emit(
                "injection",
                actor="lane_a_mutator",
                phase="model_building",
                contract_digest=sha256_json(after),
                payload=record,
            )
            return updates

        planner_with_drift.__name__ = getattr(fn, "__name__", "planner_node")
        return planner_with_drift

    return outer


def permutation_wrapper(spec: dict[str, Any]):
    seed = int(spec.get("permutation_seed", 0))

    def outer(fn):
        def build_submission_permuted(test_df, predictions, **kwargs):
            frame = fn(test_df, predictions, **kwargs)
            c = ctx()
            id_cols = [col for col in (kwargs.get("id_columns") or []) if col in frame.columns]
            value_cols = [col for col in frame.columns if col not in id_cols]
            if not value_cols or len(frame) < 2:
                c.logger.emit(
                    "injection",
                    actor="lane_a_mutator",
                    phase="model_building",
                    error_subtype="injection_not_applicable",
                    payload={"template_id": "F10", "reason": "no value columns or <2 rows"},
                )
                return frame
            before_digest = pairing_digest(frame, id_cols, value_cols)
            perm = _derangement(len(frame), seed)
            permuted = frame.copy()
            permuted[value_cols] = frame[value_cols].to_numpy()[perm]
            after_digest = pairing_digest(permuted, id_cols, value_cols)
            record = {
                "template_id": "F10",
                "node": "model_building.build_submission",
                "permutation_seed": seed,
                "n_rows": int(len(frame)),
                "fixed_points": int(np.sum(perm == np.arange(len(frame)))),
                "before_pairing_digest": before_digest,
                "after_pairing_digest": after_digest,
                "id_columns": id_cols,
                "value_columns": value_cols,
            }
            c.injections_applied.append(record)
            c.logger.emit("injection", actor="lane_a_mutator", phase="model_building", artifact_mapping_digest=after_digest, payload=record)
            return permuted

        build_submission_permuted.__name__ = "build_submission"
        return build_submission_permuted

    return outer


def node_inner_wrappers(injection: dict[str, Any]) -> dict[str, Any]:
    if (injection or {}).get("template_id") == "F05":
        return {"planner": planner_drift_wrapper(injection)}
    return {}


def action_inner_wrappers(injection: dict[str, Any]) -> dict[str, Any]:
    if (injection or {}).get("template_id") == "F10":
        return {"build_submission": permutation_wrapper(injection)}
    return {}
