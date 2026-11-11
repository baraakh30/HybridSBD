"""Unit tests of the scoring protocol (``hybridsbd.scoring``); numpy only, run with ``pytest tests``."""

import numpy as np
import pytest

from hybridsbd.scoring import (
    apply_min_scene_length,
    best_setting,
    build_pred_many_hot,
    evaluate_scenes,
    predictions_to_scenes,
    reconstruct_full_video,
    sweep,
    transitions_to_scenes,
)


def test_predictions_to_scenes_splits_at_transitions():
    pred = np.zeros(20, dtype=np.uint8)
    pred[5] = 1
    pred[7] = 1
    assert predictions_to_scenes(pred).tolist() == [[0, 4], [6, 6], [8, 19]]


def test_all_ones_track_is_a_single_scene():
    assert predictions_to_scenes(np.ones(10, dtype=np.uint8)).tolist() == [[0, 9]]


def test_min_scene_length_merges_short_scenes():
    scenes = np.array([[0, 4], [6, 6], [8, 19]], dtype=np.int32)
    assert apply_min_scene_length(scenes, 1).tolist() == scenes.tolist()
    assert apply_min_scene_length(scenes, 2).tolist() == [[0, 6], [8, 19]]


def test_single_output_rule_ignores_z_and_dual_rule_unions_it():
    y = np.array([0.0, 0.9, 0.0, 0.0], dtype=np.float32)
    z = np.full(4, 0.8, dtype=np.float32)
    assert build_pred_many_hot(y, z, 0.5, None).tolist() == [0, 1, 0, 0]
    assert build_pred_many_hot(y, z, 0.5, 0.3).tolist() == [1, 1, 1, 1]


def test_transitions_to_scenes():
    assert transitions_to_scenes([[10, 11], [30, 40]], 100).tolist() == [[0, 10], [11, 30], [40, 99]]
    assert transitions_to_scenes([], 50).tolist() == [[0, 49]]


def test_scene_matching_uses_the_frame_tolerance():
    gt = np.array([[0, 49], [50, 99]])
    near = np.array([[0, 50], [51, 99]])  # boundary one frame late
    far = np.array([[0, 60], [61, 99]])  # boundary ten frames late
    assert evaluate_scenes(gt, near, 2)[3] == (1, 0, 0)
    assert evaluate_scenes(gt, far, 2)[3] == (0, 1, 1)


def test_overlapping_windows_are_averaged():
    clips = {
        "v": [(0, np.full(128, 0.2, np.float32), np.zeros(128, np.float32)), (64, np.full(128, 0.6, np.float32), np.zeros(128, np.float32))]
    }
    y, _ = reconstruct_full_video(clips)
    assert y["v"].shape == (192,)
    assert np.isclose(y["v"][0], 0.2) and np.isclose(y["v"][100], 0.4) and np.isclose(y["v"][191], 0.6)


def _toy_case():
    y = np.zeros(200, dtype=np.float32)
    y[[50, 120]] = 0.95
    gt = {"v": np.array([[0, 49], [51, 119], [121, 199]], dtype=np.int32)}
    return {"v": y}, {"v": np.zeros(200, dtype=np.float32)}, gt


def test_sweep_finds_the_perfect_threshold_and_has_the_right_grid():
    vy, vz, gt = _toy_case()
    res = sweep(vy, vz, gt, min_scene_lengths=[1, 2])
    assert res["overall"]["f1"] == 1.0
    assert len(res["all_results"]) == 17 * 4 * 2
    assert res["best_min_scene_len"] in (1, 2)
    single = sweep(vy, None, gt, single_output=True)
    assert len(single["all_results"]) == 17 and single["best_gradual_threshold"] is None and "best_min_scene_len" not in single


def test_sweep_ties_keep_the_earliest_setting():
    vy, vz, gt = _toy_case()
    res = sweep(vy, vz, gt, single_output=True)
    first = next(iter(res["all_results"]))
    assert res["best_boundary_threshold"] == first[0]  # every tau_c up to 0.9 is perfect here, the first one wins


def test_best_setting_can_restrict_the_minimum_scene_length():
    vy, vz, gt = _toy_case()
    grid = sweep(vy, vz, gt, min_scene_lengths=[1, 2, 3])["all_results"]
    key, _ = best_setting(grid, 2)
    assert key[2] == 2


def test_videos_without_ground_truth_are_skipped():
    vy, vz, gt = _toy_case()
    vy["other"], vz["other"] = vy["v"].copy(), vz["v"].copy()
    assert sweep(vy, vz, gt)["num_videos"] == 1


def test_bbc_rule_ignores_predictions_after_the_annotated_range():
    y = np.zeros(300, dtype=np.float32)
    y[[50, 250]] = 0.95  # the second spike lies in the unannotated tail
    gt = {"v": np.array([[0, 49], [51, 199]], dtype=np.int32)}
    plain = sweep({"v": y}, None, gt, single_output=True)
    clipped = sweep({"v": y}, None, gt, single_output=True, clip_to_annotated_range=True)
    assert clipped["overall"]["f1"] == 1.0 and plain["overall"]["f1"] < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
