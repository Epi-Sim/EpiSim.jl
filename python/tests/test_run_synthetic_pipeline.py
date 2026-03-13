import pytest
from unittest.mock import patch, MagicMock
import run_synthetic_pipeline
from pathlib import Path


def test_check_baseline_success_missing_json(tmp_path):
    # Test that missing JSON explicitly returns None for failed_profiles
    success_count, success_rate, should_proceed, failed_profiles = (
        run_synthetic_pipeline.check_baseline_success(
            baseline_dir=str(tmp_path), n_profiles_requested=10
        )
    )

    assert success_count == 0
    assert success_rate == 0.0
    assert should_proceed is False
    assert failed_profiles is None


@patch("run_synthetic_pipeline.run_baseline_batch")
@patch("run_synthetic_pipeline.check_baseline_success")
@patch("run_synthetic_pipeline.shutil")
@patch("run_synthetic_pipeline.run_stage")
def test_pipeline_retry_abort_on_missing_json(
    mock_run_stage, mock_shutil, mock_check, mock_run_batch, tmp_path
):
    # Setup mocks
    # return None for failed_profiles indicates missing BATCH_RESULTS.json
    mock_check.return_value = (0, 0.0, False, None)

    run_synthetic_pipeline.OUTPUT_FOLDER = tmp_path

    # Run should abort gracefully because we broke the infinite loop
    # We expect a RuntimeError or just breaking out of loop and proceeding to error
    with pytest.raises(RuntimeError, match="Failed to generate sufficient baselines"):
        run_synthetic_pipeline.run_two_phase_pipeline(
            n_profiles=10, batch_size=5, skip_sim=False, skip_process=True
        )

    # Should only run ONE time and immediately abort
    assert mock_check.call_count == 1


@patch("run_synthetic_pipeline.run_baseline_batch")
@patch("run_synthetic_pipeline.check_baseline_success")
@patch("run_synthetic_pipeline.shutil")
@patch("run_synthetic_pipeline.run_stage")
def test_pipeline_retry_logic_success_eventually(
    mock_run_stage, mock_shutil, mock_check, mock_run_batch, tmp_path
):
    # Setup mocks
    # Iteration 1: 5 success, 5 fail
    # Iteration 2: 10 success, 0 fail
    mock_check.side_effect = [
        (5, 0.5, False, [5, 6, 7, 8, 9]),
        (10, 1.0, True, []),
        (10, 1.0, True, []),
    ]
    run_synthetic_pipeline.OUTPUT_FOLDER = tmp_path

    # Should complete without error
    try:
        run_synthetic_pipeline.run_two_phase_pipeline(
            n_profiles=10, batch_size=5, skip_sim=False, skip_process=True
        )
    except Exception as e:
        pytest.fail(f"Pipeline failed unexpectedly: {e}")

    assert mock_check.call_count == 3
