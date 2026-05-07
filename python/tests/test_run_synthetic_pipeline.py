from unittest.mock import patch

import pytest

import run_synthetic_pipeline


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


@patch("run_synthetic_pipeline.run_baseline_batch")
@patch("run_synthetic_pipeline.check_baseline_success")
@patch("run_synthetic_pipeline.shutil")
@patch("run_synthetic_pipeline.run_stage")
def test_pipeline_skips_phase2_when_no_interventions_requested(
    mock_run_stage, mock_shutil, mock_check, mock_run_batch, tmp_path
):
    mock_check.return_value = (10, 1.0, True, [])
    run_synthetic_pipeline.OUTPUT_FOLDER = tmp_path

    run_synthetic_pipeline.run_two_phase_pipeline(
        n_profiles=10,
        batch_size=5,
        skip_sim=False,
        skip_process=False,
        intervention_profile_fraction=0.0,
    )

    stage_names = [call.args[0] for call in mock_run_stage.call_args_list]
    process_cmd = next(
        call.args[1]
        for call in mock_run_stage.call_args_list
        if call.args[0] == "Process Baselines"
    )

    assert "Process Baselines" in stage_names
    assert "Generate Spike-Based Interventions" not in stage_names
    assert "--cases-missing-rate" in process_cmd
    assert "--hosp-missing-rate" in process_cmd
    assert "--deaths-missing-rate" in process_cmd
    assert "--ww-missing-rate" in process_cmd
    assert "--sparsity-mode" not in process_cmd
    assert "--sparsity-seed" not in process_cmd


@patch("run_synthetic_pipeline.run_baseline_batch")
@patch("run_synthetic_pipeline.check_baseline_success")
@patch("run_synthetic_pipeline.shutil")
@patch("run_synthetic_pipeline.run_stage")
def test_pipeline_prefers_nvme_baseline_dir_for_phase2(
    mock_run_stage, mock_shutil, mock_check, mock_run_batch, tmp_path
):
    mock_check.return_value = (2, 1.0, True, [])
    output_folder = tmp_path / "runs" / "smoke"
    nvme_base = tmp_path / "nvme"
    run_synthetic_pipeline.OUTPUT_FOLDER = output_folder

    baseline_dir = nvme_base / "baselines"
    baseline_dir.mkdir(parents=True)
    (nvme_base / "raw_synthetic_observations.zarr").mkdir()

    run_synthetic_pipeline.run_two_phase_pipeline(
        n_profiles=2,
        batch_size=1,
        skip_sim=False,
        skip_process=False,
        intervention_profile_fraction=0.5,
        nvme_base=str(nvme_base),
    )

    intervention_calls = [
        call for call in mock_run_stage.call_args_list
        if call.args[0] == "Generate Spike-Based Interventions"
    ]
    assert len(intervention_calls) == 1
    intervention_cmd = intervention_calls[0].args[1]
    assert "--intervention-only" in intervention_cmd
    assert str(baseline_dir) in intervention_cmd


@patch("run_synthetic_pipeline.run_baseline_batch")
@patch("run_synthetic_pipeline.check_baseline_success")
@patch("run_synthetic_pipeline.shutil")
@patch("run_synthetic_pipeline.run_stage")
def test_pipeline_syncs_final_zarr_from_nvme_when_phase2_skipped(
    mock_run_stage, mock_shutil, mock_check, mock_run_batch, tmp_path
):
    mock_check.return_value = (10, 1.0, True, [])
    output_folder = tmp_path / "runs" / "smoke"
    nvme_base = tmp_path / "nvme"
    final_zarr = output_folder / "raw_synthetic_observations.zarr"
    nvme_zarr = nvme_base / "raw_synthetic_observations.zarr"

    run_synthetic_pipeline.OUTPUT_FOLDER = output_folder
    nvme_zarr.mkdir(parents=True)

    run_synthetic_pipeline.run_two_phase_pipeline(
        n_profiles=10,
        batch_size=5,
        skip_sim=False,
        skip_process=False,
        intervention_profile_fraction=0.0,
        nvme_base=str(nvme_base),
    )

    mock_shutil.copytree.assert_called_with(nvme_zarr, final_zarr)
