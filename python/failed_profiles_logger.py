"""
Failed profiles logger for tracking and analyzing invalid parameter combinations.

This module provides utilities to:
1. Log failed simulation profiles with their parameters and error details
2. Analyze failure patterns to improve parameter generation
3. Generate validation rules based on historical failures
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger("FailedProfilesLogger")


class FailedProfilesLogger:
    """Track and analyze failed simulation profiles."""

    def __init__(self, log_file: str = "failed_profiles.jsonl"):
        """
        Initialize the logger.

        Args:
            log_file: Path to the log file (JSONL format for append-only writes)
        """
        self.log_file = log_file
        self._failures: List[Dict] = []

    def log_failure(
        self,
        run_id: str,
        profile: Dict[str, Any],
        error_type: str,
        error_message: str,
        config_path: Optional[str] = None,
        stacktrace: Optional[str] = None,
    ):
        """
        Log a failed profile.

        Args:
            run_id: Identifier for the failed run
            profile: Profile parameters dictionary
            error_type: Type of error (e.g., "DomainError", "ValidationError")
            error_message: Error message
            config_path: Path to config file (if available)
            stacktrace: Full stack trace (if available)
        """
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "profile": profile.copy(),
            "error_type": error_type,
            "error_message": error_message,
            "config_path": config_path,
            "stacktrace": stacktrace,
        }

        # Append to log file immediately (append-only for thread safety)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(failure_entry) + "\n")
            logger.info(f"Logged failure: {run_id} - {error_type}")
        except IOError as e:
            logger.error(f"Failed to write to log file {self.log_file}: {e}")

        # Keep in memory for analysis
        self._failures.append(failure_entry)

    def load_failures(self) -> List[Dict]:
        """
        Load all failures from the log file.

        Returns:
            List of failure entries
        """
        failures = []
        if not os.path.exists(self.log_file):
            return failures

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            failures.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse log entry: {e}")
        except IOError as e:
            logger.error(f"Failed to read log file {self.log_file}: {e}")

        self._failures = failures
        return failures

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns to identify problematic parameter ranges.

        Returns:
            Dictionary with analysis results including:
            - error_type_counts: Count of each error type
            - parameter_ranges: Min/max values for each parameter in failures
            - outlier_parameters: Parameters that consistently appear in failures
        """
        if not self._failures:
            self.load_failures()

        if not self._failures:
            logger.warning("No failures to analyze")
            return {}

        # Count error types
        error_type_counts = {}
        for failure in self._failures:
            error_type = failure.get("error_type", "Unknown")
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        # Analyze parameter ranges in failures
        param_stats = {}
        for failure in self._failures:
            profile = failure.get("profile", {})
            for key, value in profile.items():
                if isinstance(value, (int, float)):
                    if key not in param_stats:
                        param_stats[key] = {"values": [], "min": value, "max": value}
                    param_stats[key]["values"].append(value)
                    param_stats[key]["min"] = min(param_stats[key]["min"], value)
                    param_stats[key]["max"] = max(param_stats[key]["max"], value)

        # Calculate statistics
        for key, stats in param_stats.items():
            values = np.array(stats["values"])
            stats["mean"] = float(np.mean(values))
            stats["std"] = float(np.std(values))
            stats["median"] = float(np.median(values))
            stats["count"] = len(values)
            del stats["values"]  # Remove raw values to save memory

        return {
            "total_failures": len(self._failures),
            "error_type_counts": error_type_counts,
            "parameter_stats": param_stats,
        }

    def print_failure_summary(self):
        """Print a summary of logged failures."""
        analysis = self.analyze_failure_patterns()

        if not analysis:
            print("No failures logged.")
            return

        print("\n" + "=" * 60)
        print("FAILED PROFILES SUMMARY")
        print("=" * 60)
        print(f"\nTotal failures: {analysis['total_failures']}")

        print("\nError types:")
        for error_type, count in analysis["error_type_counts"].items():
            print(f"  {error_type}: {count}")

        print("\nParameter ranges in failures:")
        for param, stats in analysis["parameter_stats"].items():
            print(
                f"  {param}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                f"mean={stats['mean']:.3f}, std={stats['std']:.3f} (n={stats['count']})"
            )

        print("\n" + "=" * 60)

    def suggest_validation_rules(self) -> List[str]:
        """
        Suggest validation rules based on failure patterns.

        Returns:
            List of suggested validation rule descriptions
        """
        analysis = self.analyze_failure_patterns()
        if not analysis:
            return []

        suggestions = []
        param_stats = analysis.get("parameter_stats", {})

        # Check for parameters with high variance in failures (might indicate sensitivity)
        for param, stats in param_stats.items():
            if stats["count"] >= 3:
                cv = stats["std"] / (stats["mean"] + 1e-6)
                if cv > 0.5:  # High coefficient of variation
                    suggestions.append(
                        f"Consider narrowing {param} range: "
                        f"currently [{stats['min']:.3f}, {stats['max']:.3f}] "
                        f"has high variance (CV={cv:.2f})"
                    )

        # Check for specific error patterns
        error_counts = analysis.get("error_type_counts", {})
        if "DomainError" in error_counts:
            suggestions.append(
                "DomainError detected: Check for negative values in probability calculations. "
                "Ensure all rates (μᵍ, ηᵍ) produce valid probabilities in [0, 1]."
            )

        return suggestions

    def get_recent_failures(self, n: int = 10) -> List[Dict]:
        """
        Get the most recent failure entries.

        Args:
            n: Number of recent failures to return

        Returns:
            List of recent failure entries
        """
        if not self._failures:
            self.load_failures()

        return self._failures[-n:] if self._failures else []


def scan_and_log_existing_failures(
    batch_folder: str,
    logger_instance: Optional[FailedProfilesLogger] = None,
) -> int:
    """
    Scan a batch folder for ERROR.json files and log them.

    Args:
        batch_folder: Path to batch output folder
        logger_instance: FailedProfilesLogger instance (creates new if None)

    Returns:
        Number of failures logged
    """
    if logger_instance is None:
        logger_instance = FailedProfilesLogger(
            log_file=os.path.join(batch_folder, "failed_profiles.jsonl")
        )

    count = 0
    for item in os.listdir(batch_folder):
        if not item.startswith("run_"):
            continue

        error_file = os.path.join(batch_folder, item, "ERROR.json")
        if not os.path.exists(error_file):
            continue

        try:
            with open(error_file, "r") as f:
                error_data = json.load(f)

            # Try to extract profile from config
            config_file = os.path.join(batch_folder, item, "config_auto_py.json")
            profile = {}
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Extract relevant parameters
                epi_params = config.get("epidemic_params", {})
                profile["r0_scale"] = epi_params.get("scale_β", 1.0)
                profile["alpha_scale"] = epi_params.get("αᵍ", [1.0])[0] / 0.1 if epi_params.get("αᵍ") else 1.0

                # Extract rates
                mu_g = epi_params.get("μᵍ", [0.2])[0]
                eta_g = epi_params.get("ηᵍ", [0.2])[0]
                profile["t_inf"] = 1.0 / mu_g if mu_g > 0 else 5.0
                profile["t_inc"] = 1.0 / eta_g if eta_g > 0 else 5.0

            logger_instance.log_failure(
                run_id=error_data.get("run_folder", item),
                profile=profile,
                error_type=error_data.get("error_type", "Unknown"),
                error_message=error_data.get("error_message", ""),
                stacktrace=error_data.get("stacktrace", ""),
            )
            count += 1

        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to process {error_file}: {e}")

    logger.info(f"Logged {count} failures from {batch_folder}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze failed simulation profiles"
    )
    parser.add_argument(
        "batch_folder",
        nargs="?",
        help="Path to batch output folder to scan for failures",
    )
    parser.add_argument(
        "--log-file",
        default="failed_profiles.jsonl",
        help="Path to failure log file (default: failed_profiles.jsonl)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan batch folder for ERROR.json files and log them",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze logged failures and print summary",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Suggest validation rules based on failure patterns",
    )

    args = parser.parse_args()

    failure_logger = FailedProfilesLogger(log_file=args.log_file)

    if args.scan and args.batch_folder:
        scan_and_log_existing_failures(args.batch_folder, failure_logger)

    if args.analyze or args.suggest:
        failure_logger.load_failures()
        failure_logger.print_failure_summary()

        if args.suggest:
            suggestions = failure_logger.suggest_validation_rules()
            if suggestions:
                print("\nSUGGESTED VALIDATION RULES:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
            else:
                print("\nNo validation rule suggestions (insufficient failure data)")
    elif not args.scan:
        print("No action specified. Use --scan, --analyze, or --suggest")
