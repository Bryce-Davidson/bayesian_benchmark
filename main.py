# Import required libraries
import json
import os
import time
from pathlib import Path
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Literal, NamedTuple
import optuna


# Configuration classes to store settings
@dataclass
class SlurmSettings:
    """Settings for SLURM job submission"""

    memory_per_task: str = "2G"  # Amount of memory allocated per task
    cpus_per_task: int = 2  # Number of CPUs per task
    time_buffer: float = 1.2  # Add 20% buffer to estimated runtime
    jobs_per_batch: int = 50  # Number of simulations to run per script

    # Software versions
    julia_version: str = "1.5.3"
    r_version: str = "4.0.3"


@dataclass
class Paths:
    """Manages all file paths used in the project"""

    # Get the directory where this script is located
    base_dir: Path = Path(__file__).resolve().parent
    scripts_dir: Path = base_dir / "scripts"

    # Dictionary mapping analysis names to their script files
    analysis_scripts: dict = None

    def __post_init__(self):
        """Set up the script paths after initialization"""
        self.analysis_scripts = {
            "radial_search": self.scripts_dir / "radial_search.jl",
            "elastic_net": self.scripts_dir / "elastic_net.R",
            "leaps": self.scripts_dir / "leaps.R",
        }


@dataclass
class StudySettings:
    """Settings for the Optuna optimization study"""

    name: str = "variable_selection_benchmark"
    database_url: str = "sqlite:///benchmark_study.db"
    num_trials: int = 100
    direction: Literal["maximize", "minimize"] = "maximize"


# Class to store information about submitted jobs
class JobInfo(NamedTuple):
    """Stores information about a submitted SLURM job"""

    job_id: str  # SLURM job ID
    output_file: str  # Path to output file
    slurm_script: str  # Path to SLURM script


class BenchmarkRunner:
    """Main class for running benchmarks using SLURM"""

    def __init__(self):
        """Initialize the benchmark runner with default settings"""
        self.slurm = SlurmSettings()
        self.paths = Paths()
        self.study = StudySettings()

        # Create temporary directory for outputs
        self.output_dir = Path(tempfile.gettempdir()) / "benchmark_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def calculate_runtime(self, params: Dict) -> str:
        """
        Calculate estimated runtime for a job based on input parameters

        Args:
            params: Dictionary containing 'n' (sample size) and 'p' (number of variables)

        Returns:
            String in format "days-hours:minutes:00"
        """
        # Basic calculation: (n * p) / 1M gives base time in minutes
        base_minutes = (params["n"] * params["p"]) / 1000000

        # Apply job batch size and safety buffer
        total_minutes = (
            base_minutes * self.slurm.jobs_per_batch * self.slurm.time_buffer
        )

        # Convert to days, hours, minutes
        days = int(total_minutes // (24 * 60))
        hours = int((total_minutes % (24 * 60)) // 60)
        minutes = int(total_minutes % 60)

        return f"{days}-{hours:02d}:{minutes:02d}:00"

    def create_job_script(
        self, script_path: Path, params: Dict, output_file: str, time_limit: str
    ) -> str:
        """
        Create a SLURM submission script

        Args:
            script_path: Path to the analysis script
            params: Dictionary of parameters to pass to the script
            output_file: Where to save the results
            time_limit: Maximum runtime for the job

        Returns:
            Path to the created SLURM script
        """
        # Create temporary directory for SLURM scripts
        script_dir = Path(tempfile.gettempdir())
        slurm_script = script_dir / f"slurm_{script_path.stem}.sh"

        # Set up environment based on script type
        if script_path.suffix == ".jl":
            env_setup = f"module load julia/{self.slurm.julia_version}"
            run_command = f"julia {script_path}"
        elif script_path.suffix == ".R":
            env_setup = f"module load R/{self.slurm.r_version}"
            run_command = f"Rscript {script_path}"
        else:
            raise ValueError(f"Unsupported script type: {script_path}")

        # Create the SLURM script content
        script_content = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --output={output_file}.log
#SBATCH --error={output_file}.err
#SBATCH --job-name={script_path.stem}
#SBATCH --mem={self.slurm.memory_per_task}
#SBATCH --cpus-per-task={self.slurm.cpus_per_task}

# Load required software
{env_setup}

# Set up parallel processing environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export R_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the analysis script with parameters
{run_command} {params['n']} {params['p']} \
{params['small_big_beta_ratio']} {params['small_beta_proportion']} \
{params['covariance_structure']} {output_file}
"""
        # Write script to file
        slurm_script.write_text(script_content)
        return str(slurm_script)

    def submit_jobs(self, params: Dict) -> List[JobInfo]:
        """
        Submit all analysis jobs to SLURM

        Args:
            params: Dictionary of parameters for the analysis

        Returns:
            List of JobInfo objects for submitted jobs
        """
        time_limit = self.calculate_runtime(params)
        submitted_jobs = []

        # Submit each analysis script as a separate job
        for script_name, script_path in self.paths.analysis_scripts.items():
            output_file = str(self.output_dir / f"{script_name}_output.json")
            slurm_script = self.create_job_script(
                script_path, params, output_file, time_limit
            )

            # Submit the job using sbatch
            result = subprocess.run(
                ["sbatch", slurm_script], capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to submit job: {result.stderr}")

            job_id = result.stdout.strip().split()[-1]
            submitted_jobs.append(JobInfo(job_id, output_file, slurm_script))

        return submitted_jobs

    def wait_for_completion(self, jobs: List[JobInfo]) -> None:
        """
        Wait for all submitted jobs to complete

        Args:
            jobs: List of submitted JobInfo objects
        """
        while True:
            # Check for any jobs still in the queue
            pending_jobs = [
                job
                for job in jobs
                if subprocess.run(
                    ["squeue", "-j", job.job_id, "-h"], capture_output=True, text=True
                ).stdout.strip()
            ]

            if not pending_jobs:
                break

            # Wait 30 seconds before checking again
            time.sleep(30)

    def get_results(self, jobs: List[JobInfo]) -> List[Dict]:
        """
        Collect results from all completed jobs

        Args:
            jobs: List of completed JobInfo objects

        Returns:
            List of result dictionaries from each job
        """
        results = []
        for job in jobs:
            try:
                # Read and parse the JSON output file
                results.append(json.loads(Path(job.output_file).read_text()))
            except Exception as e:
                # If there's an error, try to get more information from log files
                error_logs = self._read_error_logs(job.output_file)
                raise RuntimeError(
                    f"Failed to get results from {job.output_file}. "
                    f"Error: {str(e)}. Logs: {error_logs}"
                )
        return results

    def _read_error_logs(self, output_file: str) -> str:
        """Read content from error and log files if they exist"""
        error_content = []
        for suffix in [".err", ".log"]:
            try:
                content = Path(output_file).with_suffix(suffix).read_text()
                error_content.append(f"{suffix} content:\n{content}")
            except Exception:
                continue
        return "\n".join(error_content)

    def cleanup(self, jobs: List[JobInfo]) -> None:
        """
        Clean up temporary files after jobs complete

        Args:
            jobs: List of completed JobInfo objects
        """
        for job in jobs:
            for file in [job.output_file, job.slurm_script]:
                try:
                    os.remove(file)
                    # Also remove log and error files
                    for suffix in [".log", ".err"]:
                        os.remove(f"{file}{suffix}")
                except FileNotFoundError:
                    continue


def main():
    """Main function to run the benchmark study"""
    runner = BenchmarkRunner()

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization

        This function:
        1. Generates parameters using Optuna
        2. Submits jobs with these parameters
        3. Waits for completion
        4. Collects and averages results

        Args:
            trial: Optuna trial object

        Returns:
            Average F1 score across all methods
        """
        # Generate parameters for this trial
        params = {
            "n": trial.suggest_int("n", 100, 10000),
            "p": trial.suggest_int("p", 10, 1000),
            "small_big_beta_ratio": trial.suggest_float(
                "small_big_beta_ratio", 0.01, 0.5
            ),
            "small_beta_proportion": trial.suggest_float(
                "small_beta_proportion", 0.1, 0.9
            ),
            "covariance_structure": trial.suggest_categorical(
                "covariance_structure",
                ["independent", "autoregressive", "compound_symmetry"],
            ),
        }

        # Submit and run jobs
        jobs = runner.submit_jobs(params)

        try:
            runner.wait_for_completion(jobs)
            results = runner.get_results(jobs)
            # Calculate average F1 score
            return sum(r["f1"] for r in results) / len(results)
        finally:
            # Always clean up, even if there's an error
            runner.cleanup(jobs)

    # Create and run the Optuna study
    study = optuna.create_study(
        study_name=runner.study.name,
        storage=runner.study.database_url,
        load_if_exists=True,
        direction=runner.study.direction,
    )

    study.optimize(objective, n_trials=runner.study.num_trials)


if __name__ == "__main__":
    main()
