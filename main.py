# config.py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SlurmConfig:
    mem_per_task: str = "2G"
    cpus_per_task: int = 2
    base_runtime_factor: float = 1.2  # 20% buffer
    simulations_per_script: int = 50
    julia_version: str = "1.5.3"
    r_version: str = "4.0.3"


@dataclass
class PathConfig:
    base_dir: Path = Path(__file__).resolve().parent
    scripts_dir: Path = base_dir / "scripts"
    analysis_scripts: dict = None

    def __post_init__(self):
        self.analysis_scripts = {
            "radial_search": self.scripts_dir / "radial_search.jl",
            "elastic_net": self.scripts_dir / "elastic_net.R",
            "leaps": self.scripts_dir / "leaps.R",
        }


@dataclass
class StudyConfig:
    name: str = "variable_selection_benchmark"
    db_url: str = "sqlite:///benchmark_study.db"
    n_trials: int = 100
    direction: Literal["maximize", "minimize"] = "maximize"


import optuna
import subprocess
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, NamedTuple
import tempfile


class JobInfo(NamedTuple):
    job_id: str
    output_file: str
    slurm_script: str


class BenchmarkRunner:
    def __init__(
        self,
        slurm_config: SlurmConfig = SlurmConfig(),
        path_config: PathConfig = PathConfig(),
        study_config: StudyConfig = StudyConfig(),
    ):
        self.slurm_config = slurm_config
        self.path_config = path_config
        self.study_config = study_config
        self.output_dir = Path(tempfile.gettempdir()) / "benchmark_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def estimate_runtime(self, params: Dict[str, Any]) -> str:
        """Estimate runtime based on parameter settings."""
        base_time = (params["n"] * params["p"]) / 1000000
        total_minutes = (
            base_time
            * self.slurm_config.simulations_per_script
            * self.slurm_config.base_runtime_factor
        )

        days = int(total_minutes // (24 * 60))
        hours = int((total_minutes % (24 * 60)) // 60)
        minutes = int(total_minutes % 60)

        return f"{days}-{hours:02d}:{minutes:02d}:00"

    def create_slurm_script(
        self,
        script_path: Path,
        params: Dict[str, Any],
        output_file: str,
        time_limit: str,
    ) -> str:
        """Create a SLURM submission script."""
        script_dir = Path(tempfile.gettempdir())
        slurm_script = script_dir / f"slurm_{script_path.stem}.sh"

        # Determine the environment setup based on file type
        if script_path.suffix == ".jl":
            module_load = f"module load julia/{self.slurm_config.julia_version}"
            run_cmd = f"julia {script_path}"
        elif script_path.suffix == ".R":
            module_load = f"module load R/{self.slurm_config.r_version}"
            run_cmd = f"Rscript {script_path}"
        else:
            raise ValueError(f"Unknown script type: {script_path}")

        content = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --output={output_file}.log
#SBATCH --error={output_file}.err
#SBATCH --job-name={script_path.stem}
#SBATCH --mem={self.slurm_config.mem_per_task}
#SBATCH --cpus-per-task={self.slurm_config.cpus_per_task}

# Load required module
{module_load}

# Set up environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export R_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the analysis script
{run_cmd} {params['n']} {params['p']} \
{params['small_big_beta_ratio']} {params['small_beta_proportion']} \
{params['covariance_structure']} {output_file}
"""

        slurm_script.write_text(content)
        return str(slurm_script)

    def submit_jobs(self, params: Dict[str, Any]) -> List[JobInfo]:
        """Submit all jobs to SLURM and return their information."""
        time_limit = self.estimate_runtime(params)
        jobs = []

        for script_name, script_path in self.path_config.analysis_scripts.items():
            output_file = str(self.output_dir / f"{script_name}_output.json")
            slurm_script = self.create_slurm_script(
                script_path, params, output_file, time_limit
            )

            result = subprocess.run(
                ["sbatch", slurm_script], capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Job submission failed: {result.stderr}")

            job_id = result.stdout.strip().split()[-1]
            jobs.append(JobInfo(job_id, output_file, slurm_script))

        return jobs

    @staticmethod
    def wait_for_jobs(jobs: List[JobInfo]) -> None:
        """Wait for all jobs to complete."""
        while True:
            pending_jobs = [
                job
                for job in jobs
                if subprocess.run(
                    ["squeue", "-j", job.job_id, "-h"], capture_output=True, text=True
                ).stdout.strip()
            ]

            if not pending_jobs:
                break

            time.sleep(30)

    def collect_results(self, jobs: List[JobInfo]) -> List[Dict[str, float]]:
        """Collect results from all output files."""
        results = []
        for job in jobs:
            try:
                results.append(json.loads(Path(job.output_file).read_text()))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                error_content = self._get_error_content(job.output_file)
                raise RuntimeError(
                    f"Failed to collect results from {job.output_file}. "
                    f"Error: {str(e)}. Error/Log output: {error_content}"
                )
        return results

    @staticmethod
    def _get_error_content(output_file: str) -> str:
        """Get content from error and log files if they exist."""
        error_content = []
        for suffix in [".err", ".log"]:
            try:
                file_content = Path(output_file).with_suffix(suffix).read_text()
                error_content.append(f"{suffix} output:\n{file_content}")
            except Exception:
                continue
        return "\n".join(error_content)

    @staticmethod
    def cleanup_files(jobs: List[JobInfo]) -> None:
        """Clean up temporary files."""
        for job in jobs:
            for file in [job.output_file, job.slurm_script]:
                try:
                    os.remove(file)
                    for suffix in [".log", ".err"]:
                        os.remove(Path(file).with_suffix(suffix))
                except FileNotFoundError:
                    continue


def main():
    runner = BenchmarkRunner()

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function that runs all three scripts and combines their metrics."""
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

        jobs = runner.submit_jobs(params)

        try:
            runner.wait_for_jobs(jobs)
            results = runner.collect_results(jobs)
            return sum(r["f1"] for r in results) / len(results)
        finally:
            runner.cleanup_files(jobs)

    study = optuna.create_study(
        study_name=runner.study_config.name,
        storage=runner.study_config.db_url,
        load_if_exists=True,
        direction=runner.study_config.direction,
    )

    study.optimize(objective, n_trials=runner.study_config.n_trials)


if __name__ == "__main__":
    main()
