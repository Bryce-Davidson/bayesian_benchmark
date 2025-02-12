# SLURM Benchmark Runner

A Python framework for running and optimizing machine learning benchmarks on SLURM clusters using Optuna.

## Overview

This program automates the process of:

1. Running multiple analysis scripts (Julia and R) on a SLURM cluster
2. Optimizing hyperparameters using Optuna
3. Managing job submissions and collecting results
4. Handling cleanup and error reporting

## How It Works

### Core Components

#### 1. Configuration Classes

The program uses three main configuration classes:

```python
@dataclass
class SlurmSettings:
    """SLURM-specific settings"""
    memory_per_task: str = "2G"
    cpus_per_task: int = 2
    time_buffer: float = 1.2
    jobs_per_batch: int = 50
    julia_version: str = "1.5.3"
    r_version: str = "4.0.3"
```

- Controls SLURM job parameters
- Defines resource allocation
- Specifies software versions

```python
@dataclass
class Paths:
    """File path management"""
    base_dir: Path = Path(__file__).resolve().parent
    scripts_dir: Path = base_dir / "scripts"
    analysis_scripts: dict = None
```

- Manages file locations
- Maps analysis names to script files
- Handles path resolution

```python
@dataclass
class StudySettings:
    """Optuna study configuration"""
    name: str = "variable_selection_benchmark"
    database_url: str = "sqlite:///benchmark_study.db"
    num_trials: int = 100
    direction: Literal["maximize", "minimize"] = "maximize"
```

- Controls optimization study
- Defines database connection
- Sets optimization parameters

#### 2. BenchmarkRunner Class

The main class that orchestrates the entire process:

```python
class BenchmarkRunner:
    def __init__(self):
        self.slurm = SlurmSettings()
        self.paths = Paths()
        self.study = StudySettings()
        self.output_dir = Path(tempfile.gettempdir()) / "benchmark_outputs"
```

Key methods:

- `calculate_runtime()`: Estimates job duration
- `create_job_script()`: Generates SLURM scripts
- `submit_jobs()`: Handles job submission
- `wait_for_completion()`: Monitors job status
- `get_results()`: Collects and processes results

## Workflow Explanation

1. **Parameter Generation**

   - Optuna generates trial parameters
   - Parameters include sample size, variables, and model settings

2. **Job Creation**

   - Estimates required runtime based on parameters
   - Creates SLURM scripts with appropriate environment setup
   - Configures resource allocation

3. **Job Submission**

   - Submits jobs to SLURM queue
   - Tracks job IDs and output files
   - Monitors job status

4. **Result Collection**

   - Waits for all jobs to complete
   - Reads output JSON files
   - Calculates average performance metrics

5. **Cleanup**
   - Removes temporary files
   - Cleans up log files
   - Handles error cases

## Usage Examples

### Basic Usage

```python
# main.py
from benchmark_runner import BenchmarkRunner

def main():
    runner = BenchmarkRunner()
    study = optuna.create_study(
        study_name=runner.study.name,
        storage=runner.study.database_url,
        direction=runner.study.direction
    )
    study.optimize(objective, n_trials=runner.study.num_trials)

if __name__ == "__main__":
    main()
```

### Customizing Settings

```python
# Custom SLURM settings
runner = BenchmarkRunner()
runner.slurm.memory_per_task = "4G"
runner.slurm.cpus_per_task = 4
runner.slurm.time_buffer = 1.5

# Custom study settings
runner.study.num_trials = 200
runner.study.direction = "minimize"
```

### Adding New Analysis Scripts

1. Add script to the `scripts` directory
2. Update the Paths configuration:

```python
@dataclass
class Paths:
    def __post_init__(self):
        self.analysis_scripts = {
            "radial_search": self.scripts_dir / "radial_search.jl",
            "elastic_net": self.scripts_dir / "elastic_net.R",
            "leaps": self.scripts_dir / "leaps.R",
            "new_analysis": self.scripts_dir / "new_analysis.py"  # Add new script
        }
```

### Modifying Parameter Space

```python
def objective(trial: optuna.Trial) -> float:
    params = {
        "n": trial.suggest_int("n", 1000, 20000),  # Modified range
        "p": trial.suggest_int("p", 50, 2000),     # Modified range
        "small_big_beta_ratio": trial.suggest_float(
            "small_big_beta_ratio", 0.01, 0.3
        ),
        # Add new parameter
        "regularization": trial.suggest_float(
            "regularization", 0.0001, 1.0, log=True
        ),
    }
    # ... rest of the objective function
```

## File Structure

```
benchmark_study/
├── main.py
├── scripts/
│   ├── radial_search.jl
│   ├── elastic_net.R
│   └── leaps.R
├── README.md
└── benchmark_study.db
```

## Script Requirements

### Julia Scripts

- Must accept parameters in order: n, p, small_big_beta_ratio, small_beta_proportion, covariance_structure
- Must output JSON with an "f1" metric
- Example:

```julia
using JSON

# Get command line arguments
n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
# ... process data ...

# Output results
results = Dict("f1" => f1_score)
open(ARGS[6], "w") do f
    JSON.print(f, results)
end
```

### R Scripts

- Similar parameter requirements as Julia scripts
- Example:

```r
library(jsonlite)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
n <- as.integer(args[1])
p <- as.integer(args[2])
# ... process data ...

# Output results
results <- list(f1 = f1_score)
write_json(results, args[6])
```

## Error Handling

The program includes comprehensive error handling:

1. Job Submission Errors

```python
try:
    jobs = runner.submit_jobs(params)
except RuntimeError as e:
    print(f"Job submission failed: {e}")
```

2. Result Collection Errors

```python
try:
    results = runner.get_results(jobs)
except Exception as e:
    print(f"Failed to collect results: {e}")
    # Check error logs
    logs = runner._read_error_logs(job.output_file)
```

## Troubleshooting

Common issues and solutions:

1. **Job Time Limits**

   - Symptom: Jobs getting killed
   - Solution: Increase `time_buffer` in SlurmSettings

2. **Memory Issues**

   - Symptom: Out of memory errors
   - Solution: Increase `memory_per_task` in SlurmSettings

3. **Missing Results**
   - Symptom: Empty or missing output files
   - Solution: Check job logs in the output directory

## Contributing

To contribute:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## License

MIT License - Feel free to use and modify as needed.
