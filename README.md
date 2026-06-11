# Perle

## Abstract

RNA sequence design aims to design RNA sequences that adopt a desired structure, but current methods often depend on thermodynamic folding engines, large training datasets, or fixed secondary-structure assumptions. Here, we developed Perle, a reinforcement learning-based framework for RNA sequence design that learns directly from user-defined target constraints. Perle uses a lightweight neural generator to produce candidate RNA sequences from random latent vectors and optimizes them with rewards based on base-pair satisfaction and sequence diversity. This target-specific strategy allows Perle to explore multiple valid sequences for the same structural target without offline pre-training or folding-engine calls during optimization. We evaluated Perle on small PDB-derived RNAs, CASP15 RNA targets, RNA-protein complex examples, and the QT45 polymerase ribozyme. Perle fully satisfied target base-pair constraints in the small PDB and CASP15 benchmarks, achieved average RMSD values below 5 Å for nine of ten small RNAs, and obtained the lowest average RMSD in four of six CASP15 targets. In the QT45 test, using an AlphaFold 3-derived target reduced the average false-pair count from more than ten to two positions per sequence. These results suggest that Perle provides a fast, flexible, and sequence-diverse framework for computational RNA design and downstream structural screening.

## Features

- Reinforcement learning-based RNA sequence generation
- Target-specific optimization without offline pre-training
- Support for base-pair constraints from dot-bracket structure strings
- Optional conserved nucleotide constraints
- Optional unpair constraints
- Sequence diversity reward to reduce simple target copying
- Optional AlphaFold 3 structural screening and RMSD calculation
- CSV summaries for training progress, runtime, generated sequences, RMSD, and constraint satisfaction

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Perle.git
cd Perle
```

Create and activate a Python environment:

```bash
conda create -n perle python=3.10 -y
conda activate perle
```

Install the required Python packages:

```bash
pip install torch numpy forgi
```

Depending on your repository layout, you may also need to make sure that any local helper modules imported by Perle, such as `structure.py`, are available in the same directory or on your `PYTHONPATH`.

For GPU acceleration, install the PyTorch build that matches your CUDA version and platform. CPU execution is possible, but training and candidate generation may be slower.

## Usage

A minimal example is shown below:

```python
from Perle import PerleConfig, train_and_generate

cfg = PerleConfig(
    name="1ANR",
    seq_len=29,
    target_str="GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
    struct_str=".((((....((((......)))).)))).",
    conserved_constraints={},
    unpair_constraints=[],
    num_seq=10,
    run_af3=True,
    target_cif="1ANR.cif",
)

summary = train_and_generate(cfg)
print(summary)
```

This example trains a Perle generator for the `1ANR` target, generates 10 candidate RNA sequences, optionally runs AlphaFold 3, and returns a summary dictionary containing generated sequences, RMSD values, and output CSV file paths.

## Main configuration options

`PerleConfig` controls the RNA design task and runtime behavior.

| Option | Description |
| --- | --- |
| `name` | Name of the design task. Used for default input/output directory and CSV names. |
| `seq_len` | Length of the RNA sequence. Must match the length of `target_str`. |
| `target_str` | Reference RNA sequence used for sequence similarity and diversity calculations. |
| `struct_str` | Dot-bracket structure string. If provided, Perle automatically converts it into base-pair constraints. |
| `pair_constraints` | Manual base-pair constraints as 1-based position pairs, for example `[[1, 15], [2, 14]]`. Used when `struct_str` is not provided. |
| `conserved_constraints` | Fixed nucleotide constraints as 1-based positions, for example `{3: "A", 10: "G"}`. |
| `unpair_constraints` | Position pairs that should not form Watson-Crick base pairs. Uses 1-based indexing. |
| `num_seq` | Number of final candidate sequences to return. |
| `rounds` | Number of reinforcement learning training rounds. |
| `batch` | Training batch size. |
| `lr` | Learning rate. |
| `diversity_beta` | Strength of the sequence-diversity penalty. Higher values discourage copying the target sequence. |
| `run_af3` | Whether to run AlphaFold 3 after sequence generation. |
| `target_cif` | Target CIF structure used for RMSD calculation when `run_af3=True`. |
| `input_dir` | Directory for generated AlphaFold 3 input JSON files and batch scripts. |
| `output_dir` | Directory for AlphaFold 3 outputs. |

Position-based constraints use **1-based indexing**. For example, position `1` means the first nucleotide in the sequence.

## Dot-bracket constraints

When `struct_str` is provided, Perle parses the dot-bracket string and creates base-pair constraints automatically.

Example:

```python
struct_str = ".((((....((((......)))).))))."
```

Supported bracket pairs include:

```text
()  []  {}  <>
```

Dots `.` are treated as unconstrained positions. The `&` character is ignored when parsing multi-chain style dot-bracket strings.

## Outputs

After running `train_and_generate(cfg)`, Perle writes several output files:

| File | Description |
| --- | --- |
| `time_<name>.csv` | Runtime information for model training, sequence generation, and optional AlphaFold 3 execution. |
| `train_summary_<name>.csv` | Training metrics across rounds, including reward and base-pairing score. |
| `detailed_summary_<name>.csv` | Generated sequences, RMSD values, sequence similarity, constraint satisfaction scores, dot-bracket comparisons, and AlphaFold 3-derived pair details. |

The returned Python dictionary contains:

```python
{
    "sequences": [...],
    "rmsd_values": [...],
    "time_csv": "time_<name>.csv",
    "train_summary_csv": "train_summary_<name>.csv",
    "detailed_summary_csv": "detailed_summary_<name>.csv",
}
```

## Running without AlphaFold 3

AlphaFold 3 is optional. To only train the generator and produce candidate sequences, set:

```python
cfg = PerleConfig(
    name="example",
    seq_len=29,
    target_str="GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
    struct_str=".((((....((((......)))).)))).",
    run_af3=False,
)
```

This is the recommended mode for quickly testing the package or running Perle on a machine where AlphaFold 3 is not installed.

## AlphaFold 3 configuration

When `run_af3=True`, Perle generates AlphaFold 3 input JSON files for the designed RNA sequences and runs them in batches. The default configuration assumes a specific Singularity-based AlphaFold 3 environment:

```python
af3_image = "/dssg/share/imgs/ai/alphafold/alphafold3-a800.sif"
af3_model_dir = "/dssg/share/data/alphafold3/models"
af3_db_dir = "/dssg/share/data/alphafold3/database"
```

Before running AlphaFold 3, update these paths to match your own system:

```python
cfg = PerleConfig(
    name="1ANR",
    seq_len=29,
    target_str="GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
    struct_str=".((((....((((......)))).)))).",
    run_af3=True,
    target_cif="1ANR.cif",
    af3_image="/path/to/your/alphafold3.sif",
    af3_model_dir="/path/to/your/alphafold3/models",
    af3_db_dir="/path/to/your/alphafold3/database",
)
```

If you run Perle on a different platform, such as another HPC cluster, a workstation, Docker, Apptainer, or a cloud environment, you may need to modify the AlphaFold 3 running process in the source code. In particular, check the function that writes the AlphaFold 3 batch script and adapt the container command, bind mounts, GPU flags, model path, database path, and output path to suit your own platform.

## Warm reminders

Please install all required packages before running Perle. Missing Python packages, missing helper modules, or incompatible PyTorch/CUDA versions may cause import errors or runtime failures.

Please make sure your sequence length, target sequence, and structure string are consistent. `seq_len`, `target_str`, and `struct_str` must describe the same RNA length.

Please check your constraint indexing carefully. Perle uses 1-based indexing for user-provided `pair_constraints`, `conserved_constraints`, and `unpair_constraints`.

Please set `run_af3=False` for a quick test or when AlphaFold 3 is not available.

Please modify the AlphaFold 3 execution script and paths when running on a platform different from the original development environment. The default AlphaFold 3 command may not work on your machine without changes.

Please provide a valid target CIF file through `target_cif` if you want Perle to calculate RMSD after AlphaFold 3 prediction.

## Example workflow

```python
from Perle import PerleConfig, train_and_generate

cfg = PerleConfig(
    name="my_rna_design",
    seq_len=29,
    target_str="GGCAGAUCUGAGCCUGGGAGCUCUCUGCC",
    struct_str=".((((....((((......)))).)))).",
    conserved_constraints={5: "A"},
    unpair_constraints=[],
    num_seq=20,
    rounds=200,
    run_af3=False,
)

summary = train_and_generate(cfg)

for i, seq in enumerate(summary["sequences"], start=1):
    print(f"Sequence {i}: {seq}")
```

## License

Add your license information here.

## Citation

If you use Perle in your research, please cite the corresponding paper or repository.
