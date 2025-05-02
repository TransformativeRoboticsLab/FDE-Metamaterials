# MetaTop for Fabrication-Directed Entanglement for Designing Chiral and Anisotropic Metamaterial Foams

MetaTop is a computational framework for topology optimization of mechanical metamaterials with extreme or targeted properties.

## Overview

This project implements optimization techniques for designing metamaterial structures with custom properties. Specifically it is configured for use with a viscous thread printing (VTP) fabrication process.

It uses finite element analysis, gradient-based optimization, and various constraints to design microstructures that achieve target mechanical behaviors.

## Key Features

- Topology optimization for mechanical metamaterial design
- Multiple optimization approaches (epigraph formulation, augmented Lagrangian)
- Customizable objective functions (Rayleigh quotient, Poisson's ratio, bulk modulus)
- Constraint handling (volume, eigenvector, trace constraints)
- FEniCS-based homogenization for property calculation
- Automatic differentiation with JAX for gradient computation
- Experiment tracking with Sacred/MongoDB

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/meta_design.git
   cd meta_design
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate meta_design
   ```

3. Set up environment variables (optional, required for database logging):
   ```bash
   # Create a .env file with the following content
   LOCAL_MONGO_URI="mongodb://localhost:27017"
   LOCAL_MONGO_DB_NAME="metatop"
   LOCAL_MONGO_EXP_NAME="extremal"
   ```

4. MongoDB setup (optional, for experiment tracking or verifying this work):
   - MongoDB is used for experiment tracking and result storage
   - The BSON file for this project is available at [here](https://bit.ly/TO-VTP)
   - Install MongoDB following [official instructions](https://www.mongodb.com/docs/manual/installation/)

## Usage

### Running Experiments

The project includes several ways to run experiments of different types:

```bash
# Run epigraph optimization
python experiments/epigraph.py

# Run Poisson's ratio optimization
python experiments/poissons_ratio.py 

# Run volume-constrained optimization
python experiments/volume_auglag.py

# Run matrix matching optimization (this work)
python experiments/matrix_matching.py

# Run an experiment 10 times
python experiments/run_experiment.py epigraph 10

# Run with specific parameters
python experiments/run_experiment.py volume_auglag 5 with E_min=0.1 nu=0.4

# Run with randomized parameters
python experiments/run_experiment.py matrix_matching 10 with basis_v="HSA" --randomize E_min nu

# Run indefinitely
python experiments/run_experiment.py epigraph -1 with norm_filter_radius=0.1
```

### Configuration Options

Experiments can be configured with various parameters using Sacred notation:

```bash
python experiments/volume_auglag.py with \
  E_max=1.0 E_min=0.1 nu=0.4 \
  start_beta=2 n_betas=7 \
  nelx=100 nely=100 \
  norm_filter_radius=0.1 \
  extremal_mode=1 basis_v="HSA"
```

Common parameters:
- `E_max/E_min`: Material property contrast
- `nu`: Poisson's ratio of base material
- `nelx/nely`: Grid resolution
- `start_beta/n_betas`: Projection sharpness parameters
- `extremal_mode`: Optimization mode (1=unimode, 2=bimode)
- `basis_v`: Basis vector for extremal eigenmode

### Visualizing Results

Results are automatically saved and can be visualized using the included notebooks:

```bash
jupyter notebook figure_creation/fig4.ipynb
```

## Project Structure

- `metatop/`: Core library components
  - `optimization/`: Optimization components and algorithms
  - `filters/`: Density filters for topology optimization
  - `mechanics/`: Homogenization and elasticity computations
- `experiments/`: Experiment scripts
- `tests/`: Test cases
- `figure_creation/`: Visualization and figure generation

## Testing

Run tests to verify functionality:

```bash
python tests/grad_check.py
python tests/homogenization.py
```

## Citation

If you use this code or the FDE methodology in your research, please cite the original paper: 

**TBD**

## License

**TBD**

