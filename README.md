# master thesis: Identifying the non-linear manifold of Kupffer cells using Rubens

This README file contains a short overview about the code and what each script does.

## Folder structure:
.
├── biolord
│   └── code
│       └── biolord_synthetic_data.ipynb
├── data_preprocessing
│   ├── code
│   │   └── perturbseq
│   │       ├── post_cellranger_processing
│   │       └── preprocessing
│   ├── data
│   │   ├── crispr_libraries
│   │   │   └── alk1_cd64_f480
│   │   ├── crispr_singlecell
│   │   │   ├── cellranger_output
│   │   │   └── libraries
│   │   └── singlecell
│   │       └── references
│   ├── dependencies.md
│   ├── develop.md
│   ├── pyproject.toml
│   ├── README.md
│   ├── requirements.txt
│   ├── software
│   │   └── cellranger-9.0.0.tar.gz
│   └── src
│       ├── crispyKC
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   └── utils
│       └── crispyKC.egg-info
│           ├── dependency_links.txt
│           ├── PKG-INFO
│           ├── requires.txt
│           ├── SOURCES.txt
│           └── top_level.txt
├── environment_biolord.yml
├── environment_crispykc.yml
├── environment_mefisto.yml
├── environment_Rubens.yml
├── MEFISTO
│   ├── test_synthetic_data.ipynb
│   └── top_genes_per_pc.csv
├── README.md
├── real_data
│   ├── ABL005-007_kupffer_fully_filtered.pkl
│   └── wild_type_cells.pkl
├── Rubens
│   ├── code
│   │   ├── biased_models.ipynb
│   │   ├── comparison_best_models.ipynb
│   │   ├── elitism_generation.py
│   │   ├── general_functions
│   │   │   ├── model_generation_functions.py
│   │   │   ├── model_visualization_functions.py
│   │   │   └── __pycache__
│   │   ├── genetic_algorithm_full_crossover_generation.py
│   │   ├── genetic_algorithm_partial_crossover_generation.py
│   │   ├── initial_exploration.ipynb
│   │   ├── pruning.ipynb
│   │   ├── random_model_generation.py
│   │   ├── synthetic_dataset_testing.ipynb
│   │   ├── visualization_elitism.ipynb
│   │   └── visualization_genetic_algorithm.ipynb
│   └── __pycache__
│       ├── latenta_modeling_visualization.cpython-312.pyc
│       └── robin_new_functions.cpython-312.pyc
├── synthetic_data
│   ├── dependencies_complex_synthetic_data.pkl
│   ├── dependencies_simple_synthetic_data.pkl
│   ├── no_dependencies_complex_synthetic_data.pkl
│   └── no_dependencies_simple_synthetic_data.pkl
└── synthetic_data_generation_script
    └── synthetic_data_generation.ipynb

## installation

git clone https://github.com/guilliottslab/crispyKC.git
