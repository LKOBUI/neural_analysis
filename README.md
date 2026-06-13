# 📂 Neural Analysis Repository Structure

This project provides a modular framework for analyzing multi-electrode neural recording data, focusing on spike statistics, feature extraction, machine learning, and statistical testing.

- neural_analysis/
- │
- ├── data/                          # Data loading and preprocessing
- │   ├── init.py                 # Module exports (defines public API)
- │   ├── structures.py               # Core data structures and enums
- │   ├── loader.py                   # CSV data loading and validation
- │   └── preprocessor.py             # Data preprocessing and quality control
- │
- ├── features/                      # Feature extraction and transformation
- │   ├── init.py                 # Module exports
- │   └── transformers.py             # Feature extraction transformers
- │
- ├── analysis/                      # Statistical analysis and ML
- │   ├── init.py                 # Module exports
- │   ├── statistics.py               # Statistical analysis and FDR correction
- │   ├── classifiers.py              # ML classifiers and pipelines
- │   └── evaluation.py               # Model evaluation and metrics
- │
- ├── Plot_Functions.py              # Visualization utilities
- ├── main.py                        # Main entry point for analysis workflows
- ├── multielectrode_data.csv        # Example dataset (multi-electrode recordings)
- ├── LICENSE                        # License information
- └── README.md                      # Project documentation