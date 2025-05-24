# FRA-DiagSys: A Transformer Winding Fault Diagnosis System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Project Description

This repository contains the code implementation for **FRA-DiagSys**, a novel system for diagnosing transformer winding faults. Leveraging Frequency Response Analysis (FRA), this system employs Multilayer Perceptron (MLP) models to directly analyze FRA data and identify both the **type** and **degree** of various winding faults. The research addresses the limitations of traditional manual FRA curve interpretation by proposing a data-driven, automated approach validated on simulated transformer winding data under different configurations.

The core contribution lies in demonstrating the efficacy of MLP models applied directly to raw FRA data and proposing a **two-stage diagnostic strategy** that achieves high accuracy, including 100% accuracy for a specific 10-disc transformer winding model case studied in the paper.

## Highlights

*   Developed deep learning models utilizing raw Frequency Response Analysis (FRA) data for transformer winding fault diagnosis.
*   Validated the model's performance on three distinct laboratory-simulated winding datasets encompassing various fault types, degrees, winding designs (10-disc, 12-disc), and connection configurations (EE, CIW).
*   Proposed a **two-stage winding detection and diagnosis system (FRA-DiagSys)** demonstrating exceptional accuracy, achieving 100% for the 10-disc transformer model data.
*   Statistically demonstrated the impact of CIW and EE wiring configurations on fault diagnosis performance, informing the two-stage system design.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/FRA-DiagSys.git
    cd FRA-DiagSys
    ```
    *(Replace `YourUsername` with the actual GitHub username or organization)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Install the required packages using the `requirements.txt` file. If `requirements.txt` is not provided directly, you will need to install the key libraries mentioned in the paper and commonly used in this type of project:
    ```bash
    pip install torch torchvision torchaudio  # Install PyTorch
    pip install scikit-learn pandas numpy matplotlib # Install other libraries
    ```
    *(It's recommended to generate a `requirements.txt` from the environment used for development: `pip freeze > requirements.txt`)*

## Data

The study utilized three distinct datasets acquired from meticulously crafted laboratory winding models:

*   **Source:** Simulated 10-disc and 12-disc transformer windings.
*   **Fault Types Simulated:** Axial Displacement (AD), Disc Space Variation (DSV), Free Buckling (FB), and Short-Circuits (SC - for 12-disc). Various degrees of severity were simulated for each fault type.
*   **Connection Configurations:** End-to-End (EE) and Capacitive Inter-winding (CIW).
*   **Datasets:**
    *   `Group1`: 10-disc winding, EE connection (1425 samples)
    *   `Group2`: 10-disc winding, CIW connection (1425 samples)
    *   `Group3`: 12-disc winding, EE connection (2835 samples)

The datasets contain raw FRA amplitude response data as 1D vectors (2000 discrete points per sample).

*   Data files are expected to be located in a `data/` directory within the repository.
*   Scripts for loading and preprocessing the data are located in `src/data/`.
*   *(Note: The actual raw data files might not be included directly in the repository due to size or licensing constraints. The repository should include scripts to load or potentially generate synthetic data samples representative of the study.)*

## Models

The core models are based on the Multilayer Perceptron (MLP) architecture, designed to accept the 1D FRA data directly. Six different MLP architectures were designed and evaluated:

*   `FRA-Dialight`: A lightweight 3-layer MLP.
*   `FRA-Diagnoser`: A 5-layer MLP model.
*   `FRA-DiaL`: A 7-layer MLP model.
*   `FRA-DiaL-D`: `FRA-DiaL` with Dropout layers.
*   `FRA-DiaXL`: A 10-layer MLP model.
*   `FRA-DiaXL-D`: `FRA-DiaXL` with Dropout layers.

Model details (number of layers, width, parameter counts) can be found in the paper (specifically Figure 3 and Table 5).

*   Model implementations are located in `src/models/`.
*   Each model's output layer dimension is configurable based on the specific diagnostic task (Fault Type or Fault Degree).
*   Training utilized PyTorch with Adam optimizer, CrossEntropyLoss, and a learning rate of 0.0001.

## Usage

The repository provides scripts to train, evaluate, and utilize the models for transformer winding fault diagnosis.

1.  **Training a Model:**
    Use the training script to train a specific model on a specific dataset.
    ```bash
    python src/train.py --model <model_name> --dataset <dataset_name> --task <fault_type/fault_degree> --epochs <num_epochs> --batch_size <batch_size>
    ```
    *   `<model_name>`: e.g., `FRA-Diagnoser`, `FRA-Dialight`
    *   `<dataset_name>`: e.g., `Group1`, `Group2`, `Group3`
    *   `<task>`: `fault_type` for classifying fault types, `fault_degree` for classifying fault severity levels.
    *   Optional arguments for epochs, batch size, learning rate, etc.

2.  **Evaluating a Model:**
    Evaluate a trained model on a dataset using k-fold cross-validation as done in the paper.
    ```bash
    python src/evaluate.py --model <model_path> --dataset <dataset_name> --task <fault_type/fault_degree> --k_folds <k>
    ```
    *   `<model_path>`: Path to the trained model weights.
    *   `<k>`: Number of folds for cross-validation (e.g., 10 as in the paper).

3.  **Using the Two-Stage FRA-DiagSys:**
    Implement the two-stage diagnostic logic based on trained models for specific transformer types and connection configurations. An example script for the 10-disc transformer (EE then CIW) can be found in `src/diag_system.py`.
    ```bash
    python src/diag_system.py --ee_model <ee_model_path> --ciw_model <ciw_model_path> --input_data <path_to_new_fra_data>
    ```
    *   `<ee_model_path>`: Path to the model trained on EE data (e.g., for fault detection/type).
    *   `<ciw_model_path>`: Path to the model trained on CIW data (e.g., for fault type/degree).
    *   `<input_data>`: Path to the FRA data file(s) for diagnosis (should contain both EE and CIW measurements for the same winding).

## Performance

The models demonstrated high performance across various tasks and datasets using 10-fold cross-validation.

*   **Fault Type Diagnosis:**
    *   `FRA-Diagnoser` generally achieved the highest accuracy: 100% (Group1, EE 10-disc), 99.7% (Group2, CIW 10-disc), 99.8% (Group3, EE 12-disc).
    *   `FRA-Dialight` also performed very well for its size and speed: 99.2% (Group1), 98.2% (Group2), 99.8% (Group3).
    *   MLP models consistently outperformed traditional methods like SVM, RF, AdaBoost, XGBoost, and ELM on these datasets (refer to Figure 8 in the paper).

*   **Fault Degree Diagnosis:**
    *   `FRA-Dialight` showed the best performance on EE datasets (Group1, Group3) with accuracies of 90.2% and 92.5%, respectively.
    *   `FRA-Diagnoser` excelled on the CIW dataset (Group2) with 99.6% accuracy for fault degree diagnosis.

*   **FRA-DiagSys Two-Stage System:**
    *   Applying the two-stage strategy (e.g., using an EE-trained model followed by a CIW-trained model for the 10-disc transformer data) achieved **100% accuracy** for combined fault type and degree diagnosis on the dataset used in the study (Group1 & Group2 logic).

Refer to Tables 6, 7, and 8 and Figures 4, 6, and 8 in the original paper for detailed results and confusion matrices.

## FRA-DiagSys: The Two-Stage Diagnostic System

The research proposes a powerful two-stage model utilization strategy (illustrated in Figure 9 of the paper) which combines the strengths of models trained on FRA data from different connection methods (e.g., EE and CIW).

The rationale is that different connection configurations are sensitive to different types and locations of faults. By using a model trained on EE data (often good for detecting overall presence or certain fault types) as the first stage, and then using a model trained on CIW data (often better for distinguishing degrees or other fault types) as the second stage for samples identified as faulty, a more robust and accurate diagnosis can be achieved.

For the 10-disc transformer data used in this study, applying this serial approach yielded a 100% accurate quantitative diagnosis of fault types and degrees. The `src/diag_system.py` script provides an example implementation of this concept.

## Limitations

*   The models were primarily validated on simulated laboratory winding data. Further validation on a more diverse collection of real-world transformer FRA data is necessary.
*   Real-world transformers may have higher voltage levels and more discs, potentially requiring larger models or more complex architectures than the lightweight MLPs explored here.
*   The two-stage system's optimal configuration (which connection data for which stage, which model for which stage) is dependent on the specific transformer type and available data, requiring empirical determination.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(Make sure to include a LICENSE file in the repository)*

## Citation

If you use this code or the concepts from this research in your work, please cite the original paper:

```bibtex
@article{Wang_FRADiagSys,
  author = {Wang, Guohao},
  title = {FRA-DiagSys: A Transformer Winding Fault Diagnosis System for Identifying Fault Types and degrees Using Frequency Response Analysis},
  doi = {https://arxiv.org/pdf/2406.19623} 
