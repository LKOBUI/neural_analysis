# Neural Analysis with Pytorch and CUDA Methodology

This project provides a modular framework for analyzing multi-electrode neural recording data, focusing on spike statistics, feature extraction, machine learning, and statistical testing.


## Table of Contents

- [A). DigitMinst](#a-digitminst)
  - [1. General Project Objectives](#1-general-project-objectives)
  - [2. Detailed Summary of Jupyter Notebooks](#2-detailed-summary-of-jupyter-notebooks)
    - [`DigitsMinst.ipynb` & `LearnMist.ipynb`](#digitsminstipynb-learnmistipynb)
    - [`DigitMinstAdditionals.ipynb`](#digitminstadditionalsipynb)
    - [`DigitMinst_AnalysisOnNeuralAndCnnArch.ipynb`](#digitminst_analysisonneuralandcnnarchipynb)
    - [`Digit_Minst_completed.ipynb` & `MinstProjectAnalysis.ipynb`](#digit_minst_completedipynb-minstprojectanalysisipynb)
  - [3. Mathematical Formulations Explained](#3-mathematical-formulations-explained)
  - [4. Deep Learning Model Architectures & Selection Reasoning](#4-deep-learning-model-architectures-selection-reasoning)
    - [Architecture A: Dense Multilayer Perceptron (ANN)](#architecture-a-dense-multilayer-perceptron-ann)
    - [Architecture B: Convolutional Neural Network (CNN)](#architecture-b-convolutional-neural-network-cnn)
- [B). Fashion-MNIST](#b-fashion-mnist)
  - [1. Project Objectives & Detailed Notebook Summary](#1-project-objectives-detailed-notebook-summary)
    - [`MinstKeras.ipynb`](#minstkerasipynb)
  - [2. Mathematical Formulations Explained](#2-mathematical-formulations-explained)
  - [3. Model Architecture & Selection Details](#3-model-architecture-selection-details)
- [C. NeuralTutorials:](#c-neuraltutorials)
  - [1. Detailed Notebook Summaries & Core Objectives](#1-detailed-notebook-summaries-core-objectives)
    - [`ANN-tutorials.ipynb` & `Keras_Learn.ipynb`](#ann-tutorialsipynb-keras_learnipynb)
    - [`Gradient_Problem_and_solutions.ipynb` & `AdditionalOnGrediantProblem.ipynb`](#gradient_problem_and_solutionsipynb-additionalongrediantproblemipynb)
    - [`BatchNormWideAndDeepNetwork.ipynb`](#batchnormwideanddeepnetworkipynb)
    - [`Convolutions.ipynb`](#convolutionsipynb)
    - [`Dropout In neural network.ipynb` & `Regularizations.ipynb`](#dropout-in-neural-networkipynb-regularizationsipynb)
    - [`MomentumAndFasterOptimizations.ipynb`](#momentumandfasteroptimizationsipynb)
    - [`Non-sequential neural network.ipynb`](#non-sequential-neural-networkipynb)
    - [`WeightAndBaisInit.ipynb`](#weightandbaisinitipynb)
    - [`VisualEffectOnDifferentVaribles.ipynb`](#visualeffectondifferentvariblesipynb)
  - [2. Mathematical Formulations Explained](#2-mathematical-formulations-explained-1)
  - [3. Deep Learning Model Architectures & Selection Reasoning](#3-deep-learning-model-architectures-selection-reasoning)
    - [Architecture 1: Wide & Deep Neural Functional Networks](#architecture-1-wide-deep-neural-functional-networks)
    - [Architecture 2: Advanced Deep MLP with Adaptive Optimization](#architecture-2-advanced-deep-mlp-with-adaptive-optimization)
    - [Architecture 3: Multi-Layer 2D Convolutional Networks (CNNs)](#architecture-3-multi-layer-2d-convolutional-networks-cnns)
- [D). TensorFlowTutorials](#d-tensorflowtutorials)
  - [1. Detailed Notebook Summaries & Core Objectives](#1-detailed-notebook-summaries-core-objectives-1)
    - [`TensorBasic.ipynb`](#tensorbasicipynb)
    - [`Reusing Pretrained Layers.ipynb`](#reusing-pretrained-layersipynb)
  - [2. Mathematical Formulations Explained](#2-mathematical-formulations-explained-2)
  - [3. Deep Learning Model Architectures & Selection Reasoning](#3-deep-learning-model-architectures-selection-reasoning-1)
    - [Architecture 1: Core Mathematical Multi-Tensor Graph (Low-Level Baseline)](#architecture-1-core-mathematical-multi-tensor-graph-low-level-baseline)
    - [Architecture 2: Pretrained Feature Extractor with Custom Head (Transfer Learning Setup)](#architecture-2-pretrained-feature-extractor-with-custom-head-transfer-learning-setup)
- [E). PyTorch and CUDA — Digit MNIST Implementation](#e-pytorch-and-cuda-digit-mnist-implementation)
  - [1. Project Objectives](#1-project-objectives)
  - [2. Data Preparation and Tensor Shapes](#2-data-preparation-and-tensor-shapes)
  - [3. PyTorch Model Architecture](#3-pytorch-model-architecture)
    - [Architecture A: Dense MLP — `PytorchTensorDigitMist`](#architecture-a-dense-mlp-pytorchtensordigitmist)
    - [Architecture B: LoRA-enabled MLP — `PytorchTensorDigitMistLoRA`](#architecture-b-lora-enabled-mlp-pytorchtensordigitmistlora)
    - [Model factory — `TensorModels`](#model-factory-tensormodels)
  - [4. Training, CUDA, and Numerical Utilities](#4-training-cuda-and-numerical-utilities)
  - [5. Evaluation and Experiment Tracking](#5-evaluation-and-experiment-tracking)
  - [6. Application and Deployment Workflow](#6-application-and-deployment-workflow)
  - [7. Purpose of the PyTorch Addition](#7-purpose-of-the-pytorch-addition)

# A). DigitMinst 
Based on the repository architecture of the LKOBUI/neural_analysis - DigitMinst subfolder, here is the detailed technical breakdown covering the objectives, notebook summaries, mathematical formulations, and model architectures.
## 1. General Project Objectives

*   **Core Aim**: Implement, benchmark, and optimize deep learning algorithms for handwritten digit recognition using the iconic **MNIST dataset**.
*   **Structural Intent**: Bridge the performance gap between classic Artificial Neural Networks (**ANNs/Dense Layers**) and spatial-aware Convolutional Neural Networks (**CNNs**).
*   **Workflow Validation**: Assess structural performance variances introduced by preprocessing modifications, variable batch setups, and optimization algorithms.
## 2. Detailed Summary of Jupyter Notebooks

The directory splits its experimentation into specialized tracking documents:

### `DigitsMinst.ipynb` & `LearnMist.ipynb`

*   **Summary**: Serve as foundational baselines. They load raw arrays via `keras.datasets.mnist`, perform min-max normalization, reshape 2D matrix blocks into flat arrays, and train basic Multilayer Perceptrons (MLPs).
*   **Key Tasks**: Establishing minimum acceptable accuracy thresholds and tracking training time metrics on non-accelerated CPU/GPU environments.
### `DigitMinstAdditionals.ipynb`

*   **Summary**: Focuses on auxiliary pipeline components. It tests edge-case modifications such as tuning batch sizes, comparing categorical cross-entropy versus sparse categorical cross-entropy, and analyzing misclassified digits via confusion matrices.

### `DigitMinst_AnalysisOnNeuralAndCnnArch.ipynb`

*   **Summary**: The core comparative analysis file. It sets up a head-to-head structural performance evaluation between deep dense neural arrays and multi-layer convolutional structures.
*   **Key Tasks**: Demonstrating why flat dense networks struggle with translation-invariance compared to feature-pooling CNN architectures.
### `Digit_Minst_completed.ipynb` & `MinstProjectAnalysis.ipynb`

*   **Summary**: Production-ready, cleaned scripts summarizing the ultimate selected architectures. They feature optimized hyperparameters, validation curves mapping training vs. testing losses, and serialized model saving modules.

---

## 3. Mathematical Formulations Explained

The pipeline implements three core mathematical routines to calculate gradients and convert hidden outputs into valid probability vectors:

*   **Min-Max Feature Scaling (Normalization)**:
    Transforms raw integer pixel scales $[0, 255]$ down to floating-point values inside a standard $[0, 1]$ range to ensure smooth gradient descents:

$$X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} = \frac{X - 0}{255 - 0} = \frac{X}{255}$$
*   **Softmax Activation Function (Output Mapping)**:
    Converts the final unnormalized layer output vectors (logits, $z_i$) of the network into a normalized categorical probability distribution over the 10 digit classes:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \dots, K \quad (K = 10)$$

*   **Categorical Cross-Entropy Loss ($L$)**:
    The loss equation minimized during backpropagation to penalize deviations between predicted probabilities ($\hat{y}_i$) and actual one-hot encoded targets ($y_i$):

$$L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
## 4. Deep Learning Model Architectures & Selection Reasoning

The notebooks utilize two core model paradigms based on target structural goals:

### Architecture A: Dense Multilayer Perceptron (ANN)

*   **Structure**: `Input (Flatten, 784 nodes)` $\rightarrow$ `Dense (Hidden 1, 128/512 nodes + ReLU)` $\rightarrow$ `Dense (Hidden 2, Optional)` $\rightarrow$ `Dense (Output, 10 nodes + Softmax)`
*   **Reasoning for Selection**: Provides a lightweight, high-speed baseline computational reference. However, it lacks spatial awareness because flattening a $28 \times 28$ pixel block into a 784-element linear vector destroys local 2D structural patterns (edges, loops, and corners).
### Architecture B: Convolutional Neural Network (CNN)

*   **Structure**: `Input (Tensor shape: 28, 28, 1)` $\rightarrow$ `Conv2D (32 filters, 3x3 kernel + ReLU)` $\rightarrow$ `MaxPooling2D (2x2 pool)` $\rightarrow$ `Conv2D (64 filters, 3x3)` $\rightarrow$ `Flatten` $\rightarrow$ `Dense (Hidden)` $\rightarrow$ `Dense (Output, 10 nodes + Softmax)`
*   **Reasoning for Selection**: Essential for maximizing classification accuracy.
    *   **Conv2D kernels** apply local weight sharing to extract translation-invariant visual features regardless of where the digit is drawn in the frame.
    *   **MaxPooling layers** reduce spatial dimensions, slashing parameter counts and preventing overfitting while retaining the most dominant structural activations.

The convolutional neural network architecture, built with tf.keras.models.Sequential, utilizes a two-layer convolutional structure (32 and 64 filters) followed by max-pooling, flattening, and dense layers for optimal MNIST digit classification. Learning rate variations significantly impact performance, where optimal rates (0.001–0.01) yield smooth, exponential loss decay, whereas excessively high or low rates result in oscillation or training stagnation, respectively.

# B). Fashion-MNIST
Build, compile, and evaluate deep learning classification models using Keras to categorize complex grayscale clothing images.
## 1. Project Objectives & Detailed Notebook Summary

### `MinstKeras.ipynb`

*   **Core Objective**: Implement a robust image classification pipeline using Keras to identify ten distinct categories of clothing items (such as shirts, trousers, and sneakers) from the **Fashion-MNIST dataset**.
*   **Pipeline Workflow**:
    *   **Data Acquisition**: Imports the clothing array natively via `tf.keras.datasets.fashion_mnist`.
    *   **Preprocessing Matrix Transformation**: Normalizes 8-bit integer pixel matrices into uniform floating-point ranges.
    *   **Feature Flattening**: Converts 2D structural arrays into 1D feature strings for standard linear vector mapping.
    *   **Feature Flattening**: Converts 2D structural arrays into 1D feature strings for standard linear vector mapping.
    *   **Model Compilation & Training**: Configures optimal losses and backpropagation trackers over successive generation epochs.
    *   **Validation Benchmarking**: Compares training metrics against validation arrays to check for predictive overfitting.

## 2. Mathematical Formulations Explained

The notebook relies on standard mathematical layers to smoothly scale values, calculate error gradients, and handle classification:

*   **Pixel Scaling Normalization**:
    Prevents exploding gradients and standardizes feature impacts by dividing every pixel input by its maximum possible numeric value:
$$X_{\text{scaled}} = \frac{X - \text{Min}}{\text{Max} - \text{Min}} = \frac{X - 0}{255 - 0} = \frac{X}{255}$$

*   **Sparse Categorical Cross-Entropy Loss ($L$)**:
    Measures error without requiring upfront one-hot vector encoding. It computes log-likelihood penalties directly using integer class labels ($y$) against predicted probability distribution profiles ($\hat{y}$):

$$L = -\log(\hat{y}_y)$$
*   **Rectified Linear Unit (ReLU) Activation**:
    Introduces non-linearity to let the network learn intricate design patterns, outputting zero for any negative input:

$$f(x) = \max(0, x)$$

---

## 3. Model Architecture & Selection Details

**Implemented Model**: Sequential Dense Multilayer Perceptron (MLP)

*   **Layer Topology**:
    1.  **Input Layer (`Flatten`)**: Takes a $28 \times 28$ matrix and flattens it into a 784-element input vector.
    2.  **Hidden Layer (`Dense`)**: A fully connected hidden layer utilizing `ReLU` to capture intermediate clothing item design shapes.
    3.  **Output Layer (`Dense`)**: A final 10-node array featuring a `Softmax` output function to yield separate probability metrics for each clothing type.

**Selection Reasoning by Region & Design**:

*   **Handling Increased Spatial Variations**: Fashion-MNIST features complex outlines, folds, and high texture variations compared to standard handwritten digits. Using an MLP with a deep hidden dense layer acts as a solid computational baseline to test whether non-spatial linear connections can distinguish varied silhouettes.
*   **Resource and Speed Efficiency**: Dense structures train quickly without requiring heavy GPU compute infrastructures. This makes them ideal for establishing a fast benchmark before scaling up to more resource-heavy architectures.

# C. NeuralTutorials:
Explore deep learning mechanics, architectural configurations, optimization variants, and stability algorithms using Keras and TensorFlow.
## 1. Detailed Notebook Summaries & Core Objectives

The `NeuralTutorials` subfolder provides a deep-dive educational path covering specific challenges encountered when building and scaling neural structures:

### `ANN-tutorials.ipynb` & `Keras_Learn.ipynb`

*   **Objective**: Establish foundational knowledge for setting up standard fully-connected feedforward pipelines.
*   **Summary**: They walk through sequential layer building, structural dataset loading, basic cross-entropy compiling, and parameter training mechanics.

### `Gradient_Problem_and_solutions.ipynb` & `AdditionalOnGrediantProblem.ipynb`

*   **Objective**: Diagnose, illustrate, and fix foundational deep network instabilities—specifically vanishing and exploding gradients.
*   **Summary**: They demonstrate what happens when gradients diminish exponentially as they backpropagate through deep layers, rendering early weights stagnant. Solutions like structural activation matching are tested.

### `BatchNormWideAndDeepNetwork.ipynb`

*   **Objective**: Combine advanced functional layouts with dynamic stabilization techniques.
*   **Summary**: Implements a "Wide & Deep" architecture that feeds raw continuous features directly to the output layer (Wide) while processing complex non-linear interactions via deep stacks (Deep). It inserts Batch Normalization layers to continuously center and scale inner layer outputs.
### `Convolutions.ipynb`

*   **Objective**: Shift from unstructured 1D vector processing to spatial 2D grid processing.
*   **Summary**: Details the structural execution of localized feature extraction kernels, sliding windows, and downsampling techniques to process imagery natively without flat pixel distortions.

### `Dropout In neural network.ipynb` & `Regularizations.ipynb`

*   **Objective**: Introduce architectural constraints to eliminate data memorization (overfitting).
*   **Summary**: `Regularizations.ipynb` implements weight penalty matrices during mathematical loss evaluation. `Dropout In neural network.ipynb` forces architectural variance by randomly severing connections between hidden neurons at each training step.
### `MomentumAndFasterOptimizations.ipynb`

*   **Objective**: Benchmark advanced mathematical engines that accelerate weight convergence.
*   **Summary**: Moves beyond simple Stochastic Gradient Descent (SGD) to compare adaptive, history-tracking weight updaters like Momentum, Nesterov Accelerated Gradient, RMSprop, and Adam.

### `Non-sequential neural network.ipynb`

*   **Objective**: Break free from rigid linear structural layering.
*   **Summary**: Utilizes the Keras Functional API to construct complex Directed Acyclic Graph (DAG) structures, multi-input structures, or multi-output prediction heads.
### `WeightAndBaisInit.ipynb`

*   **Objective**: Prevent early signal saturation right at initialization.
*   **Summary**: Compares traditional random weight fills against distribution-aware initializers like Xavier (Glorot) and He initialization.

### `VisualEffectOnDifferentVaribles.ipynb`

*   **Objective**: Provide empirical visual validation of modeling decisions.
*   **Summary**: Tracks, maps, and plots structural validation metrics visually to illustrate how adjustments to training configurations immediately alter loss trajectories.
## 2. Mathematical Formulations Explained

The notebooks leverage specific mathematical formulas to maintain training stability, balance losses, and handle optimizer updates:

*   **Batch Normalization (Transformation)**:
    Stabilizes the training of deep architectures by normalizing the activation matrix $x$ across a mini-batch:

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
    *   **Explanation**: Scaled adjustments use mean ($\mu_B$) and variance ($\sigma_B^2$). Scale ($\gamma$) and shift ($\beta$) parameters allow the network to restore the optimal distribution for learning.

*   **He Initialization (Variance Scaling)**:
    Maintains balanced signal variances across deep structures using ReLU activators:

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

    *   **Explanation**: Weights are drawn from a normal distribution with a mean of 0 and a variance scaled to the number of incoming connection nodes ($n_{\text{in}}$), keeping early layer signals within healthy bounds.

*   **$L_2$ Weight Regularization (Ridge Penalty)**:
    Constrains neural weights by appending a penalty term directly to the core objective loss function ($E_0$):
$$E = E_0 + \frac{\lambda}{2m} \sum w^2$$

    *   **Explanation**: Squares and sums all internal connection weights ($w$), scaling them by a regularizer factor ($\lambda$). This penalizes large weights, discouraging the network from relying heavily on any single feature signal.

*   **Momentum Optimization (Velocity Accumulation)**:
    Accelerates gradient updates by tracking past directional momentum:

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_{\theta} J(\theta)$$

$$\theta = \theta - \alpha v_t$$
    *   **Explanation**: Accumulates a velocity vector ($v_t$) scaled by momentum decay ($\beta$) based on current cost gradients ($\nabla_{\theta} J$). This helps the model push through flat loss plates and shallow local minima.

---

## 3. Deep Learning Model Architectures & Selection Reasoning

### Architecture 1: Wide & Deep Neural Functional Networks

*   **Details**: Combines a linear pathway bypassing hidden layers with a parallel Multi-Layer Perceptron (MLP) deep stack that uses `ReLU`, `BatchNormalization`, and `Dropout`. The outputs merge into a final dense projection head.
*   **Reasoning by Region/Intent**: Perfect for tabular datasets or structural data containing both strict rules and non-linear patterns. The "Wide" component memorizes sparse, explicit feature connections, while the "Deep" component generalizes across complex interactions.
### Architecture 2: Advanced Deep MLP with Adaptive Optimization

*   **Details**: A deep linear stack of fully connected dense layers integrated with custom parameter initializers (`kernel_initializer='he_normal'`) and optimized via adaptive momentum (`optimizer='adam'`).
*   **Reasoning by Region/Intent**: Used as a benchmark platform to evaluate how initialization adjustments and adaptive learning rates mitigate the vanishing gradient problem in deep feedforward structures.

### Architecture 3: Multi-Layer 2D Convolutional Networks (CNNs)

*   **Details**: Employs spatial feature mapping via `Conv2D` kernel tensors, spatial downsampling via `MaxPooling2D`, and feature extraction blocks before funneling signals into dense decision nodes.
*   **Reasoning by Region/Intent**: Essential for spatial datasets. Unlike standard MLPs that flatten inputs and discard local structural relationships, CNNs use local receptive fields and shared weights to capture invariant edge, shape, and boundary patterns.

# D). TensorFlowTutorials
Understand low-level TensorFLow core mechanics and implement advanced Transfer Learning strategies by reusing pretrained neural layers.
## 1. Detailed Notebook Summaries & Core Objectives

The `TensorFlowTutorials` directory focuses on moving from core data structures to efficient model reuse workflows:

### `TensorBasic.ipynb`

*   **Objective**: Master foundational multidimensional array mechanics, computational graphs, and low-level API operations in TensorFlow.
*   **Summary**: This notebook explores core data structures by creating, reshaping, and slicing tensors. It covers mathematical matrix operations, broadcasting rules, tracking gradients via `tf.GradientTape`, and managing memory efficiently during graph execution.
### `Reusing Pretrained Layers.ipynb`

*   **Objective**: Leverage Transfer Learning principles to build high-performance image or sequence classifiers using minimal data and compute resources.
*   **Summary**: This tutorial walks through splitting complex model layers to separate a pretrained base from its classification head. It covers freezing lower-level parameter weights to preserve general feature detectors (like edges and curves), appending fresh custom dense structures, and performing target fine-tuning with small learning rates.

---

## 2. Mathematical Formulations Explained

The notebooks utilize fundamental optimization and array formulas to update layer weights and evaluate errors:
*   **Matrix Multiplication Dot Product (Linear Projections)**:
    Forms the foundational feedforward signal path across dense nodes:

$$Z = X \cdot W + b$$

    *   **Explanation**: The input feature matrix ($X$) undergoes a dot product multiplication with the weight matrix ($W$) before adding the bias vector ($b$), projecting input variables into the next hidden layer space.

*   **Automatic Differentiation via Computational Tape**:
    Calculates exact analytical gradients for model optimization:

$$\nabla_{\theta} J(\theta) = \left[ \frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \dots, \frac{\partial J}{\partial \theta_n} \right]$$

    *   **Explanation**: TensorFlow tracks forward operations on a virtual "tape". It applies the calculus chain rule backward to evaluate partial derivatives of the cost function ($J$) with respect to every single active weight parameter ($\theta$).
*   **Layer Output Variance Scaling (Transfer Adjustments)**:
    Maintains signal stability across structural model splits:

$$W_{\text{new\_init}} \sim N \left( 0, \sqrt{\frac{2}{n_{\text{in}}}} \right)$$

    *   **Explanation**: When appending a brand new classification head next to a frozen, pretrained structure, the new weights must scale their variance to match the incoming feature nodes ($n_{\text{in}}$) to avoid destabilizing earlier layers.
## 3. Deep Learning Model Architectures & Selection Reasoning

### Architecture 1: Core Mathematical Multi-Tensor Graph (Low-Level Baseline)

*   **Details**: Built using low-level `tf.Variable` arrays and manual matrix dot products, mapped directly through custom optimization loops rather than the standard `Keras` high-level abstraction layers.
*   **Reasoning by Region/Intent**: Perfect for learning low-level framework mechanics. It helps developers understand how mathematical tensors interact behind the scenes before moving to high-level automation wrappers.

### Architecture 2: Pretrained Feature Extractor with Custom Head (Transfer Learning Setup)

*   **Details**: Utilizes a deep, pretrained backbone architecture (such as `MobileNetV2` or `ResNet50` trained on ImageNet) with its top layers removed (`include_top=False`). A new global pooling layer and a custom dense classification head are added on top.
*   **Reasoning by Region/Intent**: Crucial for training complex visual or sequence systems when data is limited. Reusing lower layers saves massive amounts of compute time, as the network can repurpose pre-learned feature patterns instead of training from scratch.


---

# E). PyTorch and CUDA — Digit MNIST Implementation

This section documents the PyTorch implementation added to the project. The earlier sections above are retained as they were; this section supplements the existing TensorFlow/Keras material with a PyTorch-based MNIST workflow and GPU-aware training utilities.

## 1. Project Objectives

- Implement handwritten-digit classification using **PyTorch** and the **MNIST** dataset.
- Define neural-network architectures explicitly with `torch.nn.Module`, making the layer structure and parameter initialization visible.
- Support execution on CUDA-capable NVIDIA GPUs when the installed PyTorch build and hardware support CUDA, with CPU fallback where configured.
- Compare a standard dense neural network with a Low-Rank Adaptation (**LoRA**) variant.
- Track experiment configuration, evaluation metrics, selected models, and model artifacts through the project's experiment/deployment workflow.

## 2. Data Preparation and Tensor Shapes

The implementation uses the MNIST dataset of grayscale handwritten digits. Each image is \(28 \times 28\) pixels and belongs to one of ten classes, from 0 through 9.

The data pipeline uses `torchvision.datasets.MNIST`, transforms, and `DataLoader` batches. When normalization is enabled, the transform uses the commonly used MNIST statistics:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

A batch of 64 images has shape:

- Images: `torch.Size([64, 1, 28, 28])`
- Labels: `torch.Size([64])`

The image tensor includes a channel dimension because the images are grayscale. The dense network flattens each image from \(1 \times 28 \times 28\) into a vector of 784 input features.

## 3. PyTorch Model Architecture

### Architecture A: Dense MLP — `PytorchTensorDigitMist`

The dense model is implemented as a class derived from `torch.nn.Module`. Its main stages are:

```text
MNIST image (1 × 28 × 28)
        |
      Flatten
        |
Linear: 784 → hidden
        |
Linear: hidden → hidden // 2
        |
Linear: hidden // 2 → 10
        |
Class logits for digits 0–9
```

The model uses configurable hidden-layer size, bias settings, and weight initialization. The output layer produces ten logits, one for each digit class. During classification, the predicted class is typically selected using the largest logit; a softmax transformation can be applied when class probabilities are needed.

With `hidden=300`, the layer dimensions are `784 → 300 → 150 → 10`. For a model with biases enabled, this architecture has 282,160 parameters.

### Architecture B: LoRA-enabled MLP — `PytorchTensorDigitMistLoRA`

The LoRA variant extends the dense model and replaces its linear layers with LoRA-enabled linear layers from `loralib`. LoRA introduces low-rank trainable updates to the linear transformations, providing an alternative parameter-efficient adaptation approach.

The LoRA rank is configurable (default rank 8 in the model factory). The standard MLP and LoRA-enabled MLP can therefore be compared using a consistent overall task and layer layout.

### Model factory — `TensorModels`

The `TensorModels(...)` factory creates the selected model and returns a configuration dictionary. The documented configuration options include:

- `KERNEL_INIT`: weight initialization strategy.
- `hidden`: hidden-layer width.
- `use_bias`: whether linear layers use bias parameters.
- `lora_uses`: whether to create the LoRA variant.
- `lora_rank`: rank used by LoRA layers.

The supported initialization names in the implementation are `GlorotNormal`, `GlorotUniform`, `HeNormal`, and `HeUniform`. The factory validates and normalizes initialization names before constructing the model.

## 4. Training, CUDA, and Numerical Utilities

The implementation uses PyTorch's tensor operations, autograd, and optimizer workflow for forward computation and gradient-based learning. Where CUDA is available, the model and input batches can be moved to the selected CUDA device. Otherwise, the run can use the CPU if the surrounding code supports fallback.

CUDA availability depends on the installed PyTorch build, NVIDIA driver, and compatible GPU. A CUDA-capable driver alone does not guarantee that `torch.cuda.is_available()` will return `True`.

Typical device selection is:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Training utilities in the notebook also include support helpers for automatic mixed precision (AMP), gradient scaling compatibility, and DataLoader handling. Mixed precision can reduce memory use and improve throughput on supported hardware, but its actual benefit depends on the GPU, model, and workload.

## 5. Evaluation and Experiment Tracking

The PyTorch workflow is designed to evaluate the trained classifier and report model performance through the application interface rather than relying only on console output. The experiment workflow also integrates with MLflow for tracking and model management.

The intended tracking split is:

- **Parameters:** input/configuration choices supplied through the Gradio interface, such as model selection and model/training settings.
- **Metrics and report:** output-side evaluation results and a readable summary displayed in the Gradio interface.
- **Artifacts:** selected models and other relevant outputs produced by the run.
- **Model registry:** the registered model name used by this project is `Digit Mist Pytorch Cuda`.

Exact metric names and values depend on the current experiment run; this README does not claim benchmark results that have not been supplied.

## 6. Application and Deployment Workflow

The project workflow connects the model experiment to a Gradio interface and MLflow tracking/model registration. At a high level:

1. Select the model and experiment settings in the Gradio UI.
2. Create the selected PyTorch model.
3. Prepare MNIST batches and train/evaluate the model.
4. Record input parameters, output metrics, and artifacts in MLflow.
5. Display a summary report in the UI.
6. Register and retrieve the model using the project's MLflow model-registry configuration.

Deployment environments must have compatible Python, PyTorch, and CUDA components if GPU execution is required. A CPU-only environment can still run only if the application and installed dependencies support CPU execution.

## 7. Purpose of the PyTorch Addition

This addition complements the existing Keras/TensorFlow tutorials with a lower-level PyTorch implementation. It makes model construction, parameter initialization, device placement, training utilities, LoRA configuration, experiment tracking, and deployment integration explicit, while keeping the original DigitMNIST, Fashion-MNIST, NeuralTutorials, and TensorFlowTutorials descriptions in place.

