# Iris Classification Using Three Approaches

This repository provides implementations for the Iris Classification problem using three different approaches:

1. **Plain PyTorch**
2. **PyTorch Lightning**
3. **TensorFlow Keras (`tf.keras`)**

Each approach follows the CRISP-DM methodology, ensuring a structured workflow for data science projects.

## Overview
The Iris dataset is a classic dataset for classification. The goal is to predict the species of iris flowers based on their features: sepal length, sepal width, petal length, and petal width.

### Dataset Information:
- **Features:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target Variable:**
  - Iris Setosa
  - Iris Versicolour
  - Iris Virginica

### CRISP-DM Steps
1. **Business Understanding**: Identify the goal of classifying iris species.
2. **Data Understanding**: Explore the dataset.
3. **Data Preparation**: Process the data (encoding, normalization, splitting).
4. **Modeling**: Develop and train a classification model.
5. **Evaluation**: Assess model performance using metrics.
6. **Deployment**: Save the trained model.

---

## Implementation Details

### 1. Plain PyTorch
**Description:**
- Manages every aspect of training explicitly (e.g., forward pass, loss calculation, backpropagation).

**Advantages:**
- Full control over training logic.
- Ideal for research and experimentation.

**Disadvantages:**
- Boilerplate code can be repetitive.
- Requires manual handling of training/validation loops.

**File:** `iris_pytorch.py`

### 2. PyTorch Lightning
**Description:**
- Simplifies PyTorch code by abstracting boilerplate (e.g., training loops, logging).

**Advantages:**
- Cleaner code with modularity.
- Built-in support for distributed training and logging.

**Disadvantages:**
- Somewhat less control over specific details.

**File:** `iris_pytorch_lightning.py`

### 3. TensorFlow Keras (`tf.keras`)
**Description:**
- High-level API within TensorFlow, designed for ease of use.

**Advantages:**
- Easy to prototype and deploy.
- Tight integration with TensorFlow's ecosystem.

**Disadvantages:**
- Limited flexibility for customizations compared to PyTorch.

**File:** `iris_tf_keras.py`

---

## How to Use
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required libraries:
  - `torch`, `torchvision`, `pytorch-lightning`
  - `tensorflow`
  - `sklearn`
  - `numpy`
  - `pandas`

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/iris-classification.git
   cd iris-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the desired implementation:
   - **Plain PyTorch:**
     ```bash
     python iris_pytorch.py
     ```
   - **PyTorch Lightning:**
     ```bash
     python iris_pytorch_lightning.py
     ```
   - **TensorFlow Keras:**
     ```bash
     python iris_tf_keras.py
     ```

---

## Results
### Metrics Used:
- **Accuracy**
- **Classification Report**

Each implementation outputs the evaluation metrics for the test dataset and saves the trained model.

### Model Saving:
- **Plain PyTorch:** `iris_classifier_model.pth`
- **PyTorch Lightning:** `iris_classifier_model.pth`
- **TensorFlow Keras:** `iris_classifier_model.h5`

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## Acknowledgments
- **Iris Dataset:** Provided by the UCI Machine Learning Repository.
