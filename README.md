# League of Legends Match Outcome Predictor 🎮🤖

An end-to-end Machine Learning pipeline built from scratch using **PyTorch** to predict the outcome (Win/Loss) of League of Legends matches based on competitive in-game statistics.

---

## 📌 Project Overview
This project demonstrates how to utilize deep learning frameworks to handle binary classification tasks on tabular gaming data. By constructing a custom **Logistic Regression** architecture, handling optimization components manually, and implementing hyperparameter tuning, the model successfully captures key performance indicators that drive a team toward victory.

---

## 🛠️ Key Features & Implementation Steps

### 1. Data Processing & Preparation
* Processed and scaled competitive match statistics (Kills, Assists, Gold Earned, etc.).
* Converted standard structured data into PyTorch `float32` Tensors.
* Dynamically split the dataset into clean Training and Test subsets to ensure rigorous validation.

### 2. Model Architecture
* Engineered a custom neural network class inheriting from PyTorch’s base `nn.Module`.
* Utilized a Linear Layer (`nn.Linear`) mapped to a **Sigmoid Activation Function** to map raw model outputs into probabilities ranging between `0` and `1`.

### 3. Handcrafted Training Loop
* Formulated a robust epoch-based execution loop.
* Managed exact gradient clearing via `optimizer.zero_grad()` to prevent optimization leaks.
* Implemented **Binary Cross-Entropy Loss** (`nn.BCELoss`) as the primary loss objective.
* Handled automated backpropagation via `loss.backward()` and optimized model parameters with Stochastic Gradient Descent (`optimizer.step()`).

### 4. Tuning & Regularization
* Combined **L2 Regularization (Weight Decay)** within the optimizer to scale down large weights and effectively mitigate overfitting.
* Conducted sweeping grid searches across multiple Learning Rates (`0.01`, `0.05`, and `0.1`) to isolate the absolute optimal configuration.

### 5. Evaluation & Feature Importance
* Evaluated precision using **Confusion Matrices** and **ROC-AUC curves** to evaluate discriminative capabilities.
* Extracted the model's learned weights to map **Feature Importance**, discovering that `kills`, `assists`, and `gold_earned` serve as the heaviest mathematical contributors to match victories.

---

## 💻 Tech Stack
* **Language:** Python
* **Framework:** PyTorch (`nn.Module`, `nn.Linear`, `nn.BCELoss`)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/alaa-elbaz/League-of-Legends-Match-Outcome-Predictor-using-PyTorch.git](https://github.com/alaa-elbaz/League-of-Legends-Match-Outcome-Predictor-using-PyTorch.git)
