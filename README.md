
# 📈 SVM-Gradient: Financial Risk Detection

This repository features a robust, from-scratch implementation of a **Support Vector Machine (SVM)** designed to detect systemic risk periods in financial markets, specifically targeting **Bitcoin (BTC-USD)**. 

By bypassing high-level libraries like Scikit-Learn for the solver, this project demonstrates the manual implementation of **Gradient Ascent** to solve the SVM Dual Problem.

## 🧠 Mathematical Background

The solver focuses on the **Dual Lagrange Problem** for soft-margin classification. The objective is to maximize the margin by finding the optimal $\alpha$ multipliers:

$$L(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$$

### Constraints & KKT Conditions
The algorithm enforces the **Karush-Kuhn-Tucker (KKT)** conditions through a projection step:
* **Box Constraints:** $0 \leq \alpha_i \leq C$, where $C$ is the penalty parameter for misclassification.
* **Kernel Matrix:** The Gram matrix is computed as $K_{ij} = y_i y_j (x_i \cdot x_j)$ to efficiently process feature interactions.

---

## 🚀 Key Features

* **Custom Optimization Engine:** A vectorized **Gradient Ascent** solver that updates Lagrange multipliers and clips them to the feasible region $[0, C]$.
* **Quantitative Feature Engineering:**
    * **Log-Returns:** Calculated to ensure price data stationarity.
    * **Rolling Volatility:** 10-day standard deviation used as a proxy for market stress.
    * **Z-Scores:** Identification of price anomalies relative to the 20-day moving average.
* **Dynamic Labeling:** Automated classification of market states based on the 80th percentile of historical volatility.

---

## 📊 Performance Visualization

![Market Risk Analysis](<img width="1187" height="622" alt="image" src="https://github.com/user-attachments/assets/17f91a27-68c6-45fd-9956-ae19e79b107f" />)

### Results Interpretation
The visualization demonstrates the effectiveness of the custom SVM solver in identifying **non-linear regime shifts** within the BTC-USD price action.

* **Market Risk Alerts (Red Dots):** These represent data points classified as the **Positive Class (+1)**. The solver successfully flags periods of extreme convexity in price movement and heteroskedastic volatility "top" formations.
* **Support Vector Analysis:** The model identified **946 Support Vectors**. In the context of the Dual Lagrangian Formulation, these are the critical samples where $0 < \alpha_i \leq C$. These points reside on the margin boundaries and define the geometry of the optimal separating hyperplane.
* **Decision Boundary:** By mapping features like Log-Returns and Rolling Volatility into a high-dimensional space, the SVM constructs a decision surface that separates "Stable" from "Risk" states.
* **Historical Context:** The visualization spans from 2020 to early 2026, capturing major Bitcoin cycles. The clustering of alerts during the 2024–2026 phase indicates sensitivity to **stochastic volatility** and **momentum exhaustion**.

---

## 📂 Project Architecture

```text
├── SVM-Gradient-Project.ipynb  # Main implementation & analysis
├── requirements.txt            # NumPy, Pandas, Matplotlib
├── LICENSE                     # MIT Open Source License
└── README.md                   # Project Documentation
---

