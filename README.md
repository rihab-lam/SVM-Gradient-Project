# 📈 SVM-Gradient: Financial Risk Detection

**A first-principles implementation of a Support Vector Machine (SVM) using manual Gradient Ascent to detect systemic risk regimes in Bitcoin (BTC-USD).**

---

## 🔍 Overview
This repository serves as an educational deep-dive into **SVM optimization**, implementing a manual **Gradient Ascent** solver to navigate the Dual Lagrangian space. Applied to **BTC-USD** market data, the model demonstrates how first-principles machine learning can identify high-risk financial regimes through the lens of Support Vector geometry.

By bypassing high-level libraries like Scikit-Learn, this project demonstrates the manual implementation of the optimization engine required to find the optimal separating hyperplane in a non-linear financial environment.

## 🧠 Mathematical Background

The solver focuses on the **Dual Lagrange Problem** for soft-margin classification. The objective is to maximize the margin by finding the optimal $\alpha$ multipliers:

$$L(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$$

### Constraints & KKT Conditions
The algorithm enforces the **Karush-Kuhn-Tucker (KKT)** conditions through a projection step:
* **Box Constraints:** $0 \leq \alpha_i \leq C$, where $C$ is the penalty parameter for misclassification.
* **Update Rule:** $\alpha_i^{new} = \alpha_i^{old} + \eta \nabla L(\alpha_i)$, followed by clipping to the $[0, C]$ interval.

---

## 🚀 Key Features

* **Custom Optimization Engine:** A vectorized **Gradient Ascent** solver that updates Lagrange multipliers and ensures convergence within the feasible region.
* **Quantitative Feature Engineering:**
    * **Log-Returns:** Calculated to ensure price data stationarity.
    * **Rolling Volatility:** 10-day standard deviation used as a proxy for market stress.
    * **Z-Scores:** Identification of price anomalies relative to moving averages.
* **Dynamic Labeling:** Automated classification of market states based on the 80th percentile of historical volatility.

---

## 📊 Performance Visualization

<img src="https://github.com/user-attachments/assets/17f91a27-68c6-45fd-9956-ae19e79b107f" alt="Market Risk Analysis" width="1187">

### Results Interpretation
* **Market Risk Alerts (Red Dots):** The solver successfully flags periods of extreme convexity in price movement and heteroskedastic volatility "top" formations.
* **Support Vector Analysis:** The model identified **946 Support Vectors** (where $0 < \alpha_i \leq C$). These points define the geometry of the optimal separating hyperplane.
* **Model Sensitivity:** The clustering of alerts during the 2024–2026 phase indicates sensitivity to **stochastic volatility** and **momentum exhaustion**, typical of late-cycle market behavior.

---

## 📂 Project Architecture

```text
├── SVM-Gradient-Project.ipynb  # Main implementation & analysis
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT Open Source License
└── README.md                   # Project Documentation

