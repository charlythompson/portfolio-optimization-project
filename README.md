# 📊 Portfolio Optimisation Project (TCNJ – Extended Version)

This Python program calculates and visualises the **Efficient Frontier** for a two-stock portfolio and runs a **Monte Carlo simulation** to project future portfolio performance.  
It was developed as part of coursework at **The College of New Jersey (TCNJ)** and demonstrates key principles of **Modern Portfolio Theory**.

---

## ⚙️ Features
- Calculates:
  - Expected returns
  - Risk (standard deviation)
  - Correlation
  - Sharpe ratio
- Builds and plots the **Efficient Frontier**
- Identifies:
  - **Minimum Variance Portfolio (MVP)**
  - **Optimal Portfolio** (highest Sharpe ratio)
- Runs a **Monte Carlo simulation** of future portfolio growth

---

## 🧮 File Overview
| File | Description |
|------|--------------|
| `project4 extended.py` | Main program — efficient frontier + Monte Carlo simulation |
| `portfolio.py` | Class for portfolio metrics (expected return, variance, Sharpe ratio) |
| `weights.py` | Recursive function to generate portfolio weights |
| `requirements.txt` | Python dependencies |

---

## 💻 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YOURUSERNAME/portfolio-optimization-tcnj.git
   cd portfolio-optimization-tcnj
