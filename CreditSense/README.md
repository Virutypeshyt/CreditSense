# CreditSense: Probabilistic Credit-Risk Assessment Dashboard

A full-stack Machine Learning application developed for **Unit 1: Probabilistic Machine Learning & Bayesian Reasoning** (Semester 5).

CreditSense evaluates the likelihood of loan default from applicant demographic and financial indicators using **Gaussian Naive Bayes**, demonstrating foundational probability theorems, continuous normal likelihood density, feature expectation, variance, and covariance.

---

## 📌 Project Overview

- **Core Algorithm:** Gaussian Naive Bayes (`GaussianNB` with `StandardScaler` pipeline)
- **Problem Type:** Supervised Binary Classification / Probabilistic Risk Estimation
- **Target Label:** `default`
  - `0`: Loan Repaid / Non-Default
  - `1`: Loan Default
- **Dataset:** `creditsense_loans.csv` (1,500 borrower records in the provided dataset, 6 continuous features)
- **Tech Stack:** Python, Flask, HTML5, CSS3, JavaScript (ES6+), Chart.js, pandas, scikit-learn

---

## 🧮 Unit 1 Theoretical Concepts Demonstrated

### 1. Bayes' Theorem
Calculates the posterior probability of default given borrower evidence $\mathbf{X} = [x_1, x_2, \dots, x_6]$:
$$P(Y = c \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid Y = c) \cdot P(Y = c)}{P(\mathbf{X})}$$

### 2. Class-Conditional Independence (The "Naive" Assumption)
Assumes all financial attributes are conditionally independent given the loan outcome:
$$P(\mathbf{X} \mid Y = c) = \prod_{i=1}^{6} P(x_i \mid Y = c)$$

### 3. Continuous Gaussian Likelihood
Because financial metrics are continuous variables, each feature likelihood is estimated via normal distribution density parameters (empirical mean $\mu_{c,i}$ and variance $\sigma^2_{c,i}$):
$$P(x_i \mid Y = c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$

### 4. Expectation, Variance, & Covariance
- **Expectation (Mean):** $\mathbb{E}[X] = \mu$
- **Variance:** $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \sigma^2$
- **Covariance:** $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$
  - Indicates whether continuous variables tend to vary together (e.g., positive association between Annual Income and Loan Amount). Note that covariance reflects statistical association, not causation.

---

## 📊 Dataset Features

| Feature Name | Description | Range / Unit |
| :--- | :--- | :--- |
| `income` | Applicant annual income | \$11,401 – \$277,542 |
| `credit_score` | FICO credit score rating | 408 – 850 |
| `loan_amount` | Principal loan amount requested | \$25,000 – \$400,000 |
| `employment_years` | Total years of continuous employment | 0.0 – 25.0 Years |
| `debt_to_income` | Debt-to-Income (DTI) ratio | 0.05 – 0.95 (5% – 95%) |
| `age` | Borrower age | 21 – 70 Years |
| `default` (Target) | Loan default occurrence | 0 (Repaid) / 1 (Default) |

---

## 📈 Model Performance (20% Stratified Test Split)

| Metric | Score | Underwriting Context |
| :--- | :--- | :--- |
| **Accuracy** | **76.33%** | Overall correct classifications on unseen test records |
| **Precision** | **73.20%** | Accuracy when flagging a loan as high-risk / default |
| **Recall (Sensitivity)** | **78.87%** | Proportion of actual defaulters successfully caught |
| **F1-Score** | **75.93%** | Harmonic balance between Precision and Recall |

### Test Confusion Matrix ($N = 300$)
- **True Negatives (TN):** 117 (Repaid loans correctly predicted)
- **False Positives (FP):** 41 (Repaid loans predicted as default)
- **False Negatives (FN):** 30 (Defaulted loans missed)
- **True Positives (TP):** 112 (Defaulted loans correctly identified)

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.8+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Flask Server
```bash
python app.py
```

### 4. Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🎯 Key Application Features

1. **Fintech Risk Prediction Dashboard:** Enter financial values or use calibrated quick-fill scenario buttons:
   - 🟢 **Low Risk Example:** High score, low DTI, high income ($P(\text{Default}) < 5\%$)
   - 🟡 **Medium Risk Example:** Moderate score & DTI ($P(\text{Default}) \approx 52\%$)
   - 🔴 **High Risk Example:** Subprime score, elevated DTI ($P(\text{Default}) > 95\%$)
2. **Interactive Gauge & Risk Tiers:** Real-time visual categorization into Low Risk, Medium Risk, and High Risk.
3. **Dynamic Dataset Analytics:** Empirical distributions rendered with Chart.js (Outcome ratios, Credit score distribution, DTI impact, and Gaussian mean comparison).
4. **Covariance Matrix & Relationships:** Empirical covariance calculated directly from dataset records with mathematical explanations.
5. **Viva Q&A Guide:** Accordion containing key questions and answers on Bayesian ML for academic evaluation.

---

## 📂 Project Architecture

```
CreditSense/
│
├── app.py                   # Flask server, data ingestion, GaussianNB pipeline, APIs
├── creditsense_loans.csv    # Empirical loan dataset (1,500 records)
├── requirements.txt         # Project dependencies
├── README.md                # Technical documentation & viva guide
│
├── templates/
│   └── index.html           # Full semantic dashboard UI & templates
│
└── static/
    ├── style.css            # Responsive fintech design system
    └── script.js            # Asynchronous validation, inference, and Chart.js setup
```

---

## 🎓 College Viva Cheat Sheet

- **Q: Why is StandardScaler included in the pipeline if Gaussian Naive Bayes does not require it?**
  - **A:** Gaussian Naive Bayes does not inherently require feature scaling because it estimates the mean ($\mu$) and variance ($\sigma^2$) for each continuous feature independently. Standardizing features is included here as a standard scikit-learn pipeline practice to maintain numerical consistency and modular workflow design across features of vastly different scales, without altering Gaussian NB's class-conditional likelihoods.
- **Q: Does Naive Bayes predict true probabilities?**
  - **A:** While the independence assumption often pushes probabilities toward 0 and 1, the probability rankings and relative thresholds provide reliable probabilistic risk stratification.
- **Q: How does Naive Bayes prevent division by zero in Gaussian density?**
  - **A:** Scikit-learn's `GaussianNB` adds a small smoothing variance component (`var_smoothing = 1e-9` by default) to empirical variances to prevent zero denominators.