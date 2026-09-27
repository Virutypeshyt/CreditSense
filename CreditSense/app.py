from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

# --------------------------------------------------
# 1. LOAD DATASET & PREPARE DATA
# --------------------------------------------------

DATA_FILE = "creditsense_loans.csv"
df = pd.read_csv(DATA_FILE)

# Features used by CreditSense
FEATURES = [
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "debt_to_income",
    "age"
]

FEATURE_LABELS = {
    "income": "Annual Income ($)",
    "credit_score": "Credit Score",
    "loan_amount": "Loan Amount ($)",
    "employment_years": "Employment (Years)",
    "debt_to_income": "Debt-to-Income Ratio",
    "age": "Age (Years)"
}

TARGET = "default"

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# 2. TRAIN GAUSSIAN NAIVE BAYES PIPELINE
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

model.fit(X_train, y_train)

# --------------------------------------------------
# 3. COMPUTE MODEL PERFORMANCE METRICS
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = float(accuracy_score(y_test, y_pred))
precision = float(precision_score(y_test, y_pred))
recall = float(recall_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print(f"CreditSense Model Accuracy : {accuracy:.2%}")
print(f"CreditSense Model Precision: {precision:.2%}")
print(f"CreditSense Model Recall   : {recall:.2%}")
print(f"CreditSense Model F1-Score : {f1:.2%}")

# --------------------------------------------------
# 4. COMPUTE DATASET STATISTICS & UNIT 1 METRICS
# --------------------------------------------------

def compute_dataset_stats():
    total_records = len(df)
    default_count = int((df[TARGET] == 1).sum())
    non_default_count = int((df[TARGET] == 0).sum())
    default_pct = round((default_count / total_records) * 100, 2)
    non_default_pct = round((non_default_count / total_records) * 100, 2)

    # Descriptive summary for each feature
    feature_summaries = {}
    for col in FEATURES:
        feature_summaries[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "variance": round(float(df[col].var()), 2),
            "min": round(float(df[col].min()), 2),
            "median": round(float(df[col].median()), 2),
            "max": round(float(df[col].max()), 2),
            "label": FEATURE_LABELS[col]
        }

    # Class-conditional means and stds (Gaussian NB parameters: mu and sigma)
    class_conditionals = {}
    for col in FEATURES:
        class_conditionals[col] = {
            "mean_repaid": round(float(df[df[TARGET] == 0][col].mean()), 2),
            "std_repaid": round(float(df[df[TARGET] == 0][col].std()), 2),
            "mean_default": round(float(df[df[TARGET] == 1][col].mean()), 2),
            "std_default": round(float(df[df[TARGET] == 1][col].std()), 2)
        }

    # Covariance Matrix calculation (Unit 1 requirement)
    cov_df = df[FEATURES].cov().round(2)
    corr_df = df[FEATURES].corr().round(3)

    cov_matrix = {
        "columns": FEATURES,
        "labels": [FEATURE_LABELS[col] for col in FEATURES],
        "matrix_rows": cov_df.values.tolist(),
        "corr_values": corr_df.values.tolist()
    }

    # Significant feature relationships for clear presentation
    key_relationships = [
        {
            "pair": ("income", "loan_amount"),
            "label": "Annual Income & Loan Amount",
            "cov": float(cov_df.loc["income", "loan_amount"]),
            "corr": float(corr_df.loc["income", "loan_amount"]),
            "interpretation": "Positive association (+0.73). Higher income values and larger loan amounts tend to vary together in the dataset."
        },
        {
            "pair": ("credit_score", "debt_to_income"),
            "label": "Credit Score & Debt-to-Income",
            "cov": float(cov_df.loc["credit_score", "debt_to_income"]),
            "corr": float(corr_df.loc["credit_score", "debt_to_income"]),
            "interpretation": "Negative association (-0.34). Credit score and debt-to-income ratio tend to vary inversely in the dataset."
        },
        {
            "pair": ("loan_amount", "debt_to_income"),
            "label": "Loan Amount & Debt-to-Income",
            "cov": float(cov_df.loc["loan_amount", "debt_to_income"]),
            "corr": float(corr_df.loc["loan_amount", "debt_to_income"]),
            "interpretation": "Positive association (+0.27). Loan amount and debt-to-income ratio tend to vary together in the dataset."
        }
    ]

    # Histogram distribution data for Chart.js
    # Credit Score bins
    cs_bins = [300, 500, 600, 650, 700, 750, 850]
    cs_labels = ["300-499", "500-599", "600-649", "650-699", "700-749", "750-850"]
    cs_repaid = []
    cs_default = []
    for i in range(len(cs_bins) - 1):
        low, high = cs_bins[i], cs_bins[i + 1]
        cs_repaid.append(int(((df[TARGET] == 0) & (df["credit_score"] >= low) & (df["credit_score"] < high)).sum()))
        cs_default.append(int(((df[TARGET] == 1) & (df["credit_score"] >= low) & (df["credit_score"] < high)).sum()))

    # Debt-to-Income bins
    dti_bins = [0.0, 0.20, 0.35, 0.50, 0.65, 1.01]
    dti_labels = ["0.00-0.19", "0.20-0.34", "0.35-0.49", "0.50-0.64", "0.65-1.00"]
    dti_repaid = []
    dti_default = []
    for i in range(len(dti_bins) - 1):
        low, high = dti_bins[i], dti_bins[i + 1]
        dti_repaid.append(int(((df[TARGET] == 0) & (df["debt_to_income"] >= low) & (df["debt_to_income"] < high)).sum()))
        dti_default.append(int(((df[TARGET] == 1) & (df["debt_to_income"] >= low) & (df["debt_to_income"] < high)).sum()))

    # Income bins
    inc_bins = [10000, 35000, 50000, 70000, 100000, 300000]
    inc_labels = ["$10k-$34k", "$35k-$49k", "$50k-$69k", "$70k-$99k", "$100k+"]
    inc_repaid = []
    inc_default = []
    for i in range(len(inc_bins) - 1):
        low, high = inc_bins[i], inc_bins[i + 1]
        inc_repaid.append(int(((df[TARGET] == 0) & (df["income"] >= low) & (df["income"] < high)).sum()))
        inc_default.append(int(((df[TARGET] == 1) & (df["income"] >= low) & (df["income"] < high)).sum()))

    return {
        "total_records": total_records,
        "num_features": len(FEATURES),
        "default_count": default_count,
        "non_default_count": non_default_count,
        "default_pct": default_pct,
        "non_default_pct": non_default_pct,
        "feature_summaries": feature_summaries,
        "class_conditionals": class_conditionals,
        "cov_matrix": cov_matrix,
        "key_relationships": key_relationships,
        "charts": {
            "credit_score": {
                "labels": cs_labels,
                "repaid": cs_repaid,
                "default": cs_default
            },
            "debt_to_income": {
                "labels": dti_labels,
                "repaid": dti_repaid,
                "default": dti_default
            },
            "income": {
                "labels": inc_labels,
                "repaid": inc_repaid,
                "default": inc_default
            }
        }
    }

dataset_stats = compute_dataset_stats()

# --------------------------------------------------
# 5. ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        f1=round(f1 * 100, 2),
        cm={"tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": len(y_test)},
        stats=dataset_stats,
        features=FEATURES,
        feature_labels=FEATURE_LABELS
    )

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return dataset statistics, distributions, and model performance metrics."""
    return jsonify({
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "test_size": len(y_test)
            }
        },
        "dataset": dataset_stats
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON payload received or invalid format."}), 400

        # Validate existence of all features
        for f in FEATURES:
            if f not in data or data[f] is None or str(data[f]).strip() == "":
                return jsonify({"error": f"Missing required feature: {FEATURE_LABELS.get(f, f)}"}), 400

        # Parse and type-cast with sensible bounds validation
        try:
            income = float(data["income"])
            credit_score = float(data["credit_score"])
            loan_amount = float(data["loan_amount"])
            employment_years = float(data["employment_years"])
            debt_to_income = float(data["debt_to_income"])
            age = float(data["age"])
        except ValueError:
            return jsonify({"error": "All inputs must be valid numeric values."}), 400

        if income < 0 or loan_amount < 0:
            return jsonify({"error": "Income and loan amount must be non-negative."}), 400
        if credit_score < 300 or credit_score > 850:
            return jsonify({"error": "Credit score must be between 300 and 850."}), 400
        if debt_to_income < 0.0 or debt_to_income > 1.0:
            return jsonify({"error": "Debt-to-Income ratio must be between 0.00 and 1.00 (e.g. 0.35)."}), 400
        if employment_years < 0 or employment_years > 70:
            return jsonify({"error": "Employment years must be between 0 and 70."}), 400
        if age < 18 or age > 100:
            return jsonify({"error": "Applicant age must be between 18 and 100."}), 400

        input_data = pd.DataFrame([{
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "age": age
        }])

        # Gaussian Naive Bayes prediction & probabilities
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        repayment_probability = float(probabilities[0])
        default_probability = float(probabilities[1])

        # Risk Classification thresholds
        if default_probability >= 0.70:
            risk = "HIGH"
            recommendation = "Statistical model estimate: High estimated probability of loan default based on trained Gaussian Naive Bayes densities."
        elif default_probability >= 0.40:
            risk = "MEDIUM"
            recommendation = "Statistical model estimate: Moderate estimated probability of loan default near intermediate decision threshold."
        else:
            risk = "LOW"
            recommendation = "Statistical model estimate: Low estimated probability of loan default based on trained Gaussian Naive Bayes densities."

        return jsonify({
            "prediction": prediction,
            "prediction_label": "Likely to Default" if prediction == 1 else "Likely to Repay",
            "default_probability": round(default_probability * 100, 2),
            "repayment_probability": round(repayment_probability * 100, 2),
            "risk": risk,
            "recommendation": recommendation,
            "inputs": {
                "income": income,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "employment_years": employment_years,
                "debt_to_income": debt_to_income,
                "age": age
            }
        })

    except Exception as e:
        return jsonify({
            "error": f"An error occurred during risk assessment: {str(e)}"
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
