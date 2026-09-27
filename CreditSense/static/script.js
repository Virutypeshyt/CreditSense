/**
 * CreditSense - Client-side Interactive Dashboard & Prediction Controller
 * Handles form validation, asynchronous inference, sample case loading,
 * and Chart.js dataset visualizations.
 */

// Sample test cases calibrated against the trained Gaussian NB model
const SAMPLE_CASES = {
    low: {
        income: 95000,
        credit_score: 780,
        loan_amount: 80000,
        employment_years: 8.0,
        debt_to_income: 0.15,
        age: 42
    },
    medium: {
        income: 58000,
        credit_score: 660,
        loan_amount: 120000,
        employment_years: 4.0,
        debt_to_income: 0.32,
        age: 35
    },
    high: {
        income: 28000,
        credit_score: 510,
        loan_amount: 180000,
        employment_years: 1.0,
        debt_to_income: 0.65,
        age: 24
    }
};

// DOM Elements
const form = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictButton");
const predictSpinner = document.getElementById("predictSpinner");
const predictBtnText = document.getElementById("predictButtonText");
const loadingState = document.getElementById("loadingState");
const resultSection = document.getElementById("resultSection");
const generalErrorBox = document.getElementById("generalFormError");

// Result Elements
const defaultProbEl = document.getElementById("defaultProbability");
const repaymentProbEl = document.getElementById("repaymentProbability");
const probBarFill = document.getElementById("probabilityBarFill");
const gaugeCircle = document.getElementById("gaugeCircle");
const riskBadge = document.getElementById("riskBadge");
const riskLevelText = document.getElementById("riskLevelText");
const predictionPill = document.getElementById("predictionPill");
const resultRecommendation = document.getElementById("resultRecommendation");

// Input Fields
const inputs = {
    income: document.getElementById("income"),
    credit_score: document.getElementById("credit_score"),
    loan_amount: document.getElementById("loan_amount"),
    employment_years: document.getElementById("employment_years"),
    debt_to_income: document.getElementById("debt_to_income"),
    age: document.getElementById("age")
};

const errorElements = {
    income: document.getElementById("incomeError"),
    credit_score: document.getElementById("creditScoreError"),
    loan_amount: document.getElementById("loanAmountError"),
    employment_years: document.getElementById("employmentYearsError"),
    debt_to_income: document.getElementById("dtiError"),
    age: document.getElementById("ageError")
};

/**
 * Populate form with calibrated scenario values
 */
function loadSampleCase(tier) {
    const data = SAMPLE_CASES[tier];
    if (!data) return;

    clearErrors();

    // Populate each field
    inputs.income.value = data.income;
    inputs.credit_score.value = data.credit_score;
    inputs.loan_amount.value = data.loan_amount;
    inputs.employment_years.value = data.employment_years;
    inputs.debt_to_income.value = data.debt_to_income;
    inputs.age.value = data.age;

    // Highlight fields briefly
    Object.values(inputs).forEach(input => {
        const box = input.closest(".input-prefix-box");
        if (box) {
            box.style.borderColor = "var(--primary)";
            setTimeout(() => {
                box.style.borderColor = "";
            }, 600);
        }
    });

    // Provide friendly hint
    predictBtn.focus();
}

/**
 * Validate form inputs before submission
 */
function validateInputs() {
    let isValid = true;
    clearErrors();

    const inc = parseFloat(inputs.income.value);
    const score = parseFloat(inputs.credit_score.value);
    const loan = parseFloat(inputs.loan_amount.value);
    const emp = parseFloat(inputs.employment_years.value);
    const dti = parseFloat(inputs.debt_to_income.value);
    const age = parseFloat(inputs.age.value);

    if (isNaN(inc) || inc < 0) {
        showFieldError("income", "Please enter a valid positive income.");
        isValid = false;
    }

    if (isNaN(score) || score < 300 || score > 850) {
        showFieldError("credit_score", "Credit score must be between 300 and 850.");
        isValid = false;
    }

    if (isNaN(loan) || loan < 500) {
        showFieldError("loan_amount", "Loan amount must be at least $500.");
        isValid = false;
    }

    if (isNaN(emp) || emp < 0 || emp > 60) {
        showFieldError("employment_years", "Employment must be between 0 and 60 years.");
        isValid = false;
    }

    if (isNaN(dti) || dti < 0 || dti > 1.0) {
        showFieldError("debt_to_income", "DTI ratio must be a decimal between 0.00 and 1.00.");
        isValid = false;
    }

    if (isNaN(age) || age < 18 || age > 100) {
        showFieldError("age", "Applicant age must be between 18 and 100.");
        isValid = false;
    }

    return isValid;
}

function showFieldError(field, msg) {
    if (errorElements[field]) {
        errorElements[field].textContent = msg;
    }
    if (inputs[field]) {
        const box = inputs[field].closest(".input-prefix-box");
        if (box) box.classList.add("input-error");
    }
}

function clearErrors() {
    Object.values(errorElements).forEach(el => {
        if (el) el.textContent = "";
    });
    Object.values(inputs).forEach(input => {
        const box = input.closest(".input-prefix-box");
        if (box) box.classList.remove("input-error");
    });
    if (generalErrorBox) {
        generalErrorBox.textContent = "";
        generalErrorBox.classList.add("hidden");
    }
}

/**
 * Handle form submission and ML API communication
 */
form.addEventListener("submit", async function (event) {
    event.preventDefault();

    if (!validateInputs()) {
        return;
    }

    const payload = {
        income: parseFloat(inputs.income.value),
        credit_score: parseFloat(inputs.credit_score.value),
        loan_amount: parseFloat(inputs.loan_amount.value),
        employment_years: parseFloat(inputs.employment_years.value),
        debt_to_income: parseFloat(inputs.debt_to_income.value),
        age: parseFloat(inputs.age.value)
    };

    // UI state: loading
    setLoadingState(true);
    resultSection.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        setLoadingState(false);

        if (!response.ok) {
            throw new Error(data.error || "Risk assessment failed.");
        }

        renderPredictionResult(data);

    } catch (err) {
        setLoadingState(false);
        showGeneralError(err.message || "An error occurred while connecting to the model API.");
    }
});

function setLoadingState(isLoading) {
    if (isLoading) {
        predictBtn.disabled = true;
        predictSpinner.classList.remove("hidden");
        predictBtnText.textContent = "Calculating Posterior...";
        loadingState.classList.remove("hidden");
    } else {
        predictBtn.disabled = false;
        predictSpinner.classList.add("hidden");
        predictBtnText.textContent = "Assess Credit Risk";
        loadingState.classList.add("hidden");
    }
}

function showGeneralError(msg) {
    if (generalErrorBox) {
        generalErrorBox.textContent = msg;
        generalErrorBox.classList.remove("hidden");
    }
}

/**
 * Display inference results with styled risk badges and gauges
 */
function renderPredictionResult(res) {
    const defaultPct = res.default_probability;
    const repayPct = res.repayment_probability;
    const risk = res.risk;

    defaultProbEl.textContent = defaultPct + "%";
    repaymentProbEl.textContent = repayPct + "%";
    probBarFill.style.width = Math.min(Math.max(defaultPct, 0), 100) + "%";

    riskLevelText.textContent = risk;
    predictionPill.textContent = res.prediction_label;

    // Recommendation text
    resultRecommendation.textContent = res.recommendation;

    // Style risk states dynamically
    riskBadge.className = "risk-badge";
    gaugeCircle.style.borderColor = "";

    if (risk === "HIGH") {
        riskBadge.classList.add("risk-high");
        riskBadge.textContent = "HIGH RISK (DEFAULT)";
        riskLevelText.style.color = "var(--risk-high-accent)";
        gaugeCircle.style.borderColor = "var(--risk-high-accent)";
        probBarFill.style.backgroundColor = "var(--risk-high-accent)";
        predictionPill.style.backgroundColor = "var(--risk-high-bg)";
        predictionPill.style.color = "var(--risk-high-text)";
    } else if (risk === "MEDIUM") {
        riskBadge.classList.add("risk-medium");
        riskBadge.textContent = "MEDIUM RISK";
        riskLevelText.style.color = "var(--risk-med-accent)";
        gaugeCircle.style.borderColor = "var(--risk-med-accent)";
        probBarFill.style.backgroundColor = "var(--risk-med-accent)";
        predictionPill.style.backgroundColor = "var(--risk-med-bg)";
        predictionPill.style.color = "var(--risk-med-text)";
    } else {
        riskBadge.classList.add("risk-low");
        riskBadge.textContent = "LOW RISK (FAVORABLE)";
        riskLevelText.style.color = "var(--risk-low-accent)";
        gaugeCircle.style.borderColor = "var(--risk-low-accent)";
        probBarFill.style.backgroundColor = "var(--risk-low-accent)";
        predictionPill.style.backgroundColor = "var(--risk-low-bg)";
        predictionPill.style.color = "var(--risk-low-text)";
    }

    // Reveal and smoothly scroll to result
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

/**
 * Reset form and hide previous results
 */
function resetAssessment() {
    form.reset();
    clearErrors();
    resultSection.classList.add("hidden");
    loadingState.classList.add("hidden");

    // Scroll to form card
    const formCard = document.querySelector(".form-card");
    if (formCard) {
        formCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

/**
 * Initialize Chart.js visual analytics from backend data
 */
function initCharts() {
    if (typeof Chart === "undefined" || !window.CREDITSENSE_DATA) {
        console.warn("Chart.js or backend dataset is not available.");
        return;
    }

    const { stats } = window.CREDITSENSE_DATA;
    if (!stats || !stats.charts) return;

    // 1. Target Class Prior Doughnut Chart
    const targetCtx = document.getElementById("targetDoughnutChart");
    if (targetCtx) {
        new Chart(targetCtx, {
            type: "doughnut",
            data: {
                labels: [
                    `Repaid / Class 0 (${stats.non_default_count})`,
                    `Default / Class 1 (${stats.default_count})`
                ],
                datasets: [{
                    data: [stats.non_default_count, stats.default_count],
                    backgroundColor: ["#10b981", "#ef4444"],
                    hoverBackgroundColor: ["#059669", "#dc2626"],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI", size: 12, weight: 600 },
                            color: "#334155",
                            padding: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const total = stats.total_records;
                                const val = ctx.raw;
                                const pct = ((val / total) * 100).toFixed(1);
                                return ` ${ctx.label}: ${val} (${pct}%)`;
                            }
                        }
                    }
                },
                cutout: "68%"
            }
        });
    }

    // 2. Credit Score Distribution Grouped Bar Chart
    const csCtx = document.getElementById("creditScoreChart");
    if (csCtx) {
        const csData = stats.charts.credit_score;
        new Chart(csCtx, {
            type: "bar",
            data: {
                labels: csData.labels,
                datasets: [
                    {
                        label: "Repaid (0)",
                        data: csData.repaid,
                        backgroundColor: "#10b981",
                        borderRadius: 4
                    },
                    {
                        label: "Default (1)",
                        data: csData.default,
                        backgroundColor: "#ef4444",
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: { boxWidth: 12, font: { size: 11, weight: 600 } }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 20 },
                        grid: { color: "#f1f5f9" }
                    }
                }
            }
        });
    }

    // 3. Debt-to-Income Ratio Distribution Grouped Bar Chart
    const dtiCtx = document.getElementById("dtiChart");
    if (dtiCtx) {
        const dtiData = stats.charts.debt_to_income;
        new Chart(dtiCtx, {
            type: "bar",
            data: {
                labels: dtiData.labels,
                datasets: [
                    {
                        label: "Repaid (0)",
                        data: dtiData.repaid,
                        backgroundColor: "#3b82f6",
                        borderRadius: 4
                    },
                    {
                        label: "Default (1)",
                        data: dtiData.default,
                        backgroundColor: "#f59e0b",
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: { boxWidth: 12, font: { size: 11, weight: 600 } }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        grid: { color: "#f1f5f9" }
                    }
                }
            }
        });
    }

    // 4. Class-Conditional Means Comparison (Gaussian Parameters μ₀ vs μ₁)
    const incCtx = document.getElementById("incomeChart");
    if (incCtx && stats.class_conditionals) {
        const cc = stats.class_conditionals;
        const featuresToCompare = ["credit_score", "debt_to_income", "employment_years", "age"];
        const featureLabels = ["Credit Score", "DTI Ratio (×100)", "Employment (Yrs)", "Age (Yrs)"];

        const repaidMeans = [
            cc.credit_score.mean_repaid,
            cc.debt_to_income.mean_repaid * 100, // scaled for visual readability
            cc.employment_years.mean_repaid,
            cc.age.mean_repaid
        ];

        const defaultMeans = [
            cc.credit_score.mean_default,
            cc.debt_to_income.mean_default * 100,
            cc.employment_years.mean_default,
            cc.age.mean_default
        ];

        new Chart(incCtx, {
            type: "bar",
            data: {
                labels: featureLabels,
                datasets: [
                    {
                        label: "Repaid Mean (μ₀)",
                        data: repaidMeans,
                        backgroundColor: "#10b981",
                        borderRadius: 4
                    },
                    {
                        label: "Default Mean (μ₁)",
                        data: defaultMeans,
                        backgroundColor: "#dc2626",
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: { boxWidth: 12, font: { size: 11, weight: 600 } }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        grid: { color: "#f1f5f9" }
                    }
                }
            }
        });
    }
}

/**
 * Highlight active navbar link on user scroll
 */
function initScrollSpy() {
    const sections = document.querySelectorAll("section[id]");
    const navLinks = document.querySelectorAll(".nav-link");

    window.addEventListener("scroll", () => {
        let currentSectionId = "";
        const scrollY = window.pageYOffset;

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 120;
            const sectionHeight = section.offsetHeight;
            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute("id");
            }
        });

        navLinks.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === `#${currentSectionId}`) {
                link.classList.add("active");
            }
        });
    });
}

// Bootstrap once document is ready
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    initScrollSpy();
});
