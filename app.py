import os

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adult Census Income Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Prediction")

st.write(
    "Machine Learning Classification Application"
)

st.info(
    "Upload the test CSV file, select a trained model, "
    "and generate income predictions."
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

model_options = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}


# =========================================================
# UPLOAD TEST DATA
# =========================================================

st.subheader("Upload Test Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to continue."
    )

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        "Unable to read the uploaded CSV file."
    )

    st.exception(e)

    st.stop()


st.success(
    "Test data uploaded successfully!"
)


# =========================================================
# VALIDATE TARGET COLUMN
# =========================================================

if "income" not in df.columns:

    st.error(
        "The uploaded CSV must contain an 'income' column "
        "for evaluation."
    )

    st.stop()


# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("Test Data Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

col1, col2 = st.columns(2)

col1.metric(
    "Number of Records",
    df.shape[0]
)

col2.metric(
    "Number of Features",
    df.shape[1] - 1
)


# =========================================================
# MODEL SELECTION
# =========================================================

st.subheader("Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model",
    list(model_options.keys())
)

model_path = model_options[selected_model]


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

st.subheader("Generate Predictions")

predict_clicked = st.button(
    "Predict Income",
    type="primary",
    use_container_width=True
)


if predict_clicked:

    # =====================================================
    # CHECK MODEL FILE
    # =====================================================

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found: {model_path}"
        )

        st.stop()


    # =====================================================
    # LOADER PLACEHOLDER
    # =====================================================

    loader_placeholder = st.empty()

    loader_placeholder.markdown(
        """
        <style>
        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }

        .loader-container {
            text-align: center;
            padding: 35px;
        }

        .loader {
            border: 5px solid #e6e6e6;
            border-top: 5px solid #555555;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        .loader-text {
            margin-top: 15px;
            font-size: 18px;
        }
        </style>

        <div class="loader-container">
            <div class="loader"></div>
            <div class="loader-text">
                Processing your prediction...
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # VARIABLES
    # =====================================================

    model = None
    y_pred = None
    y_prob = None

    accuracy = None
    precision = None
    recall = None
    f1 = None
    auc = None
    mcc = None

    cm = None
    report = None

    processing_error = None


    # =====================================================
    # PROCESSING
    # =====================================================

    try:

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        model = joblib.load(
            model_path
        )


        # -------------------------------------------------
        # PREPARE TEST DATA
        # -------------------------------------------------

        X_test = df.drop(
            columns=["income"]
        )

        y_test = df["income"]


        # -------------------------------------------------
        # GENERATE PREDICTIONS
        # -------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        if hasattr(
                model,
                "predict_proba"
        ):

            probabilities = model.predict_proba(
                X_test
            )

            if probabilities.shape[1] >= 2:

                y_prob = probabilities[:, 1]


        # -------------------------------------------------
        # EVALUATION METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )


        # -------------------------------------------------
        # AUC SCORE
        # -------------------------------------------------

        if y_prob is not None:

            auc = roc_auc_score(
                y_test,
                y_prob
            )

        else:

            auc = roc_auc_score(
                y_test,
                y_pred
            )


        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )


        # -------------------------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )


    except Exception as e:

        processing_error = e


    # =====================================================
    # REMOVE LOADER
    # =====================================================

    loader_placeholder.empty()


    # =====================================================
    # HANDLE PROCESSING ERROR
    # =====================================================

    if processing_error is not None:

        st.error(
            "An error occurred while generating predictions."
        )

        st.exception(
            processing_error
        )

        st.stop()


    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    with st.expander(
            "Prediction Results",
            expanded=True
    ):

        result_df = df.copy()

        result_df["Predicted Income"] = y_pred

        st.dataframe(
            result_df,
            use_container_width=True
        )


        # -------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # -------------------------------------------------

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="income_predictions.csv",
            mime="text/csv"
        )


    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    with st.expander(
            "Model Evaluation",
            expanded=True
    ):

        st.write(
            f"Selected Model: *{selected_model}*"
        )

        st.write(
            "Evaluation Metrics"
        )


        # -------------------------------------------------
        # FIRST ROW
        # -------------------------------------------------

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        metric_col2.metric(
            "Precision",
            f"{precision:.4f}"
        )

        metric_col3.metric(
            "Recall",
            f"{recall:.4f}"
        )


        # -------------------------------------------------
        # SECOND ROW
        # -------------------------------------------------

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        metric_col5.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        metric_col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    with st.expander(
            "Confusion Matrix",
            expanded=True
    ):

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual <=50K",
                "Actual >50K"
            ],
            columns=[
                "Predicted <=50K",
                "Predicted >50K"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    with st.expander(
            "Classification Report",
            expanded=False
    ):

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )