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

st.title(" Adult Census Income Prediction")

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

st.subheader(" Upload Test Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # READ CSV
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully!")

    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    st.subheader(" Test Data Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.write(
        f"*Number of records:* {df.shape[0]}"
    )

    st.write(
        f"*Number of features:* {df.shape[1] - 1}"
    )


    # =====================================================
    # MODEL SELECTION
    # =====================================================

    st.subheader(" Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose a model",
        list(model_options.keys())
    )

    model_path = model_options[selected_model]


    # =====================================================
    # LOAD MODEL
    # =====================================================

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found: {model_path}"
        )

        st.stop()


    try:

        model = joblib.load(model_path)

        st.success(
            f"{selected_model} loaded successfully!"
        )

    except Exception as e:

        st.error(
            f"Unable to load {selected_model}."
        )

        st.exception(e)

        st.stop()


    # =====================================================
    # GENERATE PREDICTIONS
    # =====================================================

    st.subheader(" Generate Predictions")

    if st.button(
            " Predict Income",
            type="primary"
    ):

        with st.status(
                " Processing your data...",
                expanded=True
        ) as status:

            try:

                # -------------------------------------------------
                # STEP 1: PREPARE DATA
                # -------------------------------------------------

                st.write(
                    " Preparing test data..."
                )

                X_test = df.drop(
                    columns=["income"]
                )

                y_test = df["income"]


                # -------------------------------------------------
                # STEP 2: MODEL PREDICTION
                # -------------------------------------------------

                st.write(
                    " Running model prediction..."
                )

                y_pred = model.predict(X_test)


                # -------------------------------------------------
                # STEP 3: PREDICTION PROBABILITY
                # -------------------------------------------------

                st.write(
                    " Calculating prediction probabilities..."
                )

                y_prob = model.predict_proba(X_test)[:, 1]


                # -------------------------------------------------
                # STEP 4: CALCULATE METRICS
                # -------------------------------------------------

                st.write(
                    " Calculating evaluation metrics..."
                )

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred
                )

                recall = recall_score(
                    y_test,
                    y_pred
                )

                f1 = f1_score(
                    y_test,
                    y_pred
                )

                auc = roc_auc_score(
                    y_test,
                    y_prob
                )

                mcc = matthews_corrcoef(
                    y_test,
                    y_pred
                )


                # -------------------------------------------------
                # STEP 5: CONFUSION MATRIX
                # -------------------------------------------------

                st.write(
                    " Generating confusion matrix..."
                )

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )


                # -------------------------------------------------
                # STEP 6: CLASSIFICATION REPORT
                # -------------------------------------------------

                st.write(
                    " Generating classification report..."
                )

                report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True
                )


                # -------------------------------------------------
                # COMPLETED
                # -------------------------------------------------

                status.update(
                    label=" Prediction completed successfully!",
                    state="complete",
                    expanded=False
                )


                # =================================================
                # PREDICTION RESULTS
                # =================================================

                st.subheader("📋 Prediction Results")

                result_df = df.copy()

                result_df["Predicted Income"] = y_pred

                st.dataframe(
                    result_df,
                    use_container_width=True
                )


                # =================================================
                # DOWNLOAD PREDICTIONS
                # =================================================

                csv_data = result_df.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇ Download Predictions",
                    data=csv_data,
                    file_name="income_predictions.csv",
                    mime="text/csv"
                )


                # =================================================
                # MODEL EVALUATION
                # =================================================

                st.subheader("Model Evaluation")


                # -------------------------------------------------
                # FIRST ROW
                # -------------------------------------------------

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )

                col2.metric(
                    "Precision",
                    f"{precision:.4f}"
                )

                col3.metric(
                    "Recall",
                    f"{recall:.4f}"
                )


                # -------------------------------------------------
                # SECOND ROW
                # -------------------------------------------------

                col4, col5, col6 = st.columns(3)

                col4.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )

                col5.metric(
                    "AUC",
                    f"{auc:.4f}"
                )

                col6.metric(
                    "MCC",
                    f"{mcc:.4f}"
                )


                # =================================================
                # CONFUSION MATRIX
                # =================================================

                st.subheader("Confusion Matrix")

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


                # =================================================
                # CLASSIFICATION REPORT
                # =================================================

                st.subheader("📄 Classification Report")

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True
                )


            except Exception as e:

                status.update(
                    label="Prediction failed",
                    state="error",
                    expanded=True
                )

                st.error(
                    "An error occurred while generating predictions."
                )

                st.exception(e)


else:

    st.info(
        "Please upload test_data.csv to continue."
    )