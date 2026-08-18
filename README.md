Adult Census Income Prediction

1. Problem Statement

The objective of this project is to build a machine learning classification system using the Adult Census Income dataset.

The application predicts whether an individual's annual income is:

- "<=50K"
- ">50K"

Multiple classification algorithms are implemented and evaluated using standard classification metrics. The trained models are then deployed through an interactive Streamlit web application.

---

2. Dataset Description

The Adult Census Income dataset is a binary classification dataset obtained from the UCI Machine Learning Repository.

Dataset Source

UCI Machine Learning Repository – Adult Dataset

Dataset URL:

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Dataset Details

- Problem Type: Binary Classification
- Number of Instances: 48,842
- Number of Features: 14
- Target Variable: "income"
- Feature Types: Categorical and Integer
- Dataset Source: UCI Machine Learning Repository

Target Variable

The target variable is:

"income"

The target has two classes:

- "<=50K" – Annual income is less than or equal to $50,000
- ">50K" – Annual income is greater than $50,000

The objective is to predict whether an individual's annual income exceeds $50,000 based on census-related demographic and employment attributes.

---

3. GitHub Repository Link

GitHub Repository:

https://github.com/Shubhnay0101/Adult-Income-Classification


The repository contains the complete source code, trained model files, requirements.txt, README.md, and test data required for the project.

---

4. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)

Each model was evaluated using the following six evaluation metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

5. Model Performance Comparison

ML Model Name| Accuracy| AUC| Precision| Recall| F1| MCC
Logistic Regression| 0.8437| 0.9013| 0.7308| 0.5853| 0.6500| 0.5566
Decision Tree| 0.8071| 0.7442| 0.6091| 0.6192| 0.6141| 0.4856
KNN| 0.8250| 0.8539| 0.6668| 0.5879| 0.6249| 0.5131
Naive Bayes| 0.6068| 0.8249| 0.3793| 0.9210| 0.5373| 0.3742
Random Forest (Ensemble)| 0.8480| 0.8996| 0.7322| 0.6103| 0.6657| 0.5724

---

6. Observations on Model Performance

Logistic Regression

Logistic Regression achieved an accuracy of 0.8437 and the highest AUC score of 0.9013 among all the evaluated models. It provides strong overall performance and demonstrates good capability to distinguish between the two income classes.

Decision Tree

Decision Tree achieved an accuracy of 0.8071 and an AUC score of 0.7442. Its performance is lower than Logistic Regression, KNN, and Random Forest across most of the evaluation metrics.

KNN

KNN achieved an accuracy of 0.8250 and an AUC score of 0.8539. It provides reasonable classification performance but does not outperform Logistic Regression or Random Forest on the overall evaluation metrics.

Naive Bayes

Naive Bayes achieved the highest recall of 0.9210, indicating that it identifies a large proportion of the ">50K" income class. However, its precision of 0.3793 and accuracy of 0.6068 are considerably lower, resulting in a lower F1 score and MCC.

Random Forest (Ensemble)

Random Forest achieved the highest accuracy of 0.8480, highest precision of 0.7322, highest F1 score of 0.6657, and highest MCC of 0.5724. Its AUC score of 0.8996 is also very close to the highest AUC achieved by Logistic Regression.

Overall Winner

Random Forest (Ensemble Model)

Random Forest is selected as the overall winner because it provides the strongest overall performance across the majority of the evaluation metrics, including Accuracy, Precision, F1 Score, and MCC.

Although Logistic Regression achieves a slightly higher AUC score of 0.9013 compared with Random Forest's 0.8996, Random Forest performs better on the other major overall performance measures.

---

7. Streamlit Web Application

An interactive Streamlit web application has been developed to demonstrate the trained machine learning models.

The application allows the user to select one machine learning model at a time and generate predictions using the uploaded test data.

7.1 Test Data Upload

Users can upload the test dataset in CSV format.

The uploaded test data must contain the target column:

"income"

7.2 Model Selection

Users can select one trained classification model at a time from the dropdown:

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest

7.3 Income Prediction

After selecting a model, the user can click the Predict Income button to generate income predictions on the uploaded test data.

7.4 Evaluation Metrics

The application displays the following evaluation metrics for the selected model:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- MCC Score

7.5 Confusion Matrix

The application displays the confusion matrix for the selected model.

The classes are displayed as:

- "<=50K"
- ">50K"

7.6 Classification Report

The application displays the classification report containing:

- Precision
- Recall
- F1 Score
- Support

for both income classes.

The class labels are displayed as:

- "<=50K"
- ">50K"

7.7 Download Predictions

Users can download the generated predictions as a CSV file using the Download Predictions option.

---

8. Repository Structure

project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
└── random_forest.pkl

---

9. Requirements

The application uses the following Python libraries:

streamlit
pandas
scikit-learn
joblib

---

10. Running the Application Locally

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will open in the browser.

---

11. Streamlit Community Cloud Deployment

The application is deployed using Streamlit Community Cloud.

Live Streamlit Application

https://adult-income-classification-shu-bhnay.streamlit.app/

The deployed application provides an interactive frontend for:

- Uploading test data
- Selecting a machine learning model
- Generating income predictions
- Viewing evaluation metrics
- Viewing the confusion matrix
- Viewing the classification report
- Downloading prediction results

---

12. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, starting from dataset selection and model implementation to model evaluation and deployment.

Five classification models were implemented and evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.

Among the evaluated models, Random Forest achieved the strongest overall performance, with an accuracy of 0.8480, precision of 0.7322, F1 score of 0.6657, and MCC of 0.5724.

The trained models are integrated into an interactive Streamlit application where users can select one model at a time, upload test data, generate predictions, and view detailed evaluation results.