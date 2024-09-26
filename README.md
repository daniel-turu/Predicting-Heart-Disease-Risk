## Project: Predicting Heart Disease Risk with Random Forest Classification

This project aims to develop a machine learning model capable of accurately predicting the likelihood of an individual having heart disease. Early detection of heart disease is crucial for timely intervention and improved patient outcomes. By leveraging readily available patient information, the model can serve as a cost-effective and effective screening tool.

### Data Description

The project utilizes two datasets:

* **Train Dataset:** A CSV file containing patient data and a target variable indicating the presence or absence of heart disease.
* **Test Dataset:** A CSV file similar to the training set but lacking the target variable. The model will be used to predict the target values for this dataset.

Key features within the datasets include:

* `age`
* `sex`
* `cp` (chest pain type)
* `trestbps` (resting blood pressure)
* `chol` (serum cholesterol in mg/dl)
* `fbs` (fasting blood sugar)
* `restecg` (resting electrocardiographic results)
* `thalach` (maximum heart rate achieved)
* `exang` (exercise induced angina)
* `oldpeak` (ST depression induced by exercise relative to rest)
* `slope` (slope of the peak exercise ST segment)
* `ca` (number of major vessels colored by flourosopy)
* `thal` (presence or absence of fixed/reversible defect)

### Methodology

#### Data Loading and Preprocessing

* Pandas is used to load the CSV files into DataFrames.
* Missing values are handled using imputation techniques (e.g., mean or median).
* Feature scaling is applied to ensure all features are within a similar range for better model performance.
* Categorical features are encoded using one-hot encoding.

#### Model Training

* A Random Forest Classifier is employed due to its robustness to overfitting, ability to identify influential features, and ensemble nature for reducing variance and improving generalization.
* The training data is split into training and validation sets using `train_test_split` to evaluate model performance.
* Hyperparameters of the Random Forest model (e.g., `n_estimators`, `max_depth`, `min_samples_split`) can be tuned for optimal results using techniques like grid search or random search (not included in this example).
* The model is trained on the training set using `model.fit(X_train, y_train)`.

#### Model Evaluation

Various metrics are used to assess the model's performance on the validation set:

* **Accuracy:** Proportion of correct predictions
* **Precision:** Ratio of true positives to all predicted positives
* **Recall:** Ratio of true positives to all actual positives
* **F1-score:** Harmonic mean of precision and recall
* **AUC-ROC:** Area Under the Receiver Operating Characteristic Curve

These metrics help determine the model's effectiveness in correctly classifying individuals with and without heart disease.

#### Prediction and Submission

* The trained model is applied to the test data using `model.predict(test_data)`, generating predicted target values (heart disease likelihood) for each data point.
* A submission file is created in CSV format with columns: "ID" (from the test data) and "Target" (predicted target values).


### Dependencies

This project requires the following Python libraries:

* pandas
* scikit-learn




> **<div style="text-align: right"> By Turu Daniel Joseph </div>**
