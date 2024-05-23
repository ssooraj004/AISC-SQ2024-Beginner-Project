# Project Overview
This project aims to develop a machine learning model to predict whether a patient has diabetes based on diagnostic measurements. The dataset used for this project is the Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. All patients in the dataset are females at least 21 years old of Pima Indian heritage.

# Dataset
## Context
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database.

## Content
The dataset consists of several medical predictor variables and one target variable, Outcome. 
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database 

The predictor variables include:
- Number of pregnancies
- Body Mass Index (BMI)
- Insulin level
- Age
- Glucose
- Blood pressure
- Skin Thickness
- Diabetes pedigree function

## Inspiration
The primary goal is to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes.

# Project Workflow
**1. Exploratory Data Analysis (EDA)**
We conducted a thorough EDA to understand the distribution of the data, identify any missing values, and uncover relationships between the variables. Key steps in EDA included:
- Data cleaning and preprocessing
- Visualizing the distribution of predictor variables
- Checking for correlations between variables

**2. Model Building**
We experimented with six different machine learning models to identify the best-performing one. The models evaluated were:
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
- Gradient Boosting
- XGBoost

**3. Model Evaluation**
Each model was evaluated based on its accuracy score. The model with the highest accuracy was selected for further development and deployment.

**4. Front End Development**
A simple front end was created to showcase the model. This allows users to input diagnostic measurements and receive a prediction on whether the patient has diabetes.

## Results
The model that performed the best was Random Forest, with an accuracy score of 79.22%. This model was chosen for deployment in the front-end application.

# Installation and Usage
## Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, flask (for the front end)

## Getting Started
To replicate or further develop this project, follow these steps:
```bash
git clone https://github.com/ssooraj004/AISC-SQ2024-Beginner-Project.git
cd AISC-SQ2024-Beginner-Project

# Install required Python packages
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the Jupyter notebooks
jupyter notebook

# Run the flask
flask run

# Run the front end (UI)
cd templates
npm run dev
```

# Conclusion
This project successfully developed a machine learning model to predict diabetes in patients based on diagnostic measurements. The model is deployed through a simple front-end interface, making it easy to use for non-technical users.

## Future Work
- Implement cross-validation techniques such as k-fold cross-validation to ensure the model's reliability
- Improve the model’s accuracy and recall rate
- Improve the frontend interface for better user experience and accessibility.
