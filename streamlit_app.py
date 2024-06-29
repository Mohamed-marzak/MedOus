import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to preprocess data
def preprocess_data(data, target):
    y = data[target]
    X = data.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, X_scaled

# Function to apply PCA
def apply_pca(data, n_components, target):
    X, y, X_scaled = preprocess_data(data, target)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X, y, X_pca, pca.explained_variance_ratio_

# Function to apply ensemble methods
def apply_ensemble(X, y, task, method, params):
    base_model_cls = DecisionTreeClassifier(random_state=42)
    base_model_reg = DecisionTreeRegressor(random_state=42)
    
    if task == 'Classification':
        if method == 'Bagging':
            model = BaggingClassifier(base_model_cls, n_estimators=params['n_estimators'], random_state=42)
        elif method == 'Boosting':
            model = AdaBoostClassifier(base_model_cls, n_estimators=params['n_estimators'], random_state=42)
        elif method == 'Stacking':
            estimators = [
                ('lr', LogisticRegression()),
                ('svc', SVC(probability=True))
            ]
            model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
        elif method == 'Voting':
            estimators = [
                ('dt', DecisionTreeClassifier()),
                ('lr', LogisticRegression()),
                ('gnb', GaussianNB())
            ]
            model = VotingClassifier(estimators=estimators, voting=params['voting'])
    else:  # Regression
        if method == 'Bagging':
            model = BaggingRegressor(base_model_reg, n_estimators=params['n_estimators'], random_state=42)
        elif method == 'Boosting':
            model = AdaBoostRegressor(base_model_reg, n_estimators=params['n_estimators'], random_state=42)
        elif method == 'Stacking':
            estimators = [
                ('lr', LinearRegression()),
                ('dt', DecisionTreeRegressor())
            ]
            model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
        elif method == 'Voting':
            estimators = [
                ('dt', DecisionTreeRegressor()),
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor())
            ]
            model = VotingRegressor(estimators=estimators)
    
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions

# Streamlit app layout
st.title('PCA and Ensemble Learning with Model Selection')

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())
    
    target = st.selectbox("Select Target Variable", data.columns)
    
    # PCA parameters
    st.sidebar.header('PCA Parameters')
    n_components = st.sidebar.slider('Number of PCA Components', 2, min(len(data.columns)-1, 10), 2)
    
    # Task selection
    st.sidebar.header('Task')
    task_choice = st.sidebar.selectbox('Select Task', ('Classification', 'Regression'))
    
    # Ensemble method selection
    st.sidebar.header('Ensemble Method')
    method_choice = st.sidebar.selectbox('Select Ensemble Method', ('Bagging', 'Boosting', 'Stacking', 'Voting'))
    
    # Method-specific parameters
    if method_choice in ['Bagging', 'Boosting']:
        n_estimators = st.sidebar.slider('Number of Estimators', 10, 100, 50, help="Number of base models used in the ensemble.")
    if method_choice == 'Voting' and task_choice == 'Classification':
        voting_type = st.sidebar.selectbox('Voting Type', ('hard', 'soft'))
    
    if st.sidebar.button('Apply PCA'):
        X, y, X_pca, explained_variance = apply_pca(data, n_components, target)
        st.write("PCA Explained Variance Ratio:")
        st.write(explained_variance)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        st.pyplot(plt)

    if st.sidebar.button('Apply Ensemble Learning'):
        params = {}
        if method_choice in ['Bagging', 'Boosting']:
            params['n_estimators'] = n_estimators
        if method_choice == 'Voting' and task_choice == 'Classification':
            params['voting'] = voting_type
        
        X, y, X_scaled = preprocess_data(data, target)
        
        model, predictions = apply_ensemble(X_scaled, y, task_choice, method_choice, params)
        
        if task_choice == 'Classification':
            accuracy = accuracy_score(y, predictions)
            st.write(f"{method_choice} {task_choice} Accuracy: {accuracy}")
        else:  # Regression
            mse = mean_squared_error(y, predictions)
            st.write(f"{method_choice} {task_choice} Mean Squared Error: {mse}")

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=predictions, palette='viridis')
        plt.title(f'{method_choice} {task_choice} Predictions')
        st.pyplot(plt)