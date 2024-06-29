import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to preprocess data
def preprocess_data(data):
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data, data_scaled

# Function to apply PCA
def apply_pca(data, n_components):
    data_preprocessed, data_scaled = preprocess_data(data)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    return data_preprocessed, data_pca, pca.explained_variance_ratio_

# Function to simulate cluster labels using KMeans
def simulate_clusters(data, n_clusters=3):
    data_preprocessed, data_scaled = preprocess_data(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    data_preprocessed['Cluster'] = cluster_labels
    return data_preprocessed, cluster_labels, data_scaled

# Function to apply ensemble learning for visualization
def apply_ensemble(data, method, params, cluster_labels):
    data_preprocessed, data_scaled = preprocess_data(data)
    base_model = DecisionTreeClassifier(random_state=42)
    
    if method == 'Bagging':
        n_estimators = params.get('n_estimators', 10)
        model = BaggingClassifier(base_model, n_estimators=n_estimators, random_state=42)
    elif method == 'Boosting':
        n_estimators = params.get('n_estimators', 50)
        model = AdaBoostClassifier(base_model, n_estimators=n_estimators, random_state=42)
    elif method == 'Stacking':
        estimators = [
            ('lr', LogisticRegression()),
            ('svc', SVC())
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
    elif method == 'Voting':
        estimators = [
            ('dt', DecisionTreeClassifier()),
            ('lr', LogisticRegression()),
            ('gnb', GaussianNB())
        ]
        voting_type = params.get('voting', 'hard')
        model = VotingClassifier(estimators=estimators, voting=voting_type)
    
    # Fit the model using simulated cluster labels
    model.fit(data_scaled, cluster_labels)
    ensemble_labels = model.predict(data_scaled)
    data_preprocessed['Cluster'] = ensemble_labels
    return data_preprocessed, ensemble_labels, data_scaled

# Streamlit app layout
st.title('PCA and Ensemble Learning with Model Selection')

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # PCA parameters
    st.sidebar.header('PCA Parameters')
    n_components = st.sidebar.slider('Number of PCA Components', 2, min(len(data.columns), 10), 2)
    
    # Ensemble method selection
    st.sidebar.header('Ensemble Method')
    method_choice = st.sidebar.selectbox('Select Ensemble Method', ('Bagging', 'Boosting', 'Stacking', 'Voting'))
    
    # Method-specific parameters
    if method_choice in ['Bagging', 'Boosting']:
        n_estimators = st.sidebar.slider('Number of Estimators', 10, 100, 50)
    if method_choice == 'Voting':
        voting_type = st.sidebar.selectbox('Voting Type', ('hard', 'soft'))
    
    if st.sidebar.button('Apply PCA'):
        data_pca, data_pca_transformed, explained_variance = apply_pca(data, n_components)
        st.write("PCA Explained Variance Ratio:")
        st.write(explained_variance)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data_pca_transformed[:, 0], y=data_pca_transformed[:, 1])
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        st.pyplot(plt)

    if st.sidebar.button('Apply Ensemble Learning'):
        params = {}
        if method_choice in ['Bagging', 'Boosting']:
            params['n_estimators'] = n_estimators
        if method_choice == 'Voting':
            params['voting'] = voting_type
        
        # Simulate cluster labels
        data_clustered, simulated_labels, data_scaled = simulate_clusters(data, n_clusters=3)
        
        # Apply ensemble method using simulated labels
        data_clustered, ensemble_labels, data_scaled = apply_ensemble(data, method_choice, params, simulated_labels)
        
        st.write(f"{method_choice} Ensemble Learning Results:")
        st.write(data_clustered[['Cluster']].value_counts())

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data_clustered['Cluster'], palette='viridis')
        plt.title(f'{method_choice} Clusters')
        st.pyplot(plt)