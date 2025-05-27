# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:50:53 2025

@author: pauli
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#import csv
#import sys
#import re
import numpy as np
import seaborn as sns
#import shap

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score
#from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import StackingClassifier, VotingClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #, LinearRegression, 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from yellowbrick.features import FeatureImportances

class RFECV_EXPERIMENT:
    
    def init(self):
        self.data = [0][0]
        self.x = [0][0]
        self.y = [0]
        
        
    def open_file(self):
        base_dir = os.path.dirname(__file__)  # Directory of the current script
    
        #data_path = os.path.join(base_dir, "/PhD/DIS9903A/Week 7/")
        data_path = os.path.join(base_dir, "/DataCollection/Source")
        print(data_path)
        os.chdir(data_path) 
    
        network_flow_path = input("Enter the file containing the network flow data: ")
        file = os.path.join(data_path, network_flow_path)
        print(file)
        print(network_flow_path)
        
        # using panda makes it easier to manipulate the data
        data = pd.read_csv(network_flow_path, sep=',')
        self.data = data
        return data
    
    def standardize_data(self, data):
        
        print("Standardizing the data")
              
        from scipy import stats
        
        # this call does all of the data standardizationd
        data_new_scaled = stats.zscore(data)
        return data_new_scaled
                    
    def extract_features(self, data, file):
        print("Extracting the features using RFECV")
        # Remove the columns with the number listing from the data preprocessing step
        y_col = "Unnamed: 0"
        
        y = data[y_col]
        
        data = data.drop([y_col], axis = 1)
        
        #y1_col = "No"
        y1_col = "number"
        
        y = data[y1_col]
        
        
        X = data.drop([y1_col], axis = 1)
        
        y_col = "info_converted"
        #y_col = "Source_address"
        #y_col = "No"  # Use this only when using the IMB SPPS data instead of the R data preprocessed data
        
        y = data[y_col] # target data for the RFECV
        #print(y)
        
        X = data.drop([y_col], axis = 1)
        
        dt_model = DecisionTreeClassifier(random_state=42)
       
        print("Making Data Classification Complete")
   
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model
        rfecv = RFECV(estimator=dt_model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        
        y = y.to_frame()
        y = np.ravel(y)
        #y = pd.DataFrame(y)
        
        #print(y)
       
        # Split the data for the DT and GNB models 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
       
        # train the model and identify the features
        features = rfecv.fit(X_train, y_train)
        #features = rfecv.fit_transform(X, y)
        #print(features)
        
        print("RFECV model fitting complete")
        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=cv, scoring='accuracy')
        # Step 6: Print results
        print(f"Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"Selected features: {rfecv.support_}", file=file)
        print(f"Cross-validation scores: {scores}", file=file)
        print(f"Cross validation Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Plot the RFECV feature data for visualization
        # Using Scatter plot
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validation Score (Accuracy)")
        plt.title("RFECV - Optimal Number of Features")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], marker='o')
        plt.grid()
        plt.show()
        
        # Plot feature rankings
        plt.figure(figsize=(10, 6))
        plt.bar(range(X.shape[1]), rfecv.ranking_, color='skyblue')
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Ranking")
        plt.title("Feature Rankings by RFECV")
        plt.xticks(range(X.shape[1]))
        plt.show()
        
        # Get the selected features
        selected_features = rfecv.support_  # Boolean mask of selected features
        
        # Heatmap of selected features
        sns.heatmap([selected_features], cmap="coolwarm", cbar=False, xticklabels=range(X.shape[1]))
        plt.xlabel("Feature Index")
        plt.title("Selected Features (1 = Selected, 0 = Not Selected)")
        plt.show()
        
        id_features = pd.DataFrame(rfecv.ranking_)
        
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(id_features, diag_kind="kde")
        plt.show()
        
        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=kf, scoring='accuracy')
        # Step 6: Print results
        print(f"KF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"KF Selected features: {rfecv.support_}", file=file)
        print(f"KF Cross-validation scores: {scores}", file=file)
        print(f"KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"KF RFECV ranking: {rfecv.ranking_}", file=file)
        
        print("Evaluating the DT Classification Model")
       
        print(f"Feature ranking: {features.ranking_}", file=file)
        importance = np.absolute(rfecv.estimator_.feature_importances_)
        print(f"Feature importance: {importance}", file,file)
        
        print("Selected Features:", selected_features)

        # Transform the dataset to include only selected features
        X_selected = rfecv.transform(X)
        
        # Split the data for the DT and GNB models 
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y)
       
        # Train the Decision Tree Classifier on the selected features
        dt_model.fit(X_train, y_train)
        
        # Plot the tree
        plt.figure(figsize=(10, 8))
        plot_tree(dt_model, filled=True)
        plt.show()

        y_pred = dt_model.predict(X_test)
        print(y_pred)
        
        dt_accuracy = dt_model.score(X_selected, y)
        print(f"DT Accuracy: {dt_accuracy:.2f}")
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
       
        print("DT Classifier Accuracy:", accuracy, file=file)
        print("DT Classifier Precision:", precision, file=file)
        print("DT Classifier Recall:", recall, file=file)
        
        viz = FeatureImportances(dt_model)
        viz.fit(X, y)
        viz.show()
        
        importances = dt_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        bars = plt.bar(range(X_selected.shape[1]), importances[indices], edgecolor="#008031", linewidth=1)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", size=8)

        plt.title("Feature Importances", size=20, loc="left", y=1.04, weight="bold")
        plt.ylabel("Importance")
        plt.xticks(range(X_selected.shape[1]), np.array(X.columns)[indices], rotation=90, size=12)
        plt.show()
        
        # Gaussian Naive Bayes Classifier
        gnb_model = GaussianNB()
        gnb_model.fit(X_train, y_train)
        gnb_predictions = gnb_model.predict(X_test)
        
        # Plot confusion matrix
        y_pred = gnb_model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        
        # Predict probabilities
        y_proba = gnb_model.predict_proba(X_test)

        # Plot probabilities for each class
        for i in range(y_proba.shape[1]):
            plt.hist(y_proba[:, i], bins=10, alpha=0.5, label=f'Class {i}')
        plt.title('GNB Model Predicted Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
        num_columns = X_train.shape[1]
        print(f"Number of columns in X_train: {num_columns}")
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}")
        
        num_columns = X_test.shape[1]
        print(f"Number of columns in X_test: {num_columns}")
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}")
        
        # Evaluate Gaussian Naive Bayes
        print("Gaussian Naive Bayes Classifier:", file=file)
        print(f"GNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}", file=file)
        print(classification_report(y_test, gnb_predictions, zero_division=1.0), file=file)
        
        # Combine using VotingClassifier
        voting_clf = VotingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], voting='hard')
        votingscore = voting_clf.fit(X_train, y_train)
        print(votingscore.predict(X_selected), file=file)
        
        y_pred = voting_clf.predict(X_test)
        print("VotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("VotingClassifier DT with GNB confusion matris: ", confusion_matrix(y_test, y_pred), file=file)
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        stacking_clf = StackingClassifier(estimators=[('dt',dt_model), ('gnb', gnb_model)], final_estimator=LogisticRegression())
        stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        
        # Evaluate
        y_pred = stacking_clf.predict(X_test)
        print("StackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("StackingClassifier DT with GNB Confusion Matrix:", confusion_matrix(y_test, y_pred), file=file)
       
        
    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
def main():
    print("Run the script")
    feature_selection = RFECV_EXPERIMENT()
    
    try:
        data_results = feature_selection.open_file()
        data = data_results
        file = open("D:/DataCollection/Source/output_combined.txt", "a")
        #data_scaled = feature_selection.standardize_data(data)
        #features = feature_selection.extract_features(data_scaled)
        feature_selection.extract_features(data, file)
        file.close()
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':


    main()

    
