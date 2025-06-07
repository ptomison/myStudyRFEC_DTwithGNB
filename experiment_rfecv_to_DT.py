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
import time

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
        data_path = os.path.join(base_dir, "/PhD/DIS9903A/ConductExperiment/DataCollection/Source")
        print(data_path)
        os.chdir(data_path) 
    
        network_flow_path = input("Enter the file containing the network flow data: ")
        file = os.path.join(data_path, network_flow_path)
        print(file)
        print(network_flow_path)
        
        # using panda makes it easier to manipulate the data
        data = pd.read_csv(network_flow_path, sep=',')
        self.data = data
        return data, file
    
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
        
        data = data.drop([y1_col], axis = 1)
        
        y_col = "info_converted"
        #y_col = "Source_address"
        #y_col = "No"  # Use this only when using the IMB SPPS data instead of the R data preprocessed data
        
        y = data[y_col] # target data for the RFECV
        #print(y)
        
        X = data.drop([y_col], axis = 1)
        feature_names = X.columns
        
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
        #y.columns = ['info_converted']
        
        #print(y)
       
        # Split the data for the DT and GNB models 
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
       
        start_time = time.time()
        # train the model and identify the features
        #features = rfecv.fit(X, y)
        features = rfecv.fit_transform(X, y)
        #print(features)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"RFECV model fitting complete in: {training_time}", file=file)
        
        # Step 6a: Make predictions on the test set
        y_pred = rfecv.estimator_.predict(features)
        
        print(f"RFECV Prediction: {y_pred}", file=file)
        
        # Get the names of the selected features
        selected_features = [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"Selected Features: {selected_features}", file=file)
        
        # Retrieve the number of selected features
        num_selected_features = rfecv.n_features_
        
        # Calculate the number of omitted features
       # num_omitted_features = X.shape[1] - features_selected
        num_omitted_features = (~rfecv.support_).sum()
        
        print(f"Number of selected features: {num_selected_features}", file=file)
        print(f"Number of omitted features: {num_omitted_features}", file=file)
        
        s_features = rfecv.support_

        # Count duplicate features
        original_features = pd.DataFrame(X).columns
        selected_feature_names = original_features[s_features]
        duplicates = len(selected_feature_names) - len(set(selected_feature_names))

        print(f"RFECV Number of duplicate features: {duplicates}", file=file)

        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=cv, scoring='accuracy')
        # Step 6: Print results
        print(f"RFECV Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV Selected features: {rfecv.support_}", file=file)
        print(f"RFECV Cross-validation scores: {scores}", file=file)
        print(f"RFECV Cross validation Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Plot the RFECV feature data for visualization
        # Using Scatter plot
        plt.figure(figsize=(10, 6))
        plt.xlabel("RFECV Number of Features Selected")
        plt.ylabel("RFECV Cross-Validation Score (Accuracy)")
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
        id_features = id_features.rename(columns={0: "Feature Ranking"})
        
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(id_features, diag_kind="kde")
        plt.show()

        best_feature = X.columns[rfecv.support_]

        # Step 4: Print results
        print(f"RFECV Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV Best Selected features: {rfecv.support_}", file=file)
        print(f"Best features : {best_feature}", file=file)
        print(f"RFECV Cross-validation scores: {scores}", file=file)
        print(f"RFECV Cross validation Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=kf, scoring='accuracy')
        # Step 6: Print results
        print(f"RFECV with KF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV with KF Selected features: {rfecv.support_}", file=file)
        print(f"RFECV with KF Cross-validation scores: {scores}", file=file)
        print(f"RFECV with KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV with KF RFECV ranking: {rfecv.ranking_}", file=file)
        
        importance = np.absolute(rfecv.estimator_.feature_importances_)
        print(f"Feature importance: {importance}", file,file)
        
        print("Selected Features:", selected_features)
        
        # Now use the RFECV features as input into the DT with GNB 
        print("Evaluating the DT Classification Model")

        # Transform the dataset to include only selected features
        X_selected = rfecv.transform(X)
        
        # Split the data for the DT and GNB models 
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y)
       
        # Train the Decision Tree Classifier on the selected features
        start_time = time.time()  
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        
        dt_time = end_time - start_time
        print(f"DT training time: {dt_time}", file=file)
        
        # Plot the tree
        #plt.figure(figsize=(10, 8))
        #plot_tree(dt_model, filled=True)
        #plt.show()

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
        
        # Generate and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=dt_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
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
        
        dt_predictions = dt_model.predict(X_test)
        
        # Evaluate Decision Tree
        print("Decision Tree Classifier:", file=file)
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}", file=file)
        dt_performance = classification_report(y_test, dt_predictions, zero_division=1.0)
        print(f"\nDT Classification performance: \n {dt_performance}", file=file)

        fOne = cross_val_score(dt_model, X_train, y_train, cv=5)

        # We print the F1 score here
        print("Average Decision Tree F1 score during cross-validation: ", np.mean(fOne))
        #print("Decision Tree f1 scores: ", fOne.mean())

        # Then print the F1 score to the output file
        print(f"Average Decision Tree F1 score during cross-validation: {np.mean(fOne)}", file=file)

        # Retrieve the number of unrelated class categories
        dt_omt_num_classes = len(dt_model.classes_)
        print(f"DT Number of unrelated class categories: {dt_omt_num_classes}", file=file)

        # Retrieve the number of class categories
        dt_num_classes = dt_model.n_classes_
        print(f"DT Number of class categories: {dt_num_classes}", file=file)

        print(f"\nDT Classification performance: \n {dt_performance}", file=file)
        
        # Gaussian Naive Bayes Classifier
        gnb_model = GaussianNB()
        start_time = time.time()
        gnb_model.fit(X_train, y_train)
        end_time = time.time()
        
        gnb_time = end_time - start_time
        
        print(f"GNB training time: {gnb_time}", file=file)
        
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

        # Retrieve the number of unrelated class categories
        gnb_num_classes = len(gnb_model.classes_)
        print(f"GNB Number of class categories: {gnb_num_classes}", file=file)
        
        fOne = cross_val_score(gnb_model, X_train, y_train, cv=5)

        # We print the F1 score here
        print("Average GNB F1 score during cross-validation: ", np.mean(fOne))
        print("GNB f1 scores: ", fOne.mean())

        # Then print the F1 score to the output file
        print(f"Average GNB F1 score during cross-validation: {np.mean(fOne)}", file=file)
    
        
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
        gnb_cm = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"\nGNB Classification Report: \n {gnb_cm}", file=file)
        
        # Combine using VotingClassifier
        voting_clf = VotingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], voting='hard')
        votingscore = voting_clf.fit(X_train, y_train)
        print(votingscore.predict(X_selected), file=file)
        
        y_pred = voting_clf.predict(X_test)
        print("\nVotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("\nVotingClassifier DT with GNB confusion matrix: \n", confusion_matrix(y_test, y_pred), file=file)
        
        voting_clf_classification = classification_report(y_test, y_pred, zero_division=1.0)
        print(f"\nVoting Classification Report: \n {voting_clf_classification}", file=file)
        voting_clf_cm = confusion_matrix(y_test, y_pred)
        print(f"\nVoting Classifier Confusion Matrix: \n {voting_clf_cm}", file=file)
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        stacking_clf = StackingClassifier(estimators=[('dt',dt_model), ('gnb', gnb_model)], final_estimator=LogisticRegression())
        stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        
        # Evaluate
        y_pred = stacking_clf.predict(X_test)
        print("StackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("\nStackingClassifier DT with GNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred), file=file)
       
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"StackingClassifier DT with GNB  Number of unrelated class categories: {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"StackingClassifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # Cross-validation scores
        cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
        print(f"Stacking Classifier Cross-Validation Scores: {cv_scores}", file=file)
        print(f"Stacking Classifier Mean CV Accuracy: {cv_scores.mean():.2f}", file=file)
        
        # Detailed classification report
        stacking_clf_performance = classification_report(y_test, y_pred, zero_division=1.0)
        print(f"\nStacking Classification Report: \n {stacking_clf_performance}", file=file)
        
    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
def main():
    print("Run the script")
    feature_selection = RFECV_EXPERIMENT()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output_combined.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        data_scaled = feature_selection.standardize_data(data)
        #feature_selection.extract_features(data_scaled, file)
        feature_selection.extract_features(data, file)
        file.close()
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':
    main()
