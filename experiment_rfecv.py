# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:50:53 2025

@author: pauli
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
#import csv
#import sys
#import re
import numpy as np
import seaborn as sns
#import shap

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score
#from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #, LinearRegression, 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
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
        data_path = os.path.join(base_dir, "/Phd/DIS9903A/ConductExperiment/DataCollection/Source")
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
        
        # Useing the R data_preprocessing generated data
        #convert_dict = {'time': float, 'Source_address': int, 'Destination_address': int, 'protocol': int, 'Length': int, 'info_converted': int}
        #convert_dict = {'time': float, 'Source_address': int, 'Destination_address': int, 'protocol_converted': int, 'Length': int, 'info_converted': int}
        # Using the IMB SPSS data preprocessing generated data
        convert_dict = {'Time': float, 'Source_address': int, 'Destination_address': int, 'protocol_convert': int, 'Length': int, 'info_converted': int}
        
        names = data.columns 
        data = data.fillna(0)
        data_scaled = data.astype(convert_dict)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_scaled)
        data_scaled = pd.DataFrame(data_scaled, columns=names)
        # this call does everything the previous lines did
        data_new_scaled = stats.zscore(data)
        return data_new_scaled
                    
    def extract_features(self, data, file):
        print("Extracting the features using RFECV")
        # Remove the columns with the number listing from the data preprocessing step
        y_col = "Unnamed: 0"
        
        y = data[y_col]
        
        data = data.drop([y_col], axis = 1)
        
        y1_col = "number"
        
        data = data.drop([y1_col], axis = 1)
        
        y_col = "info_converted"
        #y_col = "Source_address"
        #y1_col = "protocol_converted" 
        
        #y = data[[y_col, y1_col]]
        y = data[y_col]
        #print(y)
        
        X = data.drop([y_col], axis = 1)
        
        # Retrieve the names of the columns to identify he RFECV features selected
        #feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        feature_names = X.columns
        
        model = RandomForestClassifier(random_state=42)
       
        print("Making Data Classification Complete")
   
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model use the fist one with info_converted and the second one with Source_address
        rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        #rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=kf, n_jobs=-1)
        
        #y = y.to_frame()
        #y = np.ravel(y)
        #y = pd.DataFrame(y)
        
        # Split the data for the RFECV model 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        #print(y)
        
        start_time = time.time()
        features = rfecv.fit(X_train, y_train)
        features_selected = rfecv.fit_transform(X_train, y_train)
        end_time = time.time()
        
        print(f"RFECV Execution time: {end_time - start_time:.6f} seconds", file=file)
        #print(features)
        
        print("RFECV model fitting complete")
        
        # Get the names of the selected features
        selected_features = [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"Selected Features: {selected_features}", file=file)
        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Step 6: Print results
        print(f"RFECV Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV Selected features: {rfecv.support_}", file=file)
        print(f"RFECV Cross-validation scores: {scores}", file=file)
        print(f"RFECV Cross validation Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Retrieve the number of selected features
        num_selected_features = rfecv.n_features_
        
        # Calculate the number of omitted features
        num_omitted_features = X.shape[1] - features_selected
        
        print(f"Number of selected features: {num_selected_features}", file=file)
        print(f"Number of omitted features: {num_omitted_features}", file=file)
        
        # Step 6a: Make predictions on the test set
        y_pred = rfecv.estimator_.predict(features_selected)
        
        # Not helping need to use the other heatmaps
        #ranking = confusion_matrix(y_train, y_pred).ravel()
        #ranking_2d = ranking.reshape(-1,1)
        #Create normalized Confusion Matrix
        #cm_normalized = ranking_2d.astype('float') / ranking_2d.sum(axis=0)[:, np.newaxis]
        #sns.heatmap(cm_normalized, annot=True, linewidths = 0.01)
        
        print(f"RFECV Prediction: {y_pred}", file=file)
        
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
        
        selected_features = rfecv.support_

        # Heatmap of selected features
        sns.heatmap([selected_features], cmap="coolwarm", cbar=False, xticklabels=range(X.shape[1]))
        plt.xlabel("Feature Index")
        plt.title("Selected Features (1 = Selected, 0 = Not Selected)")
        plt.show()
        
        # Count duplicate features
        original_features = pd.DataFrame(X).columns
        selected_feature_names = original_features[selected_features]
        duplicates = len(selected_feature_names) - len(set(selected_feature_names))

        print(f"RFECV Number of duplicate features: {duplicates}", file=file)

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
        
        print("Evaluating the RF Model")
        
        # Split the data for the RF model 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"RF Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
       
        print("RF Classifier Accuracy:", accuracy, file=file)
        print("RF Classifier Precision:", precision, file=file)
        print("RF Classifier Recall:", recall, file=file)
        
        viz = FeatureImportances(model)
        viz.fit(X, y)
        viz.show()
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        bars = plt.bar(range(X.shape[1]), importances[indices], edgecolor="#008031", linewidth=1)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", size=8)

        plt.title("Feature Importances", size=20, loc="left", y=1.04, weight="bold")
        plt.ylabel("Importance")
        plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90, size=12)
        plt.show()
        
        return features

    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
class DT_with_GNB:
    def init(self, data):
        self.data = data
        self.x = [0][0]
        self.y = [0]
        
    def decicsionTree(self, data, file):
        
        # Remove the columns with the number listing from the data preprocessing step
        y_col = "Unnamed: 0"
        
        y = data[y_col]
        
        data = data.drop([y_col], axis = 1)
        
        #y1_col = "No"
        y1_col = "number"
        
        data = data.drop([y1_col], axis = 1)
        
        #y_col = 'Source_address'
        #y_col = 'protocol_converted'
        y_col = "info_converted"
        y = data[y_col]
        
        X = data.drop([y_col], axis = 1)
        
        y = y.to_frame()
        
        y = np.ravel(y)
        #print(y)
        
        #print(type(X))
        #print(type(y))
       
        # Random forest classifier, to classify features into big or small
        #model = RandomForestClassifier()
        
        # Find the number of members in the least-populated class, THIS IS THE LINE WHERE THE MAGIC HAPPENS :)
        #leastPopulated = [x for d in set(list(y)) for x in list(y) if x == d].count(min([x for d in set(list(y)) for x in list(y) if x == d], key=[x for d in set(list(y)) for x in list(y) if x == d].count))
        #fOne = cross_val_score(model, X, y, cv=leastPopulated, scoring='f1_weighted')

        # We print the F1 score here
        #print(f"Average F1 score during cross-validation: {np.mean(fOne)}", file=file)

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Decision Tree Classifier
        start_time = time.time()
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT Classifier Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        dt_predictions = dt_model.predict(X_test)

       # fOne = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='f1_weighted')
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

        # Plot the decision tree
        #plt.figure(figsize=(12, 8))
        #plot_tree(dt_model, filled=True)
        #plt.show()    
        
        y_pred = dt_model.predict(X_test)
        
        # Generate and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=dt_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': dt_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
        plt.title('DT Classification Model Feature Importance')
        plt.show()
        
        # Gaussian Naive Bayes Classifier
        start_time = time.time()
        gnb_model = GaussianNB()
        gnb_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"GNB Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
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
       
        num_columns = X_train.shape[1]
        print(f"Number of columns in X_train: {num_columns}")
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}")
        
        num_columns = X_test.shape[1]
        print(f"Number of columns in X_test: {num_columns}")
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}")
        
        # Evaluate Decision Tree
        print("Decision Tree Classifier:", file=file)
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}", file=file)
        dt_performance = classification_report(y_test, dt_predictions, zero_division=1.0)
        print(f"DT Classification performance: {dt_performance}", file=file)

        # Evaluate Gaussian Naive Bayes
        print("Gaussian Naive Bayes Classifier:", file=file)
        print(f"GNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}", file=file)
        gnb_performance = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"GNB Classification performance: {gnb_performance}", file=file)
        
        # Combine using VotingClassifier
        start_time = time.time()
        voting_clf = VotingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], voting='hard')
        votingscore = voting_clf.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT and GNB Voting Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        print(votingscore.predict(X), file=file)
        
        y_pred = voting_clf.predict(X_test)
        print("VotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("VotingClassifier DT with GNB confusion matris: ", confusion_matrix(y_test, y_pred), file=file)
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        start_time = time.time()
        stacking_clf = StackingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], final_estimator=LogisticRegression())
        stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        end_time = time.time()
        print(f"DT with GNB Stacking Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # Evaluate
        y_pred = stacking_clf.predict(X_test)
        print("StackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("StackingClassifier DT with GNB Confusion Matrix:", confusion_matrix(y_test, y_pred), file=file)
       
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"StackingClassifier DT with GNB  Number of unrelated class categories: {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"StackingClassifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # try to output GNB Decision boundary
        #element_X = X_train.to_numpy()
        
        # Define the range of the plot
        #x_min, x_max = element_X[:, 0].min() - 1, element_X[:, 0].max() + 1
        #_min, y_max = element_X[:, 1].min() - 1, element_X[:, 1].max() + 1

        # Generate a grid of points for the decision boundary
        #x = np.linspace(x_min, x_max, 500)
        #y = np.linspace(y_min, y_max, 500)
        
        #xx, yy = np.meshgrid(x, y)
        #grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        
        # Predict probabilities for the grid points
        #probs = gnb_model.predict_proba(grid_points)[:, 1]
        #probs = probs.reshape(xx.shape)

        # Plot the decision boundary
        #plt.contour(xx, yy, probs, levels=[0.5], colors='red', linewidths=2)
        #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k')
        #plt.title("Gaussian Naive Bayes Decision Boundary")
        #plt.xlabel("Feature 1")
        #plt.ylabel("Feature 2")
        #plt.show()

       
    
def main():
    print("Run the script")
    features = [0];
    feature_selection = RFECV_EXPERIMENT()
    attack_classification = DT_with_GNB()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        #data_scaled = feature_selection.standardize_data(data)
        #features = feature_selection.extract_features(data_scaled)
        features = feature_selection.extract_features(data, file)
        print(f" RFECV features identified : {features}", file=file)
        attack_classification.decicsionTree(data, file)
        file.close()
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':


    main()
