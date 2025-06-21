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

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score, GridSearchCV
#from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #, LinearRegression, 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler #, OneHotEncoder, LableEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from yellowbrick.features import FeatureImportances
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, wilcoxon, kruskal, chi2_contingency
import statsmodels.api as sm


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
    
    # Gaussian function
    def Gaussian_func(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    def standardize_data(self, data):
        
        print("Standardizing the data")
              
        from scipy import stats
        
        # Useing the R data_preprocessing generated data
        convert_dict = {'time': float, 'Source_address': int, 'Destination_address': int, 'protocol_converted': int, 'Length': int, 'info_converted': int}
        
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
        y = data[y_col]
        
        X = data.drop([y_col], axis = 1)
        
        # Retrieve the names of the columns to identify he RFECV features selected
        feature_names = X.columns
        
        model = RandomForestClassifier(random_state=42)
       
        print("Making Data Classification Complete")
   
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model use the fist one with info_converted and the second one with Source_address
        rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        #rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=kf, n_jobs=-1)
        
        start_time = time.time()
        features = rfecv.fit_transform(X, y)
        end_time = time.time()
        
        print(f"RFECV Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        print("RFECV model fitting complete")
        
        # Get the names of the selected features
        selected_features_names= [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"Selected Features: {selected_features_names}", file=file)
        
        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=cv, scoring='accuracy')
        
        best_feature = X.columns[rfecv.support_]
        
        # Step 6: Print results
        print(f"RFECV Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV Best Selected features: {rfecv.support_}", file=file)
        print(f"Best features : {best_feature}", file=file)
        print(f"RFECV Cross-validation scores: {scores}", file=file)
        print(f"RFECV Cross validation Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Retrieve the number of selected features
        num_selected_features = rfecv.n_features_
        
        # Calculate the number of omitted features
        num_omitted_features = (~rfecv.support_).sum()
        
        print(f"Number of selected features: {num_selected_features}", file=file)
        print(f"Number of omitted features: {num_omitted_features}", file=file)
        print("Number of omitted features: ", num_omitted_features)
        
        # Step 6a: Make predictions on the test set
        y_pred = rfecv.estimator_.predict(features)
        
        print(f"RFECV Prediction: {y_pred}", file=file)
        
        # Plot the RFECV feature data for visualization
        print('Optimal number of features :', rfecv.n_features_)
        print('Best features :', X.columns[rfecv.support_])
        print('Original features :', X.columns)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score std test score")
        plt.plot(range(1, len(rfecv.cv_results_['std_test_score']) + 1), rfecv.cv_results_['std_test_score'])
        plt.show()
        
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
        id_features = id_features.rename(columns={0: "Feature Ranking"})
        
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(id_features, diag_kind="kde")
        plt.show()
        
        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=kf, scoring='accuracy')
        # Step 6: Print results
        print(f"RFECV with KF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV with KF Selected features: {rfecv.support_}", file=file)
        print(f"RFECV with KF Cross-validation scores: {scores}", file=file)
        print(f"RFECV with KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"RFECV with KF RFECV ranking: {rfecv.ranking_}", file=file)
        
        print("Evaluating the RF Model")
        
        # Split the data for the RF model 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"RF Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        y_pred_rfecv = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred_rfecv)
        precision = precision_score(y_test, y_pred_rfecv, average='macro')
        recall = recall_score(y_test, y_pred_rfecv, average='macro')
       
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

        plt.title("RF Feature Importances", size=20, loc="left", y=1.04, weight="bold")
        plt.ylabel("Importance")
        plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90, size=12)
        plt.show()
        
        return features, selected_features_names, y_pred_rfecv

    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
class DT_with_GNB:
    def init(self, data):
        self.data = data
        self.x = [0][0]
        self.y = [0]
        
    def decicsionTree(self, data, file, features):
        
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

        dt_model = DecisionTreeClassifier(random_state=42)

        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Decision Tree Classifier
        start_time = time.time()
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT Classifier Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        dt_predictions = dt_model.predict(X_test)
        
        fOne = cross_val_score(dt_model, X_train, y_train, cv=5)

        # We print the F1 score here
        print("Average Decision Tree F1 score during cross-validation: ", np.mean(fOne))
        print("Decision Tree f1 scores: ", fOne.mean())

        # Then print the F1 score to the output file
        print(f"Average Decision Tree F1 score during cross-validation: {np.mean(fOne)}", file=file)
    
        # Retrieve the number of unrelated class categories
        dt_omt_num_classes = len(dt_model.classes_)
        print(f"DT Number of unrelated class categories: {dt_omt_num_classes}", file=file)
       
        # Retrieve the number of class categories
        dt_num_classes = dt_model.n_classes_
        print(f"DT Number of class categories: {dt_num_classes}", file=file)

        # Plot the decision tree
        #plt.figure(figsize=(10, 6))
        #plot_tree(dt_model, filled=True)
        #plt.show()    
        
        # Generate and plot the confusion matrix
        cm = confusion_matrix(y_test, dt_predictions, labels=dt_model.classes_)
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
        
        # Evaluate Decision Tree
        print("Decision Tree Classifier:", file=file)
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}", file=file)
        dt_performance = classification_report(y_test, dt_predictions, zero_division=1.0)
        print(f"\nDT Classification performance: \n {dt_performance}", file=file)

        # Step 1: Visualize the original data distribution
        plt.hist(data, bins=30, density=True, alpha=0.6, color={'blue', 'red', 'yellow', 'green', 'purple', 'orange'}, label='Original Data')
        plt.title("Original Network Flow Distribution")
        plt.xlabel("Flow Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        
        # Predict probabilities for the positive class
        y_scores = dt_model.predict_proba(X_test)[:, 1]
        #y_scores = dt_model.predict_proba(X_test)
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DT Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        y_score = dt_model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        except:
            roc_auc = 999
        print(f"DT ROC AUC Score: {roc_auc:.2f}", file=file)

        # Gaussian Naive Bayes Classifier
        # data does not follow gaussian distribution
        start_time = time.time()

        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        gnb_model = GridSearchCV(GaussianNB(), param_grid, cv=cv)
        gnb_model.fit(X_train, y_train)
        
        #gnb_model = GaussianNB()
        #gnb_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"GNB Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        gnb_predictions = gnb_model.predict(X_test)
        
        # Plot confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, gnb_predictions)
        
        
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
  
        # Predict probabilities for the positive class
        y_scores = gnb_model.predict_proba(X_test)[:, 1]
            
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('GNB Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        y_score = gnb_model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"GNB ROC AUC Score: {roc_auc:.2f}", file=file)

        num_columns = X_train.shape[1]
        print(f"Number of columns in X_train: {num_columns}", file=file)
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}", file=file)
        
        num_columns = X_test.shape[1]
        print(f"Number of columns in X_test: {num_columns}", file=file)
        num_columns = X_train.shape[0]
        print(f"Number of rows in X_train: {num_columns}", file=file)
        
        # Evaluate Gaussian Naive Bayes
        print("Gaussian Naive Bayes Classifier:", file=file)
        print(f"GNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}", file=file)
        gnb_performance = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"\nGNB Classification performance: \n {gnb_performance}", file=file)
        
        # Define base models and meta-classifier
        base_models = [
            ('dt', DecisionTreeClassifier()),
            ('gnb', GaussianNB())
            ]
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        meta_classifier = LogisticRegression(max_iter=500, solver="saga", tol=1e-2)
        
        # Define hyperparameter grid
        param_grid = {
            'dt__criterion': ['gini', 'entropy', 'log_loss'],
            'dt__max_depth': [None, 10, 20, 30],
            'dt__min_samples_split': [2, 5, 10],
            'dt__min_samples_leaf': [1, 2, 4],
            'gnb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            'final_estimator__C': [0.1, 1, 10]
            }
        
        optimal_params = {
            'dt__criterion': ['gini'],       
            'dt__max_depth': [None],
            'dt__min_samples_split': [2],
            'dt__min_samples_leaf': [1],
            'gnb__var_smoothing': [1e-9],
            'final_estimator__C': [1]
            }
                                   
       
        # Combine using VotingClassifier
        start_time = time.time()
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        votingscore = voting_clf.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT and GNB Voting Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        print(votingscore.predict(X), file=file)
        
        y_pred_voting = voting_clf.predict(X_test)
        print("\nVotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_voting), file=file)
        print("\nVotingClassifier DT with GNB confusion matrix: \n", confusion_matrix(y_test, y_pred_voting), file=file)
        
        voting_clf_classification = classification_report(y_test, y_pred_voting, zero_division=1.0)
        print(f"\nVoting Classification Report: \n {voting_clf_classification}", file=file)
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        start_time = time.time()
        stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_classifier)
        stacking_clf = GridSearchCV(estimator=stacking_clf, param_grid=optimal_params, cv=3, scoring='accuracy')
        #stacking_clf = GridSearchCV(estimator=stacking_clf, param_grid=optimal_params, cv=3, scoring='accuracy')
        stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        end_time = time.time()
        print(f"DT with GNB Stacking Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # Evaluate performance
        y_pred_stacking = stacking_clf.predict(X_test)
        print("\nStackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_stacking), file=file)
        print("StackingClassifier DT with GNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred_stacking), file=file)
       
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"StackingClassifier DT with GNB  Number of unrelated class categories: \n {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"StackingClassifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # Cross-validation scores
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Adjust n_splits to match the smallest class size
        cv_scores = cross_val_score(stacking_clf, X, y, cv=skf, scoring='accuracy')
        print(f"\nStacking Classifier Cross-Validation Scores: {cv_scores}", file=file)
        print(f"\nStacking Classifier Mean CV Accuracy: {cv_scores.mean():.2f}", file=file)
        
        # Detailed classification report
        stacking_clf_performance = classification_report(y_test, y_pred_stacking, zero_division=1.0)
        print(f"\nStacking Classification  Report: \n{stacking_clf_performance}", file=file)

        return dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking 
    
class Data_Analysis:
    
    def init(self, data_scaled, features, selected_features_names):
        self.data_scaled = [0][0]
        self.x = [0][0]
        self.y = [0]
       
    def oneAnova_MFeatures(self, data_scaled, features, selected_features_names, file):  
        source = data_scaled.Source_address
        feature = pd.DataFrame(features, columns = selected_features_names)
        index = len(selected_features_names)
        i = 0
        if (index > 1):
            for column in feature.columns:
                # Perform one-way ANOVA if the features were selected
                if (column == selected_features_names[i]):
                    features = feature[column]
                    data = data_scaled[column]
                    name = selected_features_names[i]
                    f_statistic, p_value = f_oneway(source, data,features)
                    print(f"F-statistic for: {name}, {f_statistic}", file=file)
                    print(f"P-value: {p_value}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    stat, p = kruskal(source, data, features)
                    print(f"Kruskal Statistical Results for source and {name}, {stat}, p-value: {p:.6f}", file=file)
                    i = i+1
                    f_statistic = 0
                    p_value = 0
                    stat = 0
                    p = 0
                elif (i == index):
                    break
        else:      
            print("No multiple Features identified therefore going to use one feature identified", file=file)
        
    def oneAnova(self, data_scaled, features, file):  
        source = data_scaled.Source_address
        destination = data_scaled.Destination_address
        protocol = data_scaled.Destination_address
        length = data_scaled.Length
        f_statistic, p_value = f_oneway(source, destination, protocol, length, features)
        # Output results
        print(f"F-statistic: {f_statistic}",file=file)
        print(f"P-value: {p_value:.6f}", file=file)

        # Interpretation
        if p_value < 0.05:
            print("Significant differences exist between the groups.", file=file)
        else:
            print("No significant differences between the groups.", file=file)
    
    def mannWhitneyU(self, data_scaled, features, file):
        source = data_scaled.Source_address
        destination = data_scaled.Destination_address
        protocol = data_scaled.Destination_address
        length = data_scaled.Length
        
        stat, p = mannwhitneyu(source, features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statistic on Source and Features: {stat}, p-value: {p:.6f}")
        stat, p = mannwhitneyu(destination, features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statisticon on destination and feature: {stat}, p-value: {p:.6f}")
        stat, p = mannwhitneyu(protocol, features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statistic on protocol and features: {stat}, p-value: {p:.6f}")
        stat, p = mannwhitneyu(length ,features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statistic on length and features: {stat}, p-value: {p:.6f}")
    
    def tTest_NFeatures(self, data_scaled, features, selected_features_names, file):
        feature = pd.DataFrame(features, columns = selected_features_names)
        source = data_scaled.Source_address
        length = data_scaled.Length
        # Perform two-sample t-test
        index = len(selected_features_names)
        i = 0
        if (index > 1):
            for column in feature.columns:
                # Perform two sample t-test on the features that were selected 
                if (column == selected_features_names[i]):
                    features = feature[column]
                    name = selected_features_names[i]
                    t_stat, p_value = ttest_ind(source,features)
                    print(f"T-statistic for Source and name: {name}, {t_stat}, P-value: {p_value:.6f}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    t_stat = 0
                    p_value = 0
                    t_stat, p_value = ttest_ind(length,features)
                    print(f"T-statistic for length and name: {name}, {t_stat}, P-value: {p_value:.6f}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    stat, p = mannwhitneyu(source, features, alternative='two-sided')
                    print(f"Mann-Whitney U Test Statistic for source and name: {stat}, p-value: {p}", file=file) 
                    i = i+1
                    t_stat = 0
                    p_value = 0
                    stat = 0
                    p = 0
                elif (i == index):
                    break
        else:      
            print("No multiple Features identified therefore going to use one feature identified", file=file)
    
    
    def tTest(self, data, feature, file):
        t_stat, p_value = ttest_ind(data,feature)
        print(f"T-statistic: {t_stat}, P-value: {p_value:.6f}", file=file)
        if p_value < 0.05:
              print("Significant differences exist between the groups.", file=file)
        else:
              print("No significant differences between the groups.", file=file)
        
    def olsTest_NFeature(self, data, features, selected_features_names, file):
        feature = pd.DataFrame(features, columns = selected_features_names)
        source = data.Source_address
        length = data.Length
        
        # Perform ols test
        index = len(selected_features_names)
        i = 0
        if (index > 1):
            for column in feature.columns:
                # Perform two sample t-test on the features that were selected 
                if (column == selected_features_names[i]):
                    y = feature[column]
                    #x = data[column]
                    x = source
                    name = selected_features_names[i]
                    #add constant to predictor variables
                    x = sm.add_constant(x)
                    #fit linear regression model
                    model = sm.OLS(y, x)
                    results = model.fit(cov_type='HC3')
                    #view model summary
                    ols_results = results.summary()
                    print(f"\nOLS Summary for source and: {name}\n {ols_results}", file=file)
                    x = length
                    #add constant to predictor variables
                    x = sm.add_constant(x)
                    #fit linear regression model
                    model = sm.OLS(y, x)
                    results = model.fit(cov_type='HC3')
                    #view model summary
                    ols_results = results.summary()
                    print(f"\nOLS Summary for length and: {name}\n {ols_results}", file=file)
                    i = i+1
                elif (i == index):
                    break
        

def main():
    print("Run the script")
    features = [0];
    dt_predictions = [0]
    gnb_predictions = [0]
    y_pred_voting = [0]
    y_pred_stacking = [0]
    y_pred_rfecv = [0]
    selected_features_names = [0];
    feature_selection = RFECV_EXPERIMENT()
    attack_classification = DT_with_GNB()
    data_analysis = Data_Analysis()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/rfecv_unscaled_uncompined_output.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        features, selected_features_names, y_pred_rfecv = feature_selection.extract_features(data, file)
        with open('C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/rfecv_unscaled_feartures_output.txt', 'w') as feature_file:
            for item in features:
                feature_file.write(f"{item}\n")
        feature_file.close()
        
        dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking = attack_classification.decicsionTree(data, file, features)
        
        data_analysis.oneAnova_MFeatures(data, features, selected_features_names, file)
        data_analysis.tTest_NFeatures(data, features, selected_features_names, file)
        
        # see if the models predicitions has statstical significance
        print("One ANOVA Statistical Significance for RFECV features identified", file=file)
        data_analysis.oneAnova(data, y_pred_rfecv, file)
        print("One ANOVA Statistical Significance for Decision Tree features identified", file=file)
        data_analysis.oneAnova(data, dt_predictions, file)
        print("One ANOVA Statistical Significance for GNB features identified", file=file)
        data_analysis.oneAnova(data, gnb_predictions, file)
        print("One ANOVA Statistical Significance for DT with GNB Voting stacking features identified", file=file)
        data_analysis.oneAnova(data, y_pred_voting, file)
        print("One ANOVA Statistical Significance for DT with GNB Combined features identified", file=file)
        data_analysis.oneAnova(data, y_pred_stacking, file)
        
        print("Two-group t-test for Source Address and RFECV features identified", file=file)
        data_analysis.tTest(data.Source_address, y_pred_rfecv, file )
        #stat, p = wilcoxon(data.Source_address, y_pred_rfecv)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
        
        print("Two-group t-test for Destination Address and RFECV features identified", file=file)
        data_analysis.tTest(data.Destination_address, y_pred_rfecv, file )
        #stat, p = wilcoxon(data.Destination_address, y_pred_rfecv)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        print("Two-group t-test for Protocol and RFECV features identified", file=file)
        data_analysis.tTest(data.protocol_converted, y_pred_rfecv, file )
        #stat, p = wilcoxon(data.protocol_converted, y_pred_rfecv)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}", file =file)
       
        print("Two-group t-test for Length and RFECV features identified", file=file)
        data_analysis.tTest(data.Length, y_pred_rfecv, file )
        #stat, p = wilcoxon(data.Length, y_pred_rfecv)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        print("Two-group t-test for Source Address and DT with GNB features identified", file=file)
        data_analysis.tTest(data.Source_address, y_pred_stacking, file )
        #stat, p = wilcoxon(data.Source_address, y_pred_stacking)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        print("Two-group t-test for Destination Address and DT with GNB features identified", file=file)
        data_analysis.tTest(data.Destination_address, y_pred_stacking, file )
        #stat, p = wilcoxon(data.Destination_address, y_pred_stacking)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        print("Two-group t-test for Protocol and DT with GNB features identified", file=file)
        data_analysis.tTest(data.protocol_converted, y_pred_stacking, file )
        #stat, p = wilcoxon(data.protocol_converted, y_pred_stacking)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        print("Two-group t-test for Length and DT with GNB features identified", file=file)
        data_analysis.tTest(data.Length, y_pred_stacking, file )
        #stat, p = wilcoxon(data.Length, y_pred_stacking)
        #print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}")
       
        index = len(selected_features_names)
        if (index > 1):
            data_analysis.olsTest_NFeature(data, features, selected_features_names, file)
        else:
            # Fit the ordinary least sqaure model
            #define predictor and response variables
            y = pd.DataFrame(features)
            y = y.iloc[:, :1]
            x = data['Source_address']
            #add constant to predictor variables
            x = sm.add_constant(x)
            #fit linear regression model
            model = sm.OLS(y, x)
            results = model.fit(cov_type='HC3')
            #view model summary
            ols_results = results.summary()
            print(f"\nOLS Summary: \n {ols_results}", file=file)

        
        file.close()
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':


    main()
