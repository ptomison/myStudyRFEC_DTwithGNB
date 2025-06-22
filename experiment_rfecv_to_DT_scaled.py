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
from sklearn.ensemble import StackingClassifier, VotingClassifier #, RandomForestClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #, LinearRegression, 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score, roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import GaussianNB #, CategoricalNB
from yellowbrick.features import FeatureImportances
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, wilcoxon, kruskal
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
    
    def standardize_data(self, data):
        
        print("Standardizing the data")
              
        from scipy import stats
        
        # Useing the R data_preprocessing generated data
        convert_dict = {'time': float, 'Source_address': int, 'Destination_address': int, 'protocol_converted': int, 'Length': int, 'info_converted': int}
        
        names = data.columns 
        data = data.fillna(0)
        data_scaled = data.astype(convert_dict)
        scaler = StandardScaler()
        target = TargetEncoder()
        y_col = "info_converted"
        y = data_scaled[y_col]
        target_data = data_scaled.drop([y_col], axis=1)
        
        data_scaled = scaler.fit_transform(data_scaled)
        target_scaled = target.fit_transform(target_data, y)
        target_scaled = pd.DataFrame(target_scaled)
        data_scaled = pd.DataFrame(data_scaled, columns=names)
        
        # this call does everything the previous lines did
        data_new_scaled = stats.zscore(data)
        return data_new_scaled, target_scaled
                    
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
        
        le = LabelEncoder()
        y = le.fit_transform(y)  # Converts categories to integers to resolve continous problem
        #print(y)
        
        X = data.drop([y_col], axis = 1)
        
        # Retrieve the names of the columns to identify he RFECV features selected
        #feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        feature_names = X.columns
        
        dt_model = DecisionTreeClassifier(random_state=42)
       
        print("Making Data Classification Complete")
   
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model use the fist one with info_converted and the second one with Source_address
        rfecv = RFECV(estimator=dt_model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        #rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=kf, n_jobs=-1)
        
        #y = y.to_frame()
        #y = np.ravel(y)
        #y = pd.DataFrame(y)
        
        # Split the data for the RFECV model 
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        #print(y)
        
        start_time = time.time()
        #features = rfecv.fit(X_train, y_train)
        #features_selected = rfecv.fit_transform(X_train, y_train)
        #features = rfecv.fit(X, y)
        features = rfecv.fit_transform(X, y)
        end_time = time.time()
        
        print(f"RFECV Execution time: {end_time - start_time:.6f} seconds", file=file)
        #print(features)
        
        print("RFECV model fitting complete")
        
        # Get the names of the selected features
        selected_features_names = [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"Selected Features: {selected_features_names}", file=file)
        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        #scores = cross_val_score(rfecv.estimator_, X_train, y_train, cv=cv, scoring='accuracy')
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
        #num_omitted_features = X.shape[1] - features_selected
        num_omitted_features = (~rfecv.support_).sum()
        
        print(f"\nNumber of selected features: {num_selected_features}", file=file)
        print(f"Number of omitted features: {num_omitted_features}", file=file)
        print("Number of omitted features: ", num_omitted_features)
        
        # Step 6a: Make predictions on the test set
        y_pred_rfecv = rfecv.estimator_.predict(features)
        
        # Not helping need to use the other heatmaps
        #ranking = confusion_matrix(y_train, y_pred).ravel()
        #ranking_2d = ranking.reshape(-1,1)
        #Create normalized Confusion Matrix
        #cm_normalized = ranking_2d.astype('float') / ranking_2d.sum(axis=0)[:, np.newaxis]
        #sns.heatmap(cm_normalized, annot=True, linewidths = 0.01)
        
        print(f"\nRFECV Prediction: {y_pred_rfecv}", file=file)
        
        # Plot the RFECV feature data for visualization
        print(f"Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"Best features: {X.columns[rfecv.support_]}", file=file)
        print(f"Original features: {X.columns}", file=file)
        
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
        print(f"\nKF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"KF Selected features: {rfecv.support_}", file=file)
        print(f"KF Cross-validation scores: {scores}", file=file)
        print(f"KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"KF RFECV ranking: {rfecv.ranking_}", file=file)
       
        # Generate and display the RFECV confusion matrix
        cm = confusion_matrix(y, y_pred_rfecv)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfecv.classes_)
        disp.plot(cmap='Oranges')
        print(f"\nRFECV Confusion matrix: {cm}", file=file)

        print("Evaluating the RFECV Features input into the Decision Tree Model")
        
        # Transform the dataset to include only selected features as input into DT classifier
        X_selected = rfecv.transform(X)
        
        # Split the data for the DT model 
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y)
        
        start_time = time.time()
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # Plot the decision tree
        plt.figure(figsize=(10, 6))
        plot_tree(dt_model, filled=True)
        plt.show()  
        
        dt_predictions = dt_model.predict(X_test)
        
        y_pred_dt = dt_model.predict(X_test)
        
        # Generate and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred_dt, labels=dt_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, dt_predictions)

        # Display confusion matrix
        print(f"\nDT Confusion Matrix: {cm}", file=file)
    
        accuracy = accuracy_score(y_test, y_pred_dt)
        precision = precision_score(y_test, y_pred_dt, average='macro', zero_division=1.0)
        recall = recall_score(y_test, y_pred_dt, average='macro')
       
        # Calculate F1 score for each class
        f1_scores = f1_score(y_test, y_pred_dt, average=None)

        # Display F1 scores for each class
        for i, score in enumerate(f1_scores):
            print(f"Decision Tree F1 Score for class {i}: {score}", file=file)

        print("\nDT Classifier Accuracy:", accuracy, file=file)
        print("DT Classifier Precision:", precision, file=file)
        print("DT Classifier Recall:", recall, file=file)
        print("Decision Tree Classifier:", file=file)
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}", file=file)
        
        dt_performance = classification_report(y_test, dt_predictions, zero_division=1.0)
        print(f"\nDT Classification performance: \n {dt_performance}", file=file)

        # Retrieve the number of unrelated class categories
        dt_omt_num_classes = len(dt_model.classes_)
        print(f"DT Number of unrelated class categories: {dt_omt_num_classes}", file=file)
       
        # Retrieve the number of class categories
        dt_num_classes = dt_model.n_classes_
        print(f"DT Number of class categories: {dt_num_classes}", file=file)

        viz = FeatureImportances(dt_model)
        viz.fit(X_train, y_train)
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

        # Predict probabilities for the positive class
        y_scores = dt_model.predict_proba(X_test)[:, 1]
        
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
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"DT ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Step 2: Train a binary decision tree for each class
        classifiers = {}
        for class_label in np.unique(y):
            # Create binary labels for the current class (One-vs-Rest)
            y_train_binary = (y_train == class_label).astype(int)
    
            # Train a decision tree classifier
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train_binary)
            classifiers[class_label] = clf

        # Step 3: Make predictions
        predictions = []
        for class_label, clf in classifiers.items():
            # Predict probabilities for the current class
            pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability of being in the current class
            predictions.append(pred_prob)

        # Combine predictions and determine the final class
        predictions = np.array(predictions).T
        final_predictions = np.argmax(predictions, axis=1)

        # Step 4: Evaluate the model
        accuracy = accuracy_score(y_test, final_predictions)
        print(f"Decision Tree Binary Classification Accuracy: {accuracy:.2f}", file=file)


        # Gaussian Naive Bayes Classifier
        start_time = time.time()
        gnb_model = GaussianNB()
        gnb_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"GNB Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        gnb_predictions = gnb_model.predict(X_test)
        
        # Plot confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, gnb_predictions)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, gnb_predictions)

        # Output the confusion matrix into the file
        print(f"\nGNB Confusion Matrix: {cm}", file=file)
        
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
        print(f"\nGNB Number of class categories: {gnb_num_classes}", file=file)
       
        fOne = cross_val_score(gnb_model, X_train, y_train, cv=5)

        # We print the F1 score here
        print("\nAverage GNB F1 score during cross-validation: ", np.mean(fOne))
        print("GNB f1 scores: ", fOne.mean())

        # Then print the F1 score to the output file
        print(f"\nAverage GNB F1 score during cross-validation: {np.mean(fOne)}", file=file)

        # Generate and print the classification report
        report = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"GNB Classification report: {report}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
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
        print("\nGaussian Naive Bayes Classifier:", file=file)
        print(f"\nGNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}", file=file)
        gnb_performance = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"\nGNB Classification performance: \n {gnb_performance}", file=file)
        
        # Define base models and meta-classifier
        base_models = [
            ('dt', DecisionTreeClassifier()),
            ('gnb', GaussianNB())
            ]
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        meta_classifier = LogisticRegression(max_iter=500, solver="saga", tol=1e-2)
        
        # Combine using VotingClassifier
        start_time = time.time()
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        votingscore = voting_clf.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"\nDT and GNB Voting Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # get the truth lables from the Voting classifier
        voting_predict = votingscore.predict(X_test)
        print((f" {voting_predict} "), file=file)
        
        y_pred_voting = voting_clf.predict(X_test)
        print("\nVoting Classifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_voting), file=file)
        print("\nVoting Classifier DT with GNB confusion matrix: \n", confusion_matrix(y_test, y_pred_voting), file=file)
        
        voting_clf_classification = classification_report(y_test, y_pred_voting, zero_division=1.0)
        print(f"\nVoting Classification Report: \n {voting_clf_classification}", file=file)
        
        # Step 5: Calculate ROC AUC score
        y_score = votingscore.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"Voting Classifer DT with GNB ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
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
        plt.title('Voting Classifier Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
        # Define hyperparameter grid
        #param_grid = {
        #   'dt__criterion': ['gini', 'entropy', 'log_loss'],
        #    'dt__max_depth': [None, 10, 20, 30],
        #    'dt__min_samples_split': [2, 5, 10],
        #    'dt__min_samples_leaf': [1, 2, 4],
        #    'gnb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        #    'final_estimator__C': [0.1, 1, 10]
        #    }
        
        optimal_params = {
            'dt__criterion': ['gini'],       
            'dt__max_depth': [None],
            'dt__min_samples_split': [2],
            'dt__min_samples_leaf': [1],
            'gnb__var_smoothing': [1e-9],
            'final_estimator__C': [1]
            }
                                   
        start_time = time.time()
        stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_classifier)
        # Perform grid search
        stacking_clf = GridSearchCV(estimator=stacking_clf, param_grid=optimal_params, cv=3, scoring='accuracy')
        stacking_clf.fit(X_train, y_train)
        end_time = time.time()
        print(f"DT with GNB Stacking Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # Best parameters and score
        print(f"\nStacking Classifier Best Parameters: {stacking_clf.best_params_}", file=file)
        print(f"Stacking Classifier Best Accuracy: {stacking_clf.best_score_}", file=file)

        # Perform cross-validation
        cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
        print(f"\nStacking Classifier Mean CV Accuracy: {cv_scores.mean():.4f}", file=file)
        print(f"\nStacking Classifier STD Cross-Validation Accuracy: {cv_scores.std():.4f}", file=file)

        # Evaluate performance
        y_pred_stacking = stacking_clf.predict(X_test)
        print("\nStacking Classifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_stacking), file=file)
        print("Stacking Classifier DT with GNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred_stacking), file=file)
       
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"Stacking Classifier DT with GNB  Number of unrelated class categories: {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"\nStacking Classifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # Detailed classification report
        stacking_clf_performance = classification_report(y_test, y_pred_stacking, zero_division=1.0)
        print(f"\nStacking Classification Report: \n{stacking_clf_performance}", file=file)
        
        # Step 5: Calculate ROC AUC score
        y_score = stacking_clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"Stacking Classifer DT with GNB ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
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
        plt.title('Stacking Classifier Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
        return X_selected, selected_features_names, y_pred_rfecv, dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking, X_test
    
    def stop_all(self, file):
        print("\nStopping the RFECV feature extraction and DT with GNB Classification experiment", file=file)
        pass
    
class Data_Analysis:
    
    def init(self, data_scaled, features, selected_features_names):
        self.data_scaled = [0][0]
        self.x = [0][0]
        self.y = [0]
       
    def oneAnova_MFeatures(self, data_scaled, features, selected_features_names, file):  
        source = data_scaled.Source_address
        length = data_scaled.Length
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
                    print(f"F-statistic for {name}, {f_statistic}, P-value: {p_value:.6f}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    stat, p = kruskal(source, data, features)
                    print(f"Kruskal Statistical Results for source and {name}, {stat}, p-value: {p:.6f}", file=file)
                    stat = 0
                    p = 0
                    stat, p = kruskal(length, data, features)
                    print(f"Kruskal Statistical Results for length for {name}, {stat}, p-value: {p:.6f}", file=file)
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
        print(f"P-value: {p_value}", file=file)

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
                    print(f"T-statistic for Source and {name}, T-Stat: {t_stat}, P-value: {p_value}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    t_stat = 0
                    p_value = 0
                    t_stat, p_value = ttest_ind(length,features)
                    print(f"T-statistic for length and {name}, T-Stat: {t_stat}, P-value: {p_value:.6f}", file=file)
                    # Interpretation
                    if p_value < 0.05:
                        print("Significant differences exist between the groups.", file=file)
                    else:
                        print("No significant differences between the groups.", file=file)
                    stat, p = mannwhitneyu(source, features, alternative='two-sided')
                    print(f"Mann-Whitney U Test Statistic for source and {name}, T-Stat: {stat}, p-value: {p:.6f}", file=file) 
                    i = i+1
                    t_stat = 0
                    p_value = 0
                    stat = 0
                    p = 0
                elif (i == index):
                    break
        else:      
            print("No multiple Features identified therefore going to use one feature identified", file=file)
    
    
    def t_TestAnd_WilcoxTest(self, item, data, feature, file):
        if (item == 0):
             data = data.Source_address
        elif (item == 1):
            data = data.Destination_address
        elif (item == 2):
            data = data.protocol_converted
        elif (item == 3):
             data = data.info_converted
                
        print("\nTwo-group t-test for {item}", file=file)
        stat, p = wilcoxon(data, feature)
        print(f"Wilcox Signed Rank Statistical Results: {stat}, p-value: {p:.6f}", file=file)
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
                    data = pd.DataFrame({'x': x, 'y': y})
                    name = selected_features_names[i]
                    #add constant to predictor variables
                    x = sm.add_constant(data['x'])
                    #fit linear regression model
                    model = sm.OLS(data['y'], x).fit()
                    #view model summary
                    ols_results = model.summary()
                    print(f"\nOLS Summary for source and : {name}\n {ols_results}", file=file)
                    data['predicted'] = model.predict(x)
                    #y_pred = model.predict(x)
                    #print("OLS model predicted: ", y_pred)
                    # Plot the data and the regression line
                    plt.figure(figsize=(8, 6))
                    plt.scatter(data['x'], data['y'], label='Observed Data', color='blue', alpha=0.6)
                    plt.plot(data['x'], data['predicted'], label='Regression Line', color='red', linewidth=2)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('Source Address OLS Regression: Observed vs Predicted')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    # Now repeat the linear regression with the Length variable
                    x = length
                    data = pd.DataFrame({'x': x, 'y': y})
                    #add constant to predictor variables
                    x = sm.add_constant(data['x'])
                    #fit linear regression model
                    model = sm.OLS(data['y'], x).fit()
                    #view model summary
                    ols_results = model.summary()
                    print(f"\nOLS Summary for length and : {name}\n {ols_results}", file=file)
                    data['predicted'] = model.predict(x)
                    #y_pred = model.predict(x)
                    #print("OLS model predicted: ", y_pred)
                    # Plot the data and the regression line
                    plt.figure(figsize=(8, 6))
                    plt.scatter(data['x'], data['y'], label='Observed Data', color='blue', alpha=0.6)
                    plt.plot(data['x'], data['predicted'], label='Regression Line', color='red', linewidth=2)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('Length OLS Regression: Observed vs Predicted')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    i = i+1
                elif (i == index):
                    break
    
    def olsTest_SingleFeature(self, data, features, selected_features_names, file):
        # Fit the ordinary least sqaure model
         #define predictor and response variables
        y = pd.DataFrame(features)
        y.columns = selected_features_names
        y = y.iloc[:, :1]
        x = data['Source_address']
        #add constant to predictor variables
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y, x).fit()
        #view model summary
        ols_results = model.summary()
        print(f"\nOLS Summary: \n {ols_results}", file=file)
        y_pred = model.predict(x)
        # Plotting
        plt.scatter(x.iloc[:,:1], y, label='Observed Data', color='blue', alpha=0.6)
        plt.plot(x, y_pred, label='Regression Line', color='red', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Length OLS Regression: Observed vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.show()
                
           
       
def main():
    print("Run the script")
    features = []
    dt_predictions = [0]
    gnb_predictions = [0]
    y_pred_voting = [0]
    y_pred_stacking = [0]
    y_pred_rfecv = [0]
    testing_data = [0]
    selected_features_names = [0];
    feature_selection = RFECV_EXPERIMENT()
    data_analysis = Data_Analysis()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/rfecv_output_dt_withgnbcombined.txt", "a")
        # for debuging
        #file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        data_scaled, target_data = feature_selection.standardize_data(data)
        features, selected_features_names, y_pred_rfecv, dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking, testing_data = feature_selection.extract_features(data_scaled, file)
        with open('C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/Combined_feartures_output.txt', 'a') as feature_file:
            for item in features:
                feature_file.write(f"{item}\n")
        feature_file.close()
        
        data_analysis.oneAnova_MFeatures(data_scaled, features, selected_features_names, file)
        data_analysis.tTest_NFeatures(data_scaled, features, selected_features_names, file)
        
        # see if the models predicitions has statstical significance following normality
        print("One ANOVA Statistical Significance for RFECV features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_rfecv, file)
        print("One ANOVA Statistical Significance for Decision Tree features identified", file=file)
        data_analysis.oneAnova(data_scaled, dt_predictions, file)
        print("One ANOVA Statistical Significance for GNB features identified", file=file)
        data_analysis.oneAnova(data_scaled, gnb_predictions, file)
        print("One ANOVA Statistical Significance for DT with GNB Voting stacking features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_voting, file)
        print("One ANOVA Statistical Significance for DT with GNB Combined features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_stacking, file)
        
        for i in range(4):
            data_analysis.t_TestAnd_WilcoxTest(i, data_scaled, y_pred_rfecv, file) 

        print("\nTwo-group t-test for Source Address and DT with GNB features identified", file=file)
        t_stat, p_value = ttest_ind(testing_data[:, 0], y_pred_stacking)
        if p_value < 0.05:
              print("Significant differences exist between the groups.", file=file)
        else:
              print("No significant differences between the groups.", file=file)
        stat, p = wilcoxon(testing_data[:, 0], y_pred_stacking)
        print(f"Wilcox T-statistic: {t_stat}, P-value: {p_value:.6f}", file=file)
        
        index = len(selected_features_names)
        if (index > 1):
            data_analysis.olsTest_NFeature(data_scaled, features, selected_features_names, file)
        else:
            data_analysis.olsTest_SingleFeature(data_scaled, features, selected_features_names, file)
           
        
        feature_selection.stop_all(file)
        file.close()
        
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':

    main()
