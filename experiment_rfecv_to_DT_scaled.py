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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.naive_bayes import GaussianNB #, CategoricalNB
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
        y = le.fit_transform(y)  # Converts categories to integers
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
        selected_features = [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"Selected Features: {selected_features}", file=file)
        
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
        
        print(f"Number of selected features: {num_selected_features}", file=file)
        print(f"Number of omitted features: {num_omitted_features}", file=file)
        print("Number of omitted features: ", num_omitted_features)
        
        # Step 6a: Make predictions on the test set
        y_pred = rfecv.estimator_.predict(features)
        
        # Not helping need to use the other heatmaps
        #ranking = confusion_matrix(y_train, y_pred).ravel()
        #ranking_2d = ranking.reshape(-1,1)
        #Create normalized Confusion Matrix
        #cm_normalized = ranking_2d.astype('float') / ranking_2d.sum(axis=0)[:, np.newaxis]
        #sns.heatmap(cm_normalized, annot=True, linewidths = 0.01)
        
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
        print(f"KF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"KF Selected features: {rfecv.support_}", file=file)
        print(f"KF Cross-validation scores: {scores}", file=file)
        print(f"KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"KF RFECV ranking: {rfecv.ranking_}", file=file)
        
        print("Evaluating the Decision Tree Model")
        
        # Transform the dataset to include only selected features
        X_selected = rfecv.transform(X)
        
        # Split the data for the RF model 
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y)
        
        start_time = time.time()
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"DT Model Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        
        # Plot the decision tree
        #plt.figure(figsize=(10, 6))
        #plot_tree(dt_model, filled=True)
        #plt.show()  
        
        dt_predictions = dt_model.predict(X_test)
        
        y_pred = dt_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1.0)
        recall = recall_score(y_test, y_pred, average='macro')
       
        # Calculate F1 score for each class
        f1_scores = f1_score(y_test, y_pred, average=None)

        # Display F1 scores for each class
        for i, score in enumerate(f1_scores):
            print(f"Decison Tree F1 Score for class {i}: {score}", file=file)

        print("DT Classifier Accuracy:", accuracy, file=file)
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
        print(f"\nGNB Number of class categories: {gnb_num_classes}", file=file)
       
        fOne = cross_val_score(gnb_model, X_train, y_train, cv=5)

        # We print the F1 score here
        print("\nAverage GNB F1 score during cross-validation: ", np.mean(fOne))
        print("GNB f1 scores: ", fOne.mean())

        # Then print the F1 score to the output file
        print(f"\nAverage GNB F1 score during cross-validation: {np.mean(fOne)}", file=file)
    
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
        
        # Combine using VotingClassifier
        start_time = time.time()
        voting_clf = VotingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], voting='soft')
        votingscore = voting_clf.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"\nDT and GNB Voting Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        print(votingscore.predict(X_selected), file=file)
        
        y_pred = voting_clf.predict(X_test)
        print("\nVotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("\nVotingClassifier DT with GNB confusion matrix: \n", confusion_matrix(y_test, y_pred), file=file)
        
        voting_clf_classification = classification_report(y_test, y_pred, zero_division=1.0)
        print(f"\nVoting Classification Report: \n {voting_clf_classification}", file=file)
        voting_clf_cm = confusion_matrix(y_test, y_pred)
        print(f"\nVoting Classifier Confusion Matrix: \n {voting_clf_cm}", file=file)
        
        # Define base models and meta-classifier
        base_models = [
            ('dt', DecisionTreeClassifier()),
            ('gnb', GaussianNB())
            ]
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        meta_classifier = LogisticRegression(max_iter=500, solver="saga", tol=1e-2)
        
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
        
        #stacking_clf = StackingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], final_estimator=LogisticRegression())
        #stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        

        ##stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        end_time = time.time()
        print(f"DT with GNB Stacking Classifier Execution time: {end_time - start_time:.6f} seconds", file=file)
        
        # Best parameters and score
        print(f"\nStacking Classifier Best Parameters: {stacking_clf.best_params_}", file=file)
        print(f"Stacking Classifier Best Accuracy: {stacking_clf.best_score_}", file=file)

        # Perform cross-validation
        cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
        print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", file=file)
        print(f"\nStacking Classifier Mean CV Accuracy: {cv_scores.mean():.2f}", file=file)

        #feature_importances = stacking_clf.named_estimators_['dt'].feature_importances_
        #plt.barh(feature_names, feature_importances)
        #plt.xlabel("Feature Importance")
        #plt.ylabel("Features")
        #plt.title("Feature Importance for Deciscion Tree Classifier")
        #plt.show()

        # Evaluate performance
        y_pred = stacking_clf.predict(X_test)
        print("\nStackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred), file=file)
        print("StackingClassifier DT with GNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred), file=file)
       
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"StackingClassifier DT with GNB  Number of unrelated class categories: \n {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"StackingClassifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # Detailed classification report
        stacking_clf_performance = classification_report(y_test, y_pred, zero_division=1.0)
        print(f"\nStacking Classification  Report: \n{stacking_clf_performance}", file=file)
        
        return X_selected
    
    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
    
def main():
    print("Run the script")
    features = []
    feature_selection = RFECV_EXPERIMENT()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output_combined_updated.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        data_scaled, target_data = feature_selection.standardize_data(data)
        features = feature_selection.extract_features(data_scaled, file)
        file.close()
        with open('C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/feartures_output.txt', 'w') as file:
            for item in features:
                file.write(f"{item}\n")
        file.close()
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':


    main()

    
