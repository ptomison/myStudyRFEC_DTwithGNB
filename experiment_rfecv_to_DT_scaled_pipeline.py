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
import pingouin as pg
import statistics as statistic

# Set global font properties
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.size'] = 14    


from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV, mutual_info_classif, chi2
from sklearn.ensemble import StackingClassifier, VotingClassifier #, RandomForestClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.naive_bayes import GaussianNB 
from yellowbrick.features import FeatureImportances
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, wilcoxon, kruskal
from scipy.stats import norm, pearsonr, spearmanr, ttest_rel, tukey_hsd
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance

class RFECV_EXPERIMENT:
    
    def init(self):
        self.data = [0][0]
        self.x = [0][0]
        self.y = [0]
        
        
    def open_file(self):
        base_dir = os.path.dirname(__file__)  # Directory of the current script
    
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
        #convert_dict = {'time': float, 'Source_address': int, 'Destination_address': int, 'protocol_converted': int, 'Length': int, 'info_converted': int, 'port_converted': int, 'port_burst':int}
        
        target_scaled = 0
        names = data.columns 
        data = data.fillna(0)
        #data_scaled = data.astype(convert_dict)
        data_scaled = data
        
        scaler = StandardScaler()
        
        data_scaled = scaler.fit_transform(data_scaled)
        data_scaled = pd.DataFrame(data_scaled, columns=names)
        
        # this call does everything the previous lines did
        data_new_scaled = stats.zscore(data)
        # shifted_data = data_new_scaled - data_new_scaled.min()
        # data_scaled = pd.DataFrame(shifted_data, columns=names)
        
        return data_scaled, target_scaled, data_new_scaled
                    
    def extract_features(self, data, file):
        
        cm_metrics = ComputeModelMetrics()
        
        print("Extracting the features using RFECV")
        # Remove the columns with the number listing from the data preprocessing step
        y_col = "Unnamed: 0"
        
        y = data[y_col]
        
        data = data.drop([y_col], axis = 1)
        
        y1_col = "number"
        
        data = data.drop([y1_col], axis = 1)
        
        # y1_col = "time"
        
        # data = data.drop([y1_col], axis = 1)
        
        y_col = "info_converted"
        #y_col = "Flags"
        y = data[y_col]
        
        le = LabelEncoder()
        y = le.fit_transform(y)  # Converts categories to integers to resolve continous problem
        
        X = data.drop([y_col], axis = 1)
        
        # Retrieve the names of the columns to identify he RFECV features selected
        feature_names = X.columns
        
        dt_model = DecisionTreeClassifier(random_state=42)
       
        print("Making Data Classification Complete")
        y = pd.DataFrame(y)
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model use the fist one with info_converted and the second one with Source_address
        rfecv = RFECV(estimator=dt_model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        
        # Split the data for the RFECV model 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        start_train = time.time()
        # returns the selected features datasets
        features = rfecv.fit_transform(X_train, y_train)
        end_train = time.time()
        
        print(f"\nRFECV training time: {end_train - start_train:.6f} seconds", file=file)

        # Create a pipeline
        pipeline = Pipeline([
            ('feature_selection', rfecv),
            ('classification', dt_model)
        ])

        # Define parameter grid for RandomizedSearchCV
        # param_distributions = {
        #     'classification__max_depth': randint(1, 10),
        #     'classification__min_samples_split': randint(2, 20),
        #     'classification__min_samples_leaf': randint(1, 10)
        # }

        param_distributions = {
            'classification__max_depth': [9],
            'classification__min_samples_split': [8],
            'classification__min_samples_leaf': [1]
        }
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=1,
            cv=10,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the model
        random_search.fit(X_train, y_train)
        
        # Best parameters and score
        #print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_, file=file)

        # Evaluate on test data
        test_score = random_search.score(X_test, y_test)
        print("Test Accuracy:", test_score, file=file)

        rfecv_cv_scores = rfecv.cv_results_["mean_test_score"]
        
        print("RFECV CV Scores:", rfecv_cv_scores, file=file)
        
        random_cv_scores = random_search.cv_results_["mean_test_score"]
        # Perform paired t-test
        t_stat, p_value = ttest_rel(random_cv_scores, rfecv_cv_scores)

        # Output results
        print(f"Pipeline vs RFECV T-Statistic: {t_stat}, P-Value: {p_value}", file=file)

        # Compare performance of top features vs all features
        top_features = rfecv.support_
        X_top = X.iloc[:, top_features]
        
        # Cross-validation for top features and all features
        scores_top = cross_val_score(dt_model, X_top, y, cv=10, scoring='accuracy')
        scores_all = cross_val_score(dt_model, X, y, cv=10, scoring='accuracy')

        # Perform paired t-test
        t_stat, p_value = ttest_rel(scores_top, scores_all)

        # Output results
        print(f"Top Features CV Scores: {scores_top}", file=file)
        print(f"All Features CV Scores: {scores_all}", file=file)
        print(f"T-Statistic: {t_stat}, P-Value: {p_value}", file=file)

        start_test = time.time()
        # Step 6a: Make predictions on the feature set and the test set
        y_pred_rfecv_features = rfecv.estimator_.predict(features)
        y_pred_rfecv = rfecv.predict(X_test)
        end_test = time.time()
        
        print(f"RFECV testing time: {end_test - start_test:.6f} seconds", file=file)
        
        print("RFECV model fitting complete")
        
        # Step 4: Extract selected features
        selected_features = X_test.columns[rfecv.support_]
        
        # Get the names of the selected features
        selected_features_names = [name for name, selected in zip(feature_names, rfecv.support_) if selected]

        print(f"\nRFECV Selected Features: {selected_features_names}", file=file)
        
        X_selected = rfecv.transform(X_test)
        
        # Perform CHI-Square test on the RFECV selected Features
        scaled_x = (X_selected - X_selected.min())/ (X_selected.max() - X_selected.min())
        chi2_scores, p_values = chi2(scaled_x, y_test)
        
        selected_feature_indices = [i for i, x in enumerate(selected_features) if x] 
        results = pd.DataFrame({ 'Feature Index': selected_feature_indices, 'Chi2 Score': chi2_scores, 'P-Value': p_values }) 
        print(f"{results}", file=file)
        
        # Perform ANOVA and Kruskal-Wallis test for each feature
        print("\nTarget Values and RFECV Selected Features One-way ANOVA and Kruskal", file=file)
        
        anova_results = {}
        kruskal_results = {}
        for i, feature in enumerate(selected_features):
            # Group data by target classes
            groups = [X_selected[ytest=label, i] for label in np.unique(y_test)]
            # Perform ANOVA and Kruskal-Wallis
            f_stat, p_value = f_oneway(*groups)
            k_stat, kp_value = kruskal(*groups)
            anova_results[feature] = {"F-Statistic": f_stat, "FP-Value": p_value}   
            kruskal_results[feature] = {"H-Statistic": k_stat, "KP-Value": kp_value}   
            
        # Display ANOVA results
        for feature, stats in anova_results.items():
            print(f"{feature}: One-way ANOVA F-Statistic = {stats['F-Statistic']:.4f}, P-Value = {stats['FP-Value']:.4e}", file=file)

        # Display Kruskal-Wallis results
        for feature, stats in kruskal_results.items():
            print(f"{feature}: Kruskal H-Statistic = {stats['H-Statistic']:.4f}, P-Value = {stats['KP-Value']:.4e}", file=file)

        correlation_matrix = pd.DataFrame(X_selected).corr()

        # Visualize the correlation matrix
        sns.heatmap(correlation_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of RFECV Selected Features")
        plt.show()

        correlation_matrix = pd.DataFrame(X_selected).corr(method="kendall")

        # Visualize the correlation matrix
        sns.heatmap(correlation_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Kendall Correlation Matrix of RFECV Selected Features")
        plt.show()

        corr_matrix = pd.DataFrame(X_selected).corr(method="spearman")

        # Visualize the correlation matrix
        sns.heatmap(corr_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spearman Correlation Matrix of RFECV Selected Features")
        plt.show()

        corr_matrix = pd.DataFrame(X_selected).corr(method="pearson")

        # Visualize the correlation matrix
        sns.heatmap(corr_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".4f")
        plt.title("Pearson Correlation Matrix of RFECV Selected Features")
        plt.show()

        # Identify highly correlated pairs
        threshold = 0.9  # Define a threshold for correlation
        high_corr_pairs = np.where((correlation_matrix > threshold) & (correlation_matrix < 1))
        duplicates = [(correlation_matrix.index[i], correlation_matrix.columns[j]) for i, j in zip(*high_corr_pairs)]
        print("Highly correlated feature pairs:", duplicates, file=file)

        # Access cross-validation scores
        cv_scores = rfecv.cv_results_['mean_test_score']
        print("\nRFECV Cross-validation scores:", cv_scores, file=file)

        # Plot the number of features vs. cross-validation scores
        plt.figure()
        plt.xlabel("Number of Features Selected")
        plt.ylabel("RFECV V Cross-Validation Score (Accuracy)")
        plt.plot(range(1, len(cv_scores) + 1), cv_scores)
        plt.show()

        # Access cv_results_ for split-specific test scores
        cv_results = rfecv.cv_results_
        
        # Extract split-specific test scores
        for split_idx in range(cv.get_n_splits()):
            split_scores = cv_results[f'split{split_idx}_test_score']
            print(f"Split {split_idx} test scores: {split_scores}", file=file)

        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = rfecv.score(X_test, y_test)
        
        scores_cross = cross_val_score(rfecv, X, y, cv=cv, scoring='accuracy')
        
        # Plot the scores
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(scores_cross) + 1), scores_cross, marker='o', linestyle='-', color='b', label='Cross-Validation Score')
        plt.axhline(y=np.mean(scores_cross), color='r', linestyle='--', label=f'Mean Score: {np.mean(scores_cross):.4f}')
        plt.title('RFECV Cross-Validation Scores')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.xticks(range(1, len(scores_cross) + 1))
        plt.legend()
        plt.grid(False)
        plt.show()

        # Step 6: Print results
        print(f"\nRFECV Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"RFECV Best Selected features: {rfecv.support_}", file=file)
        print(f"RFECV Cross-validation scores: {scores}", file=file)
        print(f"RFECV Accuracy Cross-validation scores: {scores_cross}", file=file)
        print(f"RFECV Accuracy Cross validation Mean accuracy: {scores_cross.mean():.4f}", file=file)
        print(f"RFECV ranking: {rfecv.ranking_}", file=file)
        
        # Retrieve the number of selected features
        num_selected_features = rfecv.n_features_
        
        # Calculate the number of omitted features
        num_omitted_features = (~rfecv.support_).sum()
        
        print(f"RFECV Number of selected features: {num_selected_features}", file=file)
        print(f"RFECV Number of omitted features: {num_omitted_features}", file=file)
                
        print(f"\nRFECV Prediction: {y_pred_rfecv}", file=file)
        print(f"\nRFECV Feature Prediction: {y_pred_rfecv_features}", file=file)
        
        precision = precision_score(y_test, y_pred_rfecv, average='weighted', zero_division=1.0)
        accuracy = accuracy_score(y_test, y_pred_rfecv)
        recall = recall_score(y_test, y_pred_rfecv, average='macro', zero_division=1.0)
        f1_scores = f1_score(y_test, y_pred_rfecv, average='macro')

        
        print("RFECV Feature Classification Precision:", precision, file=file)
        print("RFECV Feature Classification Accuracy:", accuracy, file=file)
        print("RFECV Feature Classification Recall:", recall, file=file)
        print("RFECV Feature Classification F1-score:", f1_scores, file=file)
        
        
        report = classification_report(y_test, y_pred_rfecv, zero_division=1.0)
        print(f"RFECV Classification report:\n {report}", file=file)
        
        # Perform the Kruskal-Wallis test
        stat, p_value = kruskal(precision, accuracy, recall, f1_scores, X_selected)
        
        print("Kruskal-Wallis of RFECV performance metrics and features selelected", file=file)

        # Display Kruskal-Wallis results
        for i in stat:
            h_stat = i
            print(f"Kruskal H-Statistic = {h_stat}", file=file)

        for i in p_value:
            p_val = i
            print(f"Kruskal p_value = {p_val}", file=file)
            # Interpretation
            if p_val < 0.05:
                print("There is a statistically significant difference between the groups.", file=file)
            else:
                print("No statistically significant difference between the groups.", file=file)

        # Permutation importance
        # perm_importance = permutation_importance(rfecv.estimator_, X_train, y_train, n_repeats=30, random_state=42)
        # print(perm_importance.importances_mean)
        # Step 3: Extract RFECV rankings
        rfecv_rankings = rfecv.ranking_

        # Step 4: Define another set of rankings (e.g., feature importances)
        feature_importances = dt_model.fit(X, y).feature_importances_
        importance_rankings = feature_importances.argsort().argsort() + 1  # Convert to rank format
        # Step 5: Calculate Spearman correlation
        correlation, p_value = spearmanr(rfecv_rankings, importance_rankings)

        # Step 6: Output the results
        print(f"Feature Ranking Spearman Correlation: {correlation} P-value: {p_value}", file=file)
        
        precision = precision_score(y_train, y_pred_rfecv_features, average='macro', zero_division=1.0)
        accuracy = accuracy_score(y_train, y_pred_rfecv_features)
        recall = recall_score(y_train, y_pred_rfecv_features, average='macro')
        # Calculate F1 score for each class
        f1_scores = f1_score(y_train, y_pred_rfecv_features, average='macro')
       
        print("RFECV Feature Prediction Classification Precision:", precision, file=file)
        print("RFECV Feature Prediction Classification Accuracy:", accuracy, file=file)
        print("RFECV Feature Prediction Classification Recall:", recall, file=file)
        print("RFECV Feature Prediction Classification F1-Score:", f1_scores, file=file)
        
        
        # Binarize the output
        y_true_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred_rfecv, classes=[0, 1, 2])
        precision, recall, thresholds  = [precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(y_true_bin.shape[1])]
        print(precision, file=file)
        print(recall, file=file)
        print(thresholds, file=file)
        
        # Plot the RFECV feature data for visualization
        print(f"Optimal number of features: {rfecv.n_features_}", file=file)
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
        plt.grid(False)
        plt.show()
        
        # Plot feature rankings
        plt.figure(figsize=(10, 6))
        plt.bar(range(X_test.shape[1]), rfecv.ranking_, color='skyblue')
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Ranking")
        plt.title("Feature Rankings by RFECV")
        plt.xticks(range(X.shape[1]))
        plt.show()
        
        selected_features = rfecv.support_

        # Heatmap of selected features
        sns.heatmap([selected_features], cmap="coolwarm", cbar=False, xticklabels=range(X_test.shape[1]))
        plt.xlabel("Feature Index")
        plt.title("Selected Features (1 = Selected, 0 = Not Selected)")
        plt.show()
        
        # Count duplicate features
        original_features = pd.DataFrame(X_test).columns
        selected_feature_names = original_features[selected_features]
        duplicates = len(selected_feature_names) - len(set(selected_feature_names))

        print(f"RFECV Number of duplicate features: {duplicates}", file=file)

        id_features = pd.DataFrame(rfecv.ranking_)
        id_features = id_features.rename(columns={0: "Feature Ranking"})
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(id_features, diag_kind="kde")
        plt.show()
        
        # Generate and display the RFECV confusion matrix
        cm_features = confusion_matrix(y_train, y_pred_rfecv_features, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_features, display_labels=rfecv.classes_)
        disp.plot(cmap='Purples')
        print(f"\nRFECV Feature Predicted Confusion matrix:\n {cm_features}", file=file)
 
        # Step 3: Calculate False Positive Rate (FPR) for each class
        fpr_per_class = []
        for i in range(len(cm_features)):
            fp = sum(cm_features[:, i]) - cm_features[i, i]  # False Positives
            tn = cm_features.sum() - (sum(cm_features[i, :]) + sum(cm_features[:, i]) - cm_features[i, i])  # True Negatives
            fpr = fp / (fp + tn)
            fpr_per_class.append(fpr)

        # Step 4: Perform z-test
        # Hypothesized mean FPR (e.g., 0.1)
        hypothesized_mean = 0.1
        sample_mean = np.mean(fpr_per_class)
        sample_std = np.std(fpr_per_class, ddof=1) / np.sqrt(len(fpr_per_class))  # Standard error

        # Z-score calculation
        z_score = (sample_mean - hypothesized_mean) / sample_std

        # P-value calculation
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Output results
        print(f"FPR per class: {fpr_per_class}", file=file)
        print(f"Sample mean FPR: {sample_mean}", file=file)
        print(f"Z-score: {z_score} P-value: {p_value}",file=file)

        # Interpretation
        if p_value < 0.05:
            print("Reject the null hypothesis: The FPR is significantly different from the hypothesized mean.", file=file)
        else:
            print("Fail to reject the null hypothesis: No significant difference in FPR.", file=file)

        # Calculate unrelated class creations (misclassifications)
        misclassification = cm_features.sum() - cm_features.diagonal().sum()
        print(f"RFECV Feature predicted Number of misclassifications: {misclassification} out of {cm_features.diagonal().sum()}", file=file)

        # Compute False Positives for each class
        false_positive = np.sum(cm_features, axis=0) - np.diag(cm_features)

        print(f"\nRFECV Feature Classification False Positives for each class:\n {false_positive}", file=file)
        
        # Calculate False Negatives for each class
        false_negative = np.sum(cm_features, axis=1) - np.diag(cm_features)

        print(f"\nRFECV False Negatives for each class:\n {false_negative}", file=file)
        
        true_postives = np.diag(cm_features)
        
        actual_totals = np.sum(cm_features, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nRFECV feature predicted True Positives for each class:\n {true_postives}", file=file)
        print(f"\nRFECV feature predicted True Positives rate for each class:\n {tpr}", file=file)
        
        num_classes = cm_features.shape[0]
        
        n_features = rfecv.n_features_
        cm_metrics.calcuateTrueFalse("RFECV Feature Predicted", num_classes, cm_features, n_features, file)

        # Generate and display the RFECV confusion matrix
        cm = confusion_matrix(y_test, y_pred_rfecv)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfecv.classes_)
        disp.plot(cmap=plt.cm.Oranges)
        print(f"\nRFECV Confusion matrix:\n {cm}", file=file)
        
        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"RFECV Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)

        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"\nRFECV Feature Classification False Positives for each class:\n {false_positives}", file=file)
        
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nRFECV False Negatives for each class:\n {false_negatives}", file=file)
        
        true_postives = np.diag(cm)
        
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nRFECV True Positives for each class:\n {true_postives}", file=file)
        print(f"\nRFECV True Positives rate for each class:\n {tpr}", file=file)
        
        num_classes = cm.shape[0]
        
        cm_metrics.calcuateTrueFalse("RFECV", num_classes, cm, n_features, file)
        
        print("\nRFECV Cross-validation scores for each number of features: ", file=file)
        print(f"Mean Score:\n {rfecv.cv_results_['mean_test_score']}", file=file,)
        print(f"Standard Deviation:\n {rfecv.cv_results_['std_test_score']}", file=file )
        
        # Plot the results
        plt.figure()    
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validation Score (Accuracy)")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.show()
        
        # Step 4: Extract selected features
        selected_features_OLS = X_test.columns[rfecv.support_]
        X_selected = X_test[selected_features_OLS]

        print(f"\nRFECV OFS Selected Features: {list(selected_features_OLS)}", file=file)

        # Step 5: Fit an OLS model using statsmodels
        X_selected_with_const = sm.add_constant(X_selected)  # Add intercept
        ols_model = sm.OLS(y_test, X_selected_with_const).fit()

        # Step 6: Print OLS summary
        print(f"\nRFECV {ols_model.summary()}", file=file)
        
        # Plot the RFECV results
        # Access split(k)_test_scores
        split_test_scores = np.array(rfecv.cv_results_['split0_test_score'])  
        
        # Plot split(k)_test_scores
        plt.figure(figsize=(10, 6))
        for i in range(cv.get_n_splits()):
            plt.plot(range(1, len(split_test_scores) + 1), 
                     rfecv.cv_results_[f'split{i}_test_score'], 
                     label=f'Split {i + 1}')

        plt.xlabel("Number of Features Selected")
        plt.ylabel("Test Score")
        plt.title("RFECV Split Test Scores")
        plt.legend()
        plt.grid(False)
        plt.show()
        
        print("\nEvaluating irrelavent features in RFECV", file=file)
        # Add random noise features
        np.random.seed(42)
        
        orignal_selected = rfecv.support_
        original_ranking = rfecv.ranking_
        
        dt_model.fit(X, y)
        dt_model_pred = dt_model.predict(X)
        accuracy_before = accuracy_score(y, dt_model_pred)
        
        # Model performance after RFECV
        X_selected = rfecv.transform(X)
        dt_model.fit(X_selected, y)
        y_pred_selected = dt_model.predict(X_selected)
        accuracy_after = accuracy_score(y, y_pred_selected)
        
        print("RFECV irrelevant feature accuracy", accuracy_before, "and after", accuracy_after, file=file)
        
        # Plot the cross-validation scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
                 rfecv.cv_results_['mean_test_score'], marker='o')
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Mean Cross-Validation Score")
        plt.title("RFECV Cross-Validation Scores vs Number of Features")
        plt.grid()
        plt.show()
        
        X_with_noise = pd.DataFrame(X) # create the irrelevant and duplciate features
        X_with_noise['Irrlevant'] = np.random.random(size=len(X)) # add in irrelevant data into the exiting dataset
        X_with_noise['D_SourceIp'] = X['Source_address'] # duplicate on existing one
        #X_with_noise = np.hstack((X, irrelevant_features))

        # Fit RFECV with irrelevant features
        rfecv_with_noise = RFECV(estimator=dt_model, step=1, cv=cv, scoring='accuracy')
        features_w_noise = rfecv_with_noise.fit_transform(X_with_noise, y)

        after_selected = rfecv_with_noise.support_
        after_ranking = rfecv_with_noise.ranking_

        print("Features selected", orignal_selected, " after: ", after_selected, file=file)
        print("Feature ranking:", original_ranking, " after:", after_ranking, file=file)
        
        selected_features = X_with_noise.columns[rfecv_with_noise.support_]
        print("Selected Features:", selected_features, file=file)

        # Step 5: Check if duplicate feature was removed
        if 'D_SourceIp' not in selected_features:
            print("Duplicate feature was removed.", file=file)
        else:
            print("Duplicate feature was retained.", file=file)
            
        # Step 5: Check if duplicate feature was removed
        if 'Irrlevant' not in selected_features:
            print("Irrlevant feature was removed.", file=file)
        else:
            print("Irrlevant feature was retained.", file=file)
        
        scores_noise = rfecv_with_noise.score(X_with_noise, y)
 
        scores_with_irrelevant = cross_val_score(dt_model, X_with_noise, y, cv=cv, scoring='accuracy')

        # Remove irrelevant features and evaluate again
        X_selected = rfecv.transform(X_train)
        
        scores_without_irrelevant = cross_val_score(dt_model, X_selected, y_train, cv=cv, scoring='accuracy')        
        print(f"\nRFECV with Noise: Optimal number of features (with noise): {rfecv_with_noise.n_features_}", file=file)
        print(f"RFECV With Noise Best Selected features: {rfecv_with_noise.support_}", file=file)
        print(f"RFECV With Noise Cross-validation scores: {scores_noise}", file=file)
        print(f"RFECV With Noise Accuracy Cross-validation scores: {scores_with_irrelevant}", file=file)
        print(f"RFECV With Noise Accuracy Cross validation Mean accuracy: {scores_with_irrelevant.mean():.4f}", file=file)
        print(f"RFECV With Noise ranking: {rfecv_with_noise.ranking_}", file=file)

        # Perform paired t-test
        sum1 = np.sum(scores_without_irrelevant, axis=0)
        sum2 = np.sum(scores_with_irrelevant, axis=0)
        if ( sum1 == sum2):
            print("All features are relevant", file=file)
        else:
            t_stat, p_value = ttest_rel(scores_without_irrelevant, scores_with_irrelevant)
            print(f"RFECV Irrelevant T-statistic: {t_stat}, P-value: {p_value}", file=file)
            if (p_value < 0.05):
                print("Irrelevant features impacted RFECV performane", file=file)
            else:
                print("Irrelevant features had no impact on RFECV performance", file=file)

        stat, p_value = wilcoxon(scores_without_irrelevant, scores_with_irrelevant)
        print(f"Irrelevant Features Wilcox Signed Rank Statistical Results: {stat}, p-value: {p_value:.6f}", file=file)
        if p_value < 0.05:
              print("Significant differences exist between the groups.", file=file)
        else:
              print("No significant differences between the groups.", file=file)
                
        selected_features = rfecv_with_noise.support_

        original_features = pd.DataFrame(X_with_noise).columns
        selected_feature_names = original_features[selected_features]


        # Transform the dataset to include only selected features as input into DT classifier
        features_selected = pd.DataFrame(features_w_noise, columns=selected_feature_names)
        
        # Identify if duplicate features impaced RFECV performance
        correlation_matrix = features_selected.corr()
        
        # Identify pairs of features with high correlation
        redundant_features = [
            (col1, col2) for col1 in correlation_matrix.columns
            for col2 in correlation_matrix.columns
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
        ]
        
        print("Redundant feature pairs:", redundant_features, file=file)
        
        # Calculate Variance Inflation Factor for each feature
        # Features with VIF > 10 are typically considered redundant.
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features_selected.columns
        i = len(vif_data)
        if (i > 1):
            vif_data["VIF"] = [variance_inflation_factor(features_selected.values, i) for i in range(features_selected.shape[1])]
        else:
            vif_data = 0

        print(f"RFECV Variance Inflation Factor:\n {vif_data}", file=file)
        
        # Get feature rankings
        feature_ranking = pd.DataFrame({
            "Feature": X.columns,
            "Ranking": rfecv.ranking_
        }).sort_values(by="Ranking")

        print(f"\nRFECV feature ranking to identify duplicate:\n {feature_ranking}\n", file=file)
         
        # Calculate mutual information
        mi = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi})
        print(mi_df.sort_values(by="Mutual Information", ascending=False), file=file)
        
       
        # Transform the dataset to include only selected features as input into DT classifier
        features_selected = pd.DataFrame(features_selected, columns=selected_feature_names)
        
        # Identify if duplicate features impacted RFECV performance
        correlation_matrix = features_selected.corr()
        
        # Identify pairs of features with high correlation
        redundant_features = [
            (col1, col2) for col1 in correlation_matrix.columns
            for col2 in correlation_matrix.columns
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
        ]
        
        print("Redundant feature pairs:", redundant_features, file=file)
        
        # Calculate Variance Inflation Factor for each feature
        # Features with VIF > 10 are typically considered redundant.
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features_selected.columns
        i = len(vif_data)
        if (i > 1):
            vif_data["VIF"] = [variance_inflation_factor(features_selected.values, i) for i in range(features_selected.shape[1])]
        else:
            vif_data = 0

        print(f"RFECV Variance Inflation Factor:\n {vif_data}", file=file)
        
        corr_matrix = pd.DataFrame(X_with_noise).corr(method="spearman")

        # Visualize the correlation matrix
        sns.heatmap(corr_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spearman Correlation Matrix of RFECV Irreelvant Selected Features")
        plt.show()

        # Restore the value after irrelevant and duplicate test
        selected_features = rfecv.support_

        X_selected = rfecv.transform(X_test)
        # Count duplicate features
        original_features = pd.DataFrame(X_test).columns
        selected_feature_names = original_features[selected_features]
        
        # Transform the dataset to include only selected features as input into DT classifier
        X_selected = pd.DataFrame(X_selected, columns=selected_feature_names)
        
        
        # Transform the dataset to include only selected features as input into DT classifier
        X_train_selected = rfecv.transform(X_train)
        X_test_selected = rfecv.transform(X_test)
        
        
        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=kf, scoring='accuracy')
 
        # Step 6: Print results
        print(f"\nKF Optimal number of features: {rfecv.n_features_}", file=file)
        print(f"KF Selected features: {rfecv.support_}", file=file)
        print(f"KF Cross-validation scores: {scores}", file=file)
        print(f"KF Mean accuracy: {scores.mean():.4f}", file=file)
        print(f"KF RFECV ranking: {rfecv.ranking_}", file=file)
        
        print("Evaluating the RFECV Features input into the Decision Tree Model", file=file)
       
        start_train = time.time()
        dt_model.fit(X_train_selected, y_train)
        end_train = time.time()
        
        print(f"DT Model training time: {end_train - start_train:.6f} seconds", file=file)
        
        # Plot the decision tree
        #plt.figure(figsize=(10, 6))
        #plot_tree(dt_model, filled=True)
        #plt.show()  
        
        start_test = time.time()
        dt_predictions = dt_model.predict(X_test_selected)
        end_test = time.time()
       
        print(f"DT Model testing time: {end_test - start_test:.6f} seconds", file=file)
       
        start_test = time.time()
        y_pred_dt = dt_model.predict(X_test_selected)
        end_test = time.time()
        
        print(f"DT Model second testing time: {end_test - start_test:.6f} seconds", file=file)
       
        # Find the number of classes
        num_classes = len(dt_model.classes_)
        print(f"The decision tree predicted {num_classes} classes: {dt_model.classes_}", file=file)
        
        n_nodes = dt_model.tree_.node_count
        print(f"The decision tree number of nodes {n_nodes}", file=file)
        
        # Generate and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred_dt, labels=dt_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
        disp.plot(cmap=plt.cm.Greens)
        plt.show()
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, dt_predictions)

        # Display confusion matrix
        print(f"\nDT Confusion Matrix:\n {cm}", file=file)
        
        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"DT Classification False Positives for each class:\n {false_positives}", file=file)
        
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nDT False Negatives for each class:\n {false_negatives}", file=file)
        
        true_postives = np.diag(cm)
        
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nDT True Positives for each class:\n {true_postives}", file=file)
        print(f"\nDT True Positives rate for each class:\n {tpr}", file=file)
        
        # Retrieve the number of class categories
        dt_num_classes = dt_model.n_classes_
        print(f"DT Number of class categories: {dt_num_classes}", file=file)

        num_classes = cm.shape[0]
        cm_metrics.calcuateTrueFalse("DT", num_classes, cm, dt_num_classes, file)
        
        accuracy = accuracy_score(y_test, y_pred_dt)
        precision = precision_score(y_test, y_pred_dt, average='macro', zero_division=1.0)
        recall = recall_score(y_test, y_pred_dt, average='macro')
       
        # Calculate F1 score for each class
        f1_scores = f1_score(y_test, y_pred_dt, average=None)

        # Display F1 scores for each class
        for i, score in enumerate(f1_scores):
            print(f"Decision Tree F1 Score for class {i}: {f1_scores}", file=file)

        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        dt_cv_scores = cross_val_score(dt_model, X, y, cv=cv, scoring='accuracy')

        print("\nDT Classifier Accuracy:", accuracy, file=file)
        print("DT Classifier Precision:", precision, file=file)
        print("DT Classifier Recall:", recall, file=file)
        print("Decision Tree Classifier:", np.mean(dt_cv_scores), file=file)
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}", file=file)
        
        dt_performance = classification_report(y_test, dt_predictions, zero_division=1.0)
        print(f"\nDT Classification performance: \n {dt_performance}", file=file)

        # Retrieve the number of unrelated class categories
        dt_omt_num_classes = len(dt_model.classes_)
        print(f"DT Number of unrelated class categories: {dt_omt_num_classes}", file=file)
       
        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"DT Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)

        viz = FeatureImportances(dt_model)
        viz.fit(X_train_selected, y_train)
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
        plt.xticks(range(X_test_selected.shape[1]), np.array(X.columns)[indices], rotation=90, size=12)
        plt.show()

        # Predict probabilities for the positive class
        y_scores = dt_model.predict_proba(X_test_selected)[:, 1]
        
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
        
        y_score = dt_model.predict_proba(X_test_selected)
        #roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        #print(f"DT ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Step 2: Train a binary decision tree for each class
        classifiers = {}
        for class_label in np.unique(y):
            # Create binary labels for the current class (One-vs-Rest)
            y_train_binary = (y_train == class_label).astype(int)
    
            # Train a decision tree classifier
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train_selected, y_train_binary)
            classifiers[class_label] = clf
        
        # Step 3: Make predictions
        predictions = []
        for class_label, clf in classifiers.items():
            # Predict probabilities for the current class
            pred_prob = clf.predict_proba(X_test_selected)[:, 1]  # Probability of being in the current class
            predictions.append(pred_prob)

        # Combine predictions and determine the final class
        predictions = np.array(predictions).T
        final_predictions = np.argmax(predictions, axis=1)

        # Step 4: Evaluate the model
        accuracy = accuracy_score(y_test, final_predictions)
        print(f"Decision Tree Binary Classification Accuracy: {accuracy:.2f}", file=file)

        # Get decision tree feature importances
        importances = dt_model.feature_importances_

        X_selected_with_const = sm.add_constant(dt_predictions)  # Add intercept
        ols_model = sm.OLS(y_test, X_selected_with_const).fit()

        # Step 6: Print OLS summary
        print(f"\nDT {ols_model.summary()}", file=file)

        # Plot observed vs regression results
        plt.figure(figsize=(8, 6))
        plt.scatter(X_selected_with_const[:, 1], y_test, label="DT Observed", color="blue", alpha=0.6)
        plt.plot(X_selected_with_const[:, 1], dt_predictions, label="Regression Line", color="red", linewidth=2)
        plt.xlabel("Independent Variable (X)")
        plt.ylabel("Dependent Variable (y)")
        plt.title("Observed vs Regression Results")
        plt.legend()
        plt.grid(False)
        plt.show()

        # Gaussian Naive Bayes Classifier
        start_time = time.time()
        gnb_model = GaussianNB()
        gnb_model.fit(X_train_selected, y_train)
        end_time = time.time()
        
        print(f"GNB Model training time: {end_time - start_time:.6f} seconds", file=file)
        
        start_test = time.time()
        gnb_predictions = gnb_model.predict(X_test_selected)
        end_test = time.time()
        
        print(f"GNB Model testing time: {end_test - start_test:.6f} seconds", file=file)
       
        # Plot confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, gnb_predictions)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, gnb_predictions)

        # Output the confusion matrix into the file
        print(f"\nGNB Confusion Matrix:\n {cm}", file=file)
        
        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"GNB Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)

        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"\nGNB False Positives for each class:\n {false_positives}", file=file)
 
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nGNB False Negatives for each class:\n {false_negatives}", file=file)
    
        true_postives = np.diag(cm)
    
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
    
        print(f"\nGNB True Positives for each class:\n {true_postives}", file=file)
        print(f"\nGNB True Positives rate for each class:\n {tpr}", file=file)
    
        num_classes = cm.shape[0]
        gnb_classes = len(gnb_model.classes_)
        cm_metrics.calcuateTrueFalse("GNB", num_classes, cm, gnb_classes, file)
        
        # Predict probabilities
        y_proba = gnb_model.predict_proba(X_test_selected)

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
        
        fOne = cross_val_score(gnb_model, X, y, cv=10)

        # We print the F1 score here
        print("\nAverage GNB F1 score during cross-validation: ", np.mean(fOne), file=file)
        print("GNB f1 scores: ", fOne.mean(), file=file)

        # Then print the F1 score to the output file
        print(f"\nAverage GNB F1 score during cross-validation: {np.mean(fOne)}", file=file)

        # Generate and print the classification report
        report = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"GNB Classification report:\n {report}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
        y_scores = gnb_model.predict_proba(X_test_selected)[:, 1]
            
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
    
        y_score = gnb_model.predict_proba(X_test_selected)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"GNB ROC AUC Score: {roc_auc:.2f}", file=file)
    
        num_columns = X_train_selected.shape[1]
        print(f"\nNumber of columns in X_train: {num_columns}", file=file)
        num_columns = X_train_selected.shape[0]
        print(f"Number of rows in X_train: {num_columns}", file=file)
        
        num_columns = X_test_selected.shape[1]
        print(f"Number of columns in X_test: {num_columns}", file=file)
        num_columns = X_train_selected.shape[0]
        print(f"Number of rows in X_train: {num_columns}", file=file)
        
        # Evaluate Gaussian Naive Bayes
        print("\nGaussian Naive Bayes Classifier:", file=file)
        print(f"\nGNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}", file=file)
        gnb_performance = classification_report(y_test, gnb_predictions, zero_division=1.0)
        print(f"\nGNB Classification performance: \n {gnb_performance}", file=file)
        
        X_ols = sm.add_constant(gnb_predictions)  # Add constant for intercept
        # Fit OLS regression
        ols_model = sm.OLS(y_test, X_ols).fit()
        
        # Print OLS summary
        print(f"\nGNB {ols_model.summary()}", file=file)
        
        # Plot observed vs regression results
        plt.figure(figsize=(8, 6))
        plt.scatter(X_ols[:, 1], y_test, label="GNB Observed", color="blue", alpha=0.6)
        plt.plot(X_ols[:, 1], gnb_predictions, label="Regression Line", color="red", linewidth=2)
        plt.xlabel("Independent Variable (X)")
        plt.ylabel("Dependent Variable (y)")
        plt.title("Observed vs Regression Results")
        plt.legend()
        plt.grid(False)
        plt.show()

        # Define base models and meta-classifier
        base_models = [
            ('dt', DecisionTreeClassifier()),
            ('gnb', GaussianNB())
            ]
        
        # Combine DT with GNB using StackingClassifier and default final estimator
        meta_classifier = LogisticRegression(max_iter=100, solver="saga", tol=1e-2)
        
        # Combine using VotingClassifier
        start_time = time.time()
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        votingscore = voting_clf.fit(X_train_selected, y_train)
        end_time = time.time()
        
        print(f"\nVoting Classifier training time: {end_time - start_time:.6f} seconds", file=file)
        
        # get the truth labels from the Voting classifier
        start_test = time.time()
        voting_predict = votingscore.predict(X_test_selected)
        end_test = time.time()
        
        print(f"Voting Classifier testing time: {end_test - start_test:.6f} seconds", file=file)
        
        print((f"\nVoting Classiferr DT with GNB predictions: {voting_predict} "), file=file)
        
        y_pred_voting = votingscore.predict(X_test_selected)
        print("\nVoting Classifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_voting), file=file)
        print("\nVoting Classifier DT with GNB confusion matrix:\n", confusion_matrix(y_test, y_pred_voting), file=file)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_voting)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=voting_clf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"Voting Classification Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)
        
        # Compute precision
        precision = precision_score(y_test, y_pred_voting, average='weighted')

        print(f"Voting Classification Precision Score (weighted): {precision:.4f}", file=file)

        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"Voting Classification False Positives for each class:\n {false_positives}", file=file)
        
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nVoting Classification False Negatives for each class:\n {false_negatives}", file=file)
        
        true_postives = np.diag(cm)
        
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nVoting Classification True Positives for each class:\n {true_postives}", file=file)
        print(f"\nVoting Classification True Positives rate for each class:\n {tpr}", file=file)
        
        num_classes = cm.shape[0]
        v_classes = len(voting_clf.classes_)
        cm_metrics.calcuateTrueFalse("Voting Classifier", num_classes, cm, v_classes, file)
        
        voting_clf_classification = classification_report(y_test, y_pred_voting, zero_division=1.0)
        print(f"\nVoting Classification Report: \n {voting_clf_classification}", file=file)
        
        # Step 5: Calculate ROC AUC score
        y_score = votingscore.predict_proba(X_test_selected)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"Voting Classifer DT with GNB ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
        y_scores = gnb_model.predict_proba(X_test_selected)[:, 1]
            
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
        
        # Stacking Classifier
        start_time = time.time()
        stacking_clf = StackingClassifier(
            estimators=[
                ('dt', dt_model),
                ('gnb', gnb_model)
        ],
        final_estimator=LogisticRegression()
        )

        # Pipeline with RFECV
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Optional: Scale features
            ('feature_selection', RFECV(estimator=dt_model, step=1, cv=10, scoring='accuracy')),
            ('stacking', stacking_clf)
         ])
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate the pipeline
        accuracy = pipeline.score(X_test, y_test)
        print(f"RFECV DT and GNB Pipeline Accuracy: {accuracy:.2f}", file=file)

        # Evaluate performance
        start_test = time.time()
        y_pred_pipeline = pipeline.predict(X_test)
        end_test = time.time()
        
        precision = precision_score(y_test, y_pred_pipeline, average='micro', zero_division=1.0)
        accuracy = accuracy_score(y_test, y_pred_pipeline)
        recall = recall_score(y_test, y_pred_pipeline, average='micro')
       
        print("Pipeline Precision:", precision, file=file)
        print("Pipeline Accuracy:", accuracy, file=file)
        print("Pipeline Recall:", recall, file=file)
        
        # Detailed classification report
        pipeline_performance = classification_report(y_test, y_pred_pipeline, zero_division=1.0)
        print(f"\nPipeline Classification Report: \n{pipeline_performance}", file=file)
       
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_pipeline)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        disp.plot(cmap=plt.cm.Blues)

        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"Pipeline Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)

        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"Pipeline False Positives for each class:\n {false_positives}", file=file)
        
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nPipeline False Negatives for each class:\n {false_negatives}", file=file)
        
        true_postives = np.diag(cm)
        
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nPipeline True Positives for each class:\n {true_postives}", file=file)
        print(f"\nPipeline True Positives rate for each class:\n {tpr}", file=file)
        
        pip_scores_cross = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        # We print the F1 score here
        print("\nAverage Pipeline cross-validation score: ", np.mean(pip_scores_cross), file=file)
        
        # Display F1 scores for each class
        for i, score in enumerate(pip_scores_cross):
            print(f"Pipeline cross validation score for class {i}: {pip_scores_cross}", file=file)

        # Step 5: Calculate ROC AUC score
        y_score = pipeline.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"Pipeline RFECV into DT into GNB ROC AUC Score: {roc_auc:.2f}", file=file)

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_pipeline, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Pipeline Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        optimal_params = {
            'dt__criterion': ['gini'],       
            'dt__max_depth': [None],
            'dt__min_samples_split': [2],
            'dt__min_samples_leaf': [1],
            'gnb__var_smoothing': [1e-9],
            'final_estimator__C': [1]
            }
                                   
        stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, passthrough=True )
        # Perform grid search
        stacking_clf = GridSearchCV(estimator=stacking_clf, param_grid=optimal_params, cv=5, scoring='accuracy', return_train_score=True)
        # train on the RFECV selected critical features
        stacking_clf.fit(X_train_selected, y_train)
        # train on the whole dataset 
        #stacking_clf.fit(X_train, y_train)
        end_time = time.time()
        print(f"\nDT with GNB Stacking Classifier training time: {end_time - start_time:.6f} seconds", file=file)
        
        meta_feature_train = stacking_clf.transform(X_test_selected)
        
        stacking_features = stacking_clf.cv_results_['rank_test_score']
        
        print("Stacking features score: ", stacking_features, file=file)
       
        stacking_cv_scores = stacking_clf.cv_results_["mean_test_score"]
        # Perform paired t-test
        t_stat, p_value = ttest_rel(stacking_cv_scores, rfecv_cv_scores)

        # Output results
        print(f"Stacking vs RFECV T-Statistic: {t_stat}, P-Value: {p_value}", file=file)

        # Best parameters and score
        #print(f"\nStacking Classifier Best Parameters: {stacking_clf.best_params_}", file=file)
        print(f"Stacking Classifier Best Accuracy: {stacking_clf.best_score_}", file=file)
        print(f"Stacking Classifier CV Scores: {stacking_cv_scores}", file=file)

        # Perform cross-validation
        cv_scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
        
        print(f"\nStacking Classifier Mean CV Accuracy: {cv_scores.mean():.4f}", file=file)
        print(f"\nStacking Classifier STD Cross-Validation Accuracy: {cv_scores.std():.4f}", file=file)

        # Evaluate performance
        start_test = time.time()
        y_pred_stacking = stacking_clf.predict(X_test_selected)
        end_test = time.time()
        
        print(f"Stacking Classifer testing time: {end_test - start_test:.6f} seconds", file=file)
        
        X_selected_with_const = sm.add_constant(y_pred_stacking)  # Add intercept
        ols_model = sm.OLS(y_test, X_selected_with_const).fit()

        # Step 6: Print OLS summary
        print(f"\nRFECV in DL DT with GNB {ols_model.summary()}", file=file)

        # Plot observed vs regression results
        plt.figure(figsize=(8, 6))
        plt.scatter(X_selected_with_const[:, 1], y_test, label="RFECV in DT with GNB Observed", color="blue", alpha=0.6)
        plt.plot(X_selected_with_const[:, 1], y_pred_stacking, label="Regression Line", color="red", linewidth=2)
        plt.xlabel("Independent Variable (X)")
        plt.ylabel("Dependent Variable (y)")
        plt.title("Observed vs Regression Results")
        plt.legend()
        plt.grid(False)
        plt.show()

        print("\nStacking Classifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred_stacking), file=file)
        print("Stacking Classifier DT with GNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred_stacking), file=file)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_stacking)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stacking_clf.classes_)
        disp.plot(cmap=plt.cm.Blues)

        # Calculate unrelated class creations (misclassifications)
        misclassifications = cm.sum() - cm.diagonal().sum()
        print(f"Stacking Classification Number of misclassifications: {misclassifications} out of {cm.diagonal().sum()}", file=file)

        # Compute precision
        precision = precision_score(y_test, y_pred_stacking, average='macro', zero_division=1)

        print(f"Stacking Classification Precision Score (macro): {precision:.4f}", file=file)

        fOne = f1_score(y_test, y_pred_stacking, average="macro")
        print(f"Stacking Classification F1 Score (macro): {fOne:.4f}", file=file)

        recall = recall_score(y_test, y_pred_stacking, average='macro')
        print(f"Stacking Classification Recall Score (macro): {recall:.4f}", file=file)

        accuracy = accuracy_score(y_test, y_pred_stacking)
        print(f"Stacking Classification Accuracy Score: {accuracy:.4f}", file=file)

        # Perform the Kruskal-Wallis test
        stat, p_value = kruskal(precision, accuracy, recall, fOne, meta_feature_train)

        # Display Kruskal-Wallis results
        for i in stat:
            h_stat = i
            print(f"Kruskal H-Statistic = {h_stat}", file=file)

        for i in p_value:
            p_val = i
            print(f"Kruskal p_value = {p_val}", file=file)
            # Interpretation
            if p_val < 0.05:
                print("There is a statistically significant difference between the groups.", file=file)
            else:
                print("No statistically significant difference between the groups.", file=file)

        # Compute False Positives for each class
        false_positives = np.sum(cm, axis=0) - np.diag(cm)

        print(f"Stacking Classifier False Positives for each class:\n {false_positives}", file=file)
        
        # Calculate False Negatives for each class
        false_negatives = np.sum(cm, axis=1) - np.diag(cm)

        print(f"\nStacking Classification False Negatives for each class:\n {false_negatives}", file=file)
        
        true_postives = np.diag(cm)
        
        actual_totals = np.sum(cm, axis=1)  # Sum of rows
        tpr = true_postives / actual_totals
        
        print(f"\nStacking Classification True Positives for each class:\n {true_postives}", file=file)
        print(f"\nStacking Classification True Positives rate for each class:\n {tpr}", file=file)
        
        # Step 6: Retrieve the best model and evaluate
        # best_model = stacking_clf.cv_results_
        
        # print(f"Best Parameters: {best_model}", file=file)
        print(f"Best Score: {stacking_clf.best_score_}", file=file)

        num_classes = cm.shape[0]
        stc_classed = len(stacking_clf.classes_)
        cm_metrics.calcuateTrueFalse("Stacking Classification", num_classes, cm, stc_classed, file)
        
        # Retrieve the number of unrelated class categories
        clf_num_omt_classes = len(stacking_clf.classes_)
        print(f"\nStacking Classifier DT with GNB  Number of unrelated class categories: {clf_num_omt_classes}", file=file)
        
        # Retrieve the number of class categories
        clf_num_classes = stacking_clf.classes_
        print(f"\nStacking Classifier DT with GNB Number of class categories: {clf_num_classes}", file=file)
        
        # Detailed classification report
        stacking_clf_performance = classification_report(y_test, y_pred_stacking, zero_division=1.0)
        print(f"\nStacking Classification Report: \n{stacking_clf_performance}", file=file)
        
        # Step 5: Calculate ROC AUC score
        y_score = stacking_clf.predict_proba(X_test_selected)
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')
        print(f"Stacking Classifer DT with GNB ROC AUC Score: {roc_auc:.2f}", file=file)
        
        # Predict probabilities for the positive class for the ROC and AOC analysis
        #y_scores = gnb_model.predict_proba(X_test_selected)[:, 1]
            
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
        
        return X_selected, selected_features_names, y_pred_rfecv, dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking, X_train_selected, X_test
    
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
                # Perform one-way ANOVA on the features that were selected
                if (column == selected_features_names[i]):
                    features = feature[column]
                    data = data_scaled[column]
                    name = selected_features_names[i]
                    #f_statistic, p_value = f_oneway(source, data,features)
                    f_statistic, p_value = f_oneway(data,features)
                    print(f"\nF-statistic for feature {name}, {f_statistic}, P-value: {p_value:.6f}", file=file)
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
                    print(f"Kruskal Statistical Results for length and {name}, {stat}, p-value: {p:.6f}", file=file)
                    i = i+1
                    f_statistic = 0
                    p_value = 0
                    stat = 0
                    p = 0
                elif (i == index):
                    break
        else:      
            print("No multiple Features identified therefore going to use one feature identified", file=file)
    
    def kruskalWallis(self, group1, group2, group3, group4, group5, file):
        # Perform the Kruskal-Wallis test
        stat, p_value = kruskal(group1, group2, group3, group4, group5)

        # Display the results
        print(f"\nFirst four data and flags with burst rate Kruskal-Wallis H-statistic: {stat} P-value: {p_value:.6f}", file=file)
        # Interpretation
        if p_value < 0.05:
            print("There is a statistically significant difference between the groups.", file=file)
        else:
            print("No statistically significant difference between the groups.", file=file)
            
    def kruskalWallisM(self, group1, group2, file):
        source = group1.Source_address
        destination = group1.Destination_address
        protocol = group1.Destination_address
        length = group1.Length
        # Perform the Kruskal-Wallis test
        stat, p_value = kruskal(source, destination, protocol, length, group2)

        # Display the results
        print(f"RFECV Feature and 3 Data Kruskal-Wallis H-statistic: {stat} P-value: {p_value:.6f}", file=file)
        # Interpretation
        if p_value < 0.05:
            print("There is a statistically significant difference between the groups.", file=file)
        else:
            print("No statistically significant difference between the groups.", file=file)
        
            
    def oneAnova(self, data_scaled, features, file):  
        source = data_scaled.Source_address
        destination = data_scaled.Destination_address
        protocol = data_scaled.Destination_address
        length = data_scaled.Length
        f_statistic, p_value = f_oneway(source, destination, protocol, length, features)
        # Output results
        print(f"One-Way ANOVA F-statistic: {f_statistic} ",file=file)
        print(f"One-Way ANOVA P-value: {p_value:.10f}", file=file)

        # Interpretation
        if p_value < 0.05:
            print("Significant differences exist between the groups.", file=file)
        else:
            print("No significant differences between the groups.", file=file)
            
        group = [source, destination, protocol, length]
        # Perform Tukey's HSD test
        tukey_result = tukey_hsd(*group)
        print(f"Tukey resutls:\n {tukey_result}", file=file)
        
    
    def mannWhitneyU(self, data_scaled, features, file):
        source = data_scaled.Source_address
        destination = data_scaled.Destination_address
        protocol = data_scaled.Destination_address
        length = data_scaled.Length
        
        stat, p = mannwhitneyu(source, features, alternative='two-sided')
        print(f"\nMann-Whitney U Test Statistic on Source and Features: {stat}, p-value: {p:.6f}", file=file)
        stat, p = mannwhitneyu(destination, features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statisticon on destination and feature: {stat}, p-value: {p:.6f}", file=file)
        stat, p = mannwhitneyu(protocol, features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statistic on protocol and features: {stat}, p-value: {p:.6f}", file=file)
        stat, p = mannwhitneyu(length ,features, alternative='two-sided')
        print(f"Mann-Whitney U Test Statistic on length and features: {stat}, p-value: {p:.6f}", file=file)
    
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
    
    
    def t_TestAnd_WilcoxTest(self, item, data, feature, name, file):
        data = data.iloc[:, item]
        
        print("\nTwo-group t-test and Wilcox Test for {name}", file=file)
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
                    plt.title(f"Source Address and {name} OLS Regression: Observed vs Predicted")
                    plt.legend()
                    plt.grid(False)
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
                    plt.title(f"Length and {name} OLS Regression: Observed vs Predicted")
                    plt.legend()
                    plt.grid(False)
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
        plt.grid(False)
        plt.show()
                

class Omega_Metric:
           
    def calculate_mcdonalds_omega(self, data):
        """
        Calculate McDonald's Omega for a dataset.

        Parameters:
         - data: A 2D NumPy array or Pandas DataFrame where rows are 
         observations and columns are items.

        Returns:
         - omega: McDonald's Omega reliability coefficient.
        """
        # Perform factor analysis to extract loadings
        fa = FactorAnalyzer(n_factors=1, rotation=None, method='ml')
        fa.fit(data)
        loadings = fa.loadings_[:, 0]  # Extract the first factor loadings
        # Calculate specific variances (uniqueness)
        uniqueness = fa.get_uniquenesses()

        # Compute McDonald's Omega
        numerator = np.sum(loadings) ** 2
        denominator = numerator + np.sum(uniqueness)
        omega = numerator / denominator
        return omega

class ComputeModelMetrics:
    def calcuateTrueFalse(self, model, num_classes, cm, features, file):
        print("\nComputing the models FPR, TPR, and confusion matrix metrics", file=file)
        true_negatives = []
        false_positives= []
        tn = []
        tn_rates = []
        fn_rates = []
        fp_rates = []
        
        for i in range(num_classes):
            # Exclude the row and column of the current class
            mask = np.ones(cm.shape, dtype=bool)
            mask[i, :] = False
            mask[:, i] = False
            # Sum the remaining elements
            tn = cm[mask].sum()
            true_negatives.append(tn)
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            
            # False Positives: Sum of the current column except the diagonal element
            false_positives = np.sum(cm[:, i]) - cm[i, i]
            # TN Rate calculation
            tn_rate = true_negatives / (tn + false_positives)
            tn_rates.append(tn_rate)
            
            # False Negatives: Sum of the row excluding the diagonal element
            false_negatives = np.sum(cm[i, :]) - cm[i, i]
            true_positives = cm[i, i]
            
            # FN Rate calculation
            fn_rate = false_negatives / (false_negatives + true_positives)
            fn_rates.append(fn_rate)
            fp_rate = false_positives / (false_positives + tn)
            fp_rates.append(fp_rate)
            
        # Summarize (e.g., average FPR across all classes)
        average_fpr = np.mean(fp_rates)
        average_fnr = np.mean(fn_rate)
        averagte_tn = np.mean(true_negatives)
        average_tp = np.mean(true_positives)
        
        # Evalute the fn_rates and fp_rates using the z-test
        hypothesized_fnr = 0.25  # Hypothesized false negative rate or false positive rate
        sample_size = features
        # Calculate the standard error for the false negative rate
        std_error = np.sqrt((hypothesized_fnr * (1 - hypothesized_fnr)) / sample_size)

        # Calculate the z-statistic for the false positiver rate
        z_statistic = (average_fpr - hypothesized_fnr) / std_error

        # Calculate the p-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

        # Output results
        print(f"\n{model} Z-Statistic: {z_statistic:.4f} P-Value: {p_value:.4f}", file=file)

        # Interpretation
        if p_value < 0.05:
            print("The difference in FPR is statistically significant (p < 0.05).", file=file)
        else:
            print("The difference in FPR is not statistically significant (p >= 0.05).", file=file)
        
        # Calculate the z-statistic
        z_statistic = (average_fnr - hypothesized_fnr) / std_error

        # Calculate the p-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

        # Output results
        print(f"\n{model} Z-Statistic: {z_statistic:.4f} P-Value: {p_value:.4f}", file=file)

        # Interpretation
        if p_value < 0.05:
            print("The difference in FNR is statistically significant (p < 0.05).", file=file)
        else:
            print("The difference in FNR is not statistically significant (p >= 0.05).", file=file)

        print(f"\n{model} Average FPR: {average_fpr}", file=file)
        print(f"{model} Average FNR: {average_fnr}", file=file)
        
        print(f"\n{model} Average True Postive for each class: {average_tp}\n", file=file)
        #print(f"\n{model} Average True Negative Rates for each class: {averagte_tn}\n", file=file)
        print(f"\n{model} True Negative Rates for each class:\n{tn_rates}", file=file)
        print(f"\n{model} False Negative Rates for each class:\n{fn_rates}", file=file)
        print(f"\n{model} False Positive Rates for each class:\n{fp_rates}", file=file)
        
            
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
    om_metric = Omega_Metric()
    
    try:
        data_results, analysis_file = feature_selection.open_file()
        data = data_results
        file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/rfecv_dt_gnb_wPorts_12.txt", "a")
        #file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output.txt", "a")
        # for debuging
        #file = open("C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/output.txt", "a")
        print(f"Data under analysis: {analysis_file}", file=file)
        data_scaled, target_data, data_new_scaled = feature_selection.standardize_data(data)
        #calculate Cronbach's Alpha and corresponding 99% confidence interval
        df = data_scaled[['Source_address', 'Destination_address', 'protocol_converted']]
       
        alpha = pg.cronbach_alpha(data=df, ci=.99)
        print(f"First three Scaled Data Cronbach alpha: {alpha}", file=file)
        
        #calculate Cronbach's Alpha and corresponding 99% confidence interval
        df = data_scaled[['Length', 'info_converted']]
       
        alpha = pg.cronbach_alpha(data=df, ci=.99)
        print(f"Length and Info Scaled Data Cronbach alpha: {alpha}", file=file)
        
        df = data[['Source_address', 'Destination_address']]
       
        alpha = pg.cronbach_alpha(data=df, ci=.99)
        print(f"Data Cronbach alpha: {alpha}", file=file)
        
        df1 = data[['Source_address', 'Destination_address']]
        # Since the data violates normal distribution use the Spearman-brown test for 
        # testing the data's internal consistency
        # Split the items into two halves (e.g., first half and second half)
        half1 = df1.iloc[:, :len(df1.columns)//2].sum(axis=1)
        half2 = df1.iloc[:, len(df1.columns)//2:].sum(axis=1)

        # Calculate the Pearson correlation between the two halves
        correlation, _ = pearsonr(half1, half2)

        # Apply the Spearman-Brown prophecy formula
        spearman_brown_reliability = 2 * correlation / (1 + correlation)
        print(f"Data Pearson Correlation: {correlation:.4f}", file=file)
        print(f"Data Spearman-Brown Reliability: {spearman_brown_reliability:.4f}", file=file)
        
        df1 = data_scaled[['Source_address', 'info_converted']]
        # Since the data violates normal distribution use the Spearman-brown test for 
        # testing the data's internal consistency
        # Split the items into two halves (e.g., first half and second half)
        half1 = df1.iloc[:, :len(df1.columns)//2].sum(axis=1)
        half2 = df1.iloc[:, len(df1.columns)//2:].sum(axis=1)

        # Calculate the Pearson correlation between the two halves
        correlation, _ = pearsonr(half1, half2)

        # Apply the Spearman-Brown prophecy formula
        spearman_brown_reliability = 2 * correlation / (1 + correlation)
        print(f"Data Scaled Pearson Correlation: {correlation:.4f}", file=file)
        print(f"Data Scaled Spearman-Brown Reliability: {spearman_brown_reliability:.4f}", file=file)
        
        x = data['Source_address']
        #y = data['Destination_address']
        y = data['info_converted']
        # Using the spearman ranking test
        # Perform Spearman rank correlation test
        correlation, p_value = spearmanr(x, y)
        print(f"Data Spearman ranking correlation coefficient: {correlation}, P-value: {p_value:.6f}", file=file)

        x = data_scaled['Source_address']
        y = data_scaled['info_converted']
        # Using the spearman ranking test
        # Perform Spearman rank correlation test
        correlation, p_value = spearmanr(x, y)

        print(f"Data Scaled Spearman ranking correlation coefficient: {correlation}", file=file)
        print(f"Data Scaled P-value: {p_value:.6f}", file=file)
        
        l = data_scaled.shape[1]
        y = data_scaled['info_converted']
        print("Spearman Correlation accesses each item and info_converted", file=file)
        for i in range(l):
            x = data_scaled.iloc[:,i]
            correlation, p_value = spearmanr(x, y)
            print(f"Data Scaled Spearman ranking correlation coefficient: {correlation}, P-value: {p_value:.6f}", file=file)
           

        y_col = "Unnamed: 0"
        
        d = data_scaled.drop([y_col], axis = 1)
        
        y1_col = "number"
        
        d = d.drop([y1_col], axis = 1)
        
        corr_matrix = pd.DataFrame(d.corr(method="spearman"))
        
        # Visualize the correlation matrix
        sns.heatmap(corr_matrix, linewidths=0.5, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("DataSet Spearman Correlation Matrix ")
        plt.show()

        y = data_scaled[['Source_address', 'Destination_address', 'protocol_converted']]
        y = y.to_numpy()
        
        omega = om_metric.calculate_mcdonalds_omega(y)
        print(f"\nMcDonald's Omega: {omega:.4f}", file=file)

        group1 = data_scaled['Source_address']
        group2 = data_scaled['Destination_address']
        group3 = data_scaled['protocol_converted']
        group4 = data_scaled['info_converted']
        #group5 = data_scaled['Flags']
        group5 = data_scaled['Length']
        data_analysis.kruskalWallis(group1, group2, group3, group4, group5, file)
        
        print("Done performing statistical testing on the dataset", file=file)
        
        row_count = len(data_scaled)
        print(f"Number of rows: {row_count}", file=file)
        mean1 = statistic.mean(data['Source_address'])
        mean2 = statistic.mean(data['Destination_address'])
        mean3 = statistic.mean(data['protocol_converted'])
        mean4 = statistic.mean(data['Length'])
        mean5 = statistic.mean(data['info_converted'])
        
        print(f"Datasets Mean: {mean1} {mean2} {mean3} {mean4} {mean5}", file=file)
        
        std1 = statistic.stdev(data['Source_address'])
        std2 = statistic.stdev(data['Destination_address'])
        std3 = statistic.stdev(data['protocol_converted'])
        std4 = statistic.stdev(data['Length'])
        std5 = statistic.stdev(data['info_converted'])
        
        print(f"Standard Deviation: {std1} {std2} {std3} {std4} {std5}", file=file)
        
        features, selected_features_names, y_pred_rfecv, dt_predictions, gnb_predictions, y_pred_voting, y_pred_stacking, testing_data, group3 = feature_selection.extract_features(data_scaled, file)
        #with open('C:/PhD/DIS9903A/ConductExperiment/DataCollection/Source/Combined_feartures_output.txt', 'a') as feature_file:
        #    for item in features:
        #       feature_file.write(f"{item}\n")
        #feature_file.close()
                
        data_analysis.oneAnova_MFeatures(data_scaled, features, selected_features_names, file)
        print("Indepentend t-test on scaled data and RFRCV selected features", file=file)
        data_analysis.tTest_NFeatures(data_scaled, features, selected_features_names, file)
        
        # see if the models predicitions has statstical significance following normality
        print("\nOne ANOVA Statistical Significance Data Scaled and RFECV features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_rfecv, file)
        print("One ANOVA Statistical Significance Data Scaled and Decision Tree features identified", file=file)
        data_analysis.oneAnova(data_scaled, dt_predictions, file)
        print("One ANOVA Statistical Significance Data Scaled and GNB features identified", file=file)
        data_analysis.oneAnova(data_scaled, gnb_predictions, file)
        print("One ANOVA Statistical Significance Data Scaled and DT with GNB Voting stacking features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_voting, file)
        print("One ANOVA Statistical Significance Data Scaled and DT with GNB Combined features identified", file=file)
        data_analysis.oneAnova(data_scaled, y_pred_stacking, file)
        
        print("\nKruskal-Wallis Statistical Significance Data Scaled and RFECV features identified", file=file)
        data_analysis.kruskalWallisM(data_scaled, y_pred_rfecv, file)
        print("Kruskal-Wallis Statistical Significance Data Scaled and Decision Tree features identified", file=file)
        data_analysis.kruskalWallisM(data_scaled, dt_predictions, file)
        print("Kruskal-Wallis Statistical Significance Data Scaled and GNB features identified", file=file)
        data_analysis.kruskalWallisM(data_scaled, gnb_predictions, file)
        print("OKruskal-Wallis  Statistical Significance Data Scaled and DT with GNB Voting stacking features identified", file=file)
        data_analysis.kruskalWallisM(data_scaled, y_pred_voting, file)
        print("Kruskal-Wallis  Statistical Significance Data Scaled and DT with GNB Combined features identified", file=file)
        data_analysis.kruskalWallisM(data_scaled, y_pred_stacking, file)
     
        for i in range(features.shape[1]):
            name = features.columns[i]
            data_analysis.t_TestAnd_WilcoxTest(i, features, y_pred_rfecv, name, file) 
        
        print("\nTwo-group t-test for X_transform_Selected (RFECV features) and DT with GNB features identified", file=file)
        for i in range(testing_data.shape[1]):
            group1 = testing_data[:,i]
            group2 = y_pred_stacking
            t_stat, p_value = ttest_ind(group1, group2)
            print(f"T-statistic: {t_stat}, P-value: {p_value:.6f}", file=file)
            if p_value < 0.05:
                  print("Significant differences exist between the groups.", file=file)
            else:
                  print("No significant differences between the groups.", file=file)
            
        group = pd.DataFrame(testing_data)    
        print("\nTwo-group t-test for Source Address and RFECV features and DT with GNB features identified", file=file)
        t_stat, p_value = ttest_ind(group.iloc[:,0], y_pred_stacking)
        print(f"T-statistic: {t_stat}, P-value: {p_value:.6f}", file=file)
        if p_value < 0.05:
              print("Significant differences exist between the groups.", file=file)
        else:
              print("No significant differences between the groups.", file=file)
        
        stat, p = wilcoxon(group3.iloc[:,1], y_pred_stacking, alternative='two-sided')
        print(f"Two-Sided Wilcox Signed Rank T-statistic: {stat}, P-value: {p:.6f}", file=file)
        
        d2 = np.round(group3.iloc[:,1] - y_pred_stacking, decimals=3)
        stat, p = wilcoxon(d2, alternative='greater')
        print(f"Greater Wilcox Signed Rank T-statistic: {t_stat}, P-value: {p_value:.6f}", file=file)
        
        correlation, p_value = spearmanr(group3.iloc[:,1], y_pred_stacking)
        print(f"\nEnsembled Model Spearman ranking correlation coefficient: {correlation}, P-value: {p_value}", file=file)

        feature_selection.stop_all(file)
        file.close()
        
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':

    main()

    


