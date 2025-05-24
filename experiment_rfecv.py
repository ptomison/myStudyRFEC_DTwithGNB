# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:50:53 2025

@author: pauli
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
#import csv
#import sys
#import re
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #, LinearRegression, 
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB



class RFECV_EXPERIMENT:
    
    def init(self):
        self.data = [0][0]
        self.x = [0][0]
        self.y = [0]
        
        
    def open_file(self):
        base_dir = os.path.dirname(__file__)  # Directory of the current script
    
        data_path = os.path.join(base_dir, "/PhD/DIS9903A/Week 7/")
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
                    
    def extract_features(self, data):
        print("Extracting the features using RFECV")
        # first drop the first two columns of the dataset since it just contains the number of elements
        #y_col = "Unnamed: 0"
        
        # = data[y_col]
        
        #data = data.drop([y_col], axis = 1)
        
        #y1_col = "number"
        y1_col = "No"
        
        y = data[y1_col]
        
        y = y.to_frame()
        
        #y_col = "info_converted"
        y_col = "Source_address"
        #y_col = "No"  # Use this only when using the IMB SPPS data instead of the R data preprocessed data
        y = data[y_col]
        #print(y)
        
        X = data.drop([y_col], axis = 1)
        
        model = RandomForestClassifier(random_state=42)
       
        # Split the data for the RF model 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        print("Making Data Classification Complete")
   
        # Setup the cross_validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Cross-validation
        
        #Initializing RFE model
        rfecv = RFECV(estimator=model, step=1, min_features_to_select=1, scoring='accuracy', cv=cv, n_jobs=-1)
        
        #features = rfecv.fit(X_train, y_train)
        features = rfecv.fit_transform(X, y)
        #print(features)
        
        print("RFECV model fitting complete")
        
        # Step 5: Evaluate the RFECV model using StrartifiedKFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=cv, scoring='accuracy')
        # Step 6: Print results
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Selected features: {rfecv.support_}")
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"RFECV ranking: {rfecv.ranking_}")
        
        # Step 5a: Evaluate the RFECV model using KFold cross_val_score
        scores = cross_val_score(rfecv.estimator_, X, y, cv=kf, scoring='accuracy')
        # Step 6: Print results
        print(f"KF Optimal number of features: {rfecv.n_features_}")
        print(f"KF Selected features: {rfecv.support_}")
        print(f"KF Cross-validation scores: {scores}")
        print(f"KF Mean accuracy: {scores.mean():.4f}")
        print(f"KF RFECV ranking: {rfecv.ranking_}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
       
        print("RF Classifier Accuracy:", accuracy)
        print("RF Classifier Precision:", precision)
        print("RF Classifier Recall:", recall)
        
        return features

    def stop_all(self):
        print("Stopping the feature extraction and DT with GNB Classification")
        pass
    
class DT_with_GNB:
    def init(self, data):
        self.data = data
        self.x = [0][0]
        self.y = [0]
        
    def decicsionTree(self, data):
        
        # Remove the columns with the number listing from the data preprocessing step
        #y_col = "Unnamed: 0"
        
        #y = data[y_col]
        
        #data = data.drop([y_col], axis = 1)
        
        y1_col = "No"
        
        y = data[y1_col]
        
        y = y.to_frame()
        
        X = data.drop([y1_col], axis = 1)
        
        y_col = 'Source_address'
        #y_col = 'protocol_convert'
        #y_col = "info_converted"
        y = data[y_col]
        
        #y = np.ravel(y)
        #print(y)
        
        #print(type(X))
        #print(type(y))
       
        # Random forest classifier, to classify dogs into big or small
        model = RandomForestClassifier()
        
        # Find the number of members in the least-populated class, THIS IS THE LINE WHERE THE MAGIC HAPPENS :)
        leastPopulated = [x for d in set(list(y)) for x in list(y) if x == d].count(min([x for d in set(list(y)) for x in list(y) if x == d], key=[x for d in set(list(y)) for x in list(y) if x == d].count))
        fOne = cross_val_score(model, X, y, cv=leastPopulated, scoring='f1_weighted')

        # We print the F1 score here
        print(f"Average F1 score during cross-validation: {np.mean(fOne)}")

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        #X_train, y_train, X_test, y_test = iterative_train_test_split(X, y1, test_size = 0.2)
        
        # Decision Tree Classifier
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)

        # Gaussian Naive Bayes Classifier
        gnb_model = GaussianNB()
        gnb_model.fit(X_train, y_train)
        gnb_predictions = gnb_model.predict(X_test)
        
        # Evaluate Decision Tree
        print("Decision Tree Classifier:")
        print(f"DT Accuracy: {accuracy_score(y_test, dt_predictions):.4f}")
        print(classification_report(y_test, dt_predictions, zero_division=1.0))

        # Evaluate Gaussian Naive Bayes
        print("Gaussian Naive Bayes Classifier:")
        print(f"GNB Accuracy: {accuracy_score(y_test, gnb_predictions):.4f}")
        print(classification_report(y_test, gnb_predictions, zero_division=1.0))
        
        # Combine using VotingClassifier
        voting_clf = VotingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], voting='hard')
        votingscore = voting_clf.fit(X_train, y_train)
        print(votingscore.predict(X))
        
        y_pred = voting_clf.predict(X_test)
        print("VotingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred))
        print("VotingClassifier DT with GNB confusion matris: ", confusion_matrix(y_test, y_pred))
        
        # Combine using StackingClassifier
        stacking_clf = StackingClassifier(estimators=[('dt', dt_model), ('gnb', gnb_model)], final_estimator=LogisticRegression())
        stacking_clf.fit(X_train, y_train).score(X_test, y_test)
        
        # Evaluate
        y_pred = stacking_clf.predict(X_test)
        print("StackingClassifier DT with GNB Accuracy:", accuracy_score(y_test, y_pred))
        print("StackingClassifier DT with GNB Confusion Matrix:", confusion_matrix(y_test, y_pred))
       
       
    
def main():
    print("Run the script")
    features = [0];
    feature_selection = RFECV_EXPERIMENT()
    attack_classification = DT_with_GNB()
    
    try:
        data_results = feature_selection.open_file()
        data = data_results
        #data_scaled = feature_selection.standardize_data(data)
        #features = feature_selection.extract_features(data_scaled)
        features = feature_selection.extract_features(data)
        print(f" RFECV features identified : {features}")
        attack_classification.decicsionTree(data)
        feature_selection.stop_all()
    except KeyboardInterrupt:
        feature_selection.stop_all()

if __name__ == '__main__':


    main()
