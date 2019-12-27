# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:34:25 2019

@author: Sylwek Szewczyk
"""
import pandas as pd 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

class Parkinson:
    
    def __init__ (self, db):
        self.db = db
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
    
    def showData(self):
        print(self.db.head())
        print(self.db.describe())
    
    def splitData(self, testsize):
        scaler = MinMaxScaler((-1,1))
        X = scaler.fit_transform(self.db.iloc[:, self.db.columns != 'status'].values[:, 1:])
        y = self.db.iloc[:, self.db.columns == 'status'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = testsize, random_state = 100)
    
    def solve(self):
        self.classifier = XGBClassifier()
        self.classifier.fit(self.X_train, self.y_train.ravel())
        self.y_pred = self.classifier.predict(self.X_test)
        print(f'Model has {round(accuracy_score(self.y_test, self.y_pred),2)*100}% accuracy.')
        
    def predict(self, data):
        return self.classifier.predict(data)
    
    @classmethod
    def getData(cls, df):
        return cls(db = pd.read_csv(df))

r = Parkinson.getData('parkinsons.data')
r.splitData(0.3)
r.solve()