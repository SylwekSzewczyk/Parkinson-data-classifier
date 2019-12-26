# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:34:25 2019

@author: Sylwek Szewczyk
"""

import pandas as pd 
from xgboost import XGBClassifier

class Parkinson:
    
    def __init__ (self, db):
        self.db = db
    
    def showData(self):
        print(self.db.head())
        print(self.db.describe())
    
    @classmethod
    def getData(cls, df):
        return cls(db = pd.read_csv(df))

r = Parkinson.getData('parkinsons.data')