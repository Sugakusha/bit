import numpy as np
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from functools import reduce
import lightgbm as lgbm
import pickle


class RFLGB(BaseEstimator, ClassifierMixin):

    def __init__(self, seed=0):
        self.models = [
            lgbm.LGBMClassifier(class_weight='balanced', max_bin=210, max_depth=10,
                                min_child_samples=6, min_child_weight=5,
                                n_estimators=40, n_jobs=-1, num_leaves=93, random_state = 1),
            lgbm.LGBMClassifier(class_weight='balanced', max_bin=220, max_depth=11,
                                min_child_samples=6, min_child_weight=5,
                                n_estimators=40, n_jobs=-1, num_leaves=94, random_state = 42),
            lgbm.LGBMClassifier(class_weight='balanced', max_bin=211, max_depth=12,
                                min_child_samples=5, min_child_weight=5,
                                n_estimators=40, n_jobs=-1, num_leaves=95, random_state = 651),
            lgbm.LGBMClassifier(class_weight='balanced', max_bin=210, max_depth=13,
                                min_child_samples=5, min_child_weight=5,
                                n_estimators=40, n_jobs=-1, num_leaves=96, random_state = 561),
            RandomForestClassifier(class_weight='balanced',
                                   criterion='gini', max_depth=10, max_features=566, min_samples_split=2,
                                   n_estimators=40, n_jobs=-1, random_state=1
                                   ),
            RandomForestClassifier(class_weight='balanced',
                                   criterion='gini', max_depth=11, max_features=600, min_samples_split=3,
                                   n_estimators=45, n_jobs=-1, random_state=314
                                   ),
            RandomForestClassifier(class_weight='balanced',
                                   criterion='gini', max_depth=12, max_features=570, min_samples_split=5,
                                   n_estimators=41, n_jobs=-1, random_state=974
                                   ),
            RandomForestClassifier(class_weight='balanced',
                                   criterion='gini', max_depth=20, max_features=560, min_samples_split=4,
                                   n_estimators=43, n_jobs=-1, random_state=3
                                   )
                      ]
    def fit(self, X, y=None):
        self.n_class = len(np.unique(y))
        for t, clf in enumerate(self.models):
            #print('train ', t)
            clf.fit(X, y)
        return self

    def predict(self, X):
        summa = np.zeros((X.shape[0], self.n_class))
        for i, clf in enumerate(self.models):
            print(clf.predict_proba(X).shape)
            summa += clf.predict_proba(X) 
        return clf.classes_[np.argmax(summa, axis = 1)]
        
        
class Model:
	def __init__(self, degree = 2):
		self.pf = PolynomialFeatures(degree = degree)
		self.encoder = OneHotEncoder(sparse=False)
		self.ss = StandardScaler()
		self.clf = RFLGB()
		
	def poly_fit(self, data):
		self.pf.fit(data)
		
	def encoder_fit(self, data):
		self.encoder.fit(data)
		
	def binning_fit(self, data, bin = 4):
		self.bins = {}
		for i, row in enumerate(data.columns):
			binning = np.linspace(data[row].min(), data[row].max(), bin)
			self.bins[row] = binning
			
	def standarscaler_fit(self, data):
		self.ss.fit(data)
			
	def get_encoder(self, data):
		return self.encoder.transform(data)
				
	def get_poly(self, data):
		return self.pf.transform(data)
		
	def get_binning(self, data):
		matrix_binning = np.zeros_like(data)
		for i, row in enumerate(data.columns):
			matrix_binning[:, i] = np.digitize(data[row], bins=self.bins[row])
		return matrix_binning
			
	def get_angle(self, data, enc):
		return np.hstack([enc * data[row][:, np.newaxis] for row in data.columns])
		
	def get_standartscaler(self, data):
		return self.ss.transform(data)
		
	def train(self, data, target):
		self.clf.fit(data, target)
		
	def predict(self, data):
		return self.clf.predict(data)
	
		
	def pickle_clf(self, path='clf.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.clf, f)
			print("Pickled classifier at {}".format(path))
			
	def pickle_standartscaler(self, path='ss.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.ss, f)
			print("Pickled ss at {}".format(path))
			
	def pickle_encoder(self, path='encoder.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.encoder, f)
			print("Pickled encoder at {}".format(path))
			
	def pickle_pf(self, path='pf.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.pf, f)
			print("Pickled pf at {}".format(path))
			
	def json_bins(self, path='bins.json'):
		pd.DataFrame.from_dict(self.bins).to_csv('bins.csv')	













