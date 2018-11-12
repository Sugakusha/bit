import pandas as pd
import numpy as np
from model import Model


def build():
	model = Model()
	data = pd.read_csv('data/winequality-red.csv', delimiter = ';')
	target = data.quality.copy()
	data.drop('quality', axis=1, inplace=True)
	
	data['total sulfur dioxide'] = np.log(np.log(data['total sulfur dioxide']))
	data['alcohol'] = np.log(data['alcohol'])
	data['chlorides'] = np.log(data['chlorides'])
	data['free sulfur dioxide'] = np.log(data['free sulfur dioxide'])
	data['residual sugar'] = np.log(data['residual sugar'])
	
	model.poly_fit(data)
	poly = model.get_poly(data)
	
	model.binning_fit(data)
	binning = model.get_binning(data)

	model.encoder_fit(binning)
	one_hot = model.get_encoder(binning)
	
	angle = model.get_angle(data, one_hot)
	
	data =  np.hstack((data, one_hot, poly, angle))
	
	model.standarscaler_fit(data)
	data = model.get_standartscaler(data)
	
	model.train(data, target)
	
	model.pickle_clf()
	model.pickle_encoder()
	model.pickle_pf()
	model.pickle_standartscaler()
	model.json_bins()
	
if __name__ == "__main__":
	build()
