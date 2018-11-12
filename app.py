from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import json
import numpy as np
from model import Model
import pandas as pd

app = Flask(__name__)
api = Api(app)

model = Model()

with open('clf.pkl', 'rb') as f:
	model.clf = pickle.load(f)
			
with open('ss.pkl', 'rb') as f:
	model.ss = pickle.load(f)
			
with open('encoder.pkl', 'rb') as f:
	model.encoder = pickle.load(f)
			
with open('pf.pkl', 'rb') as f:
	model.pf = pickle.load(f)
		
with open('bins.json', "rb",) as f:
	model.bins = pd.read_csv('bins.csv').to_dict(orient = 'list')

columns = ['fixed acidity', 'volatile acidity', 
        'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
        'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol']

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('vector')


class Predict(Resource):
    def get(self):        
        args = parser.parse_args()
        vector = eval(args['vector'])
        data = {row:[t] for (row, t) in zip(columns, vector)}
        data = pd.DataFrame.from_dict(data)
        data['total sulfur dioxide'] = np.log(np.log(data['total sulfur dioxide']))	    
        data['alcohol'] = np.log(data['alcohol'])
        data['chlorides'] = np.log(data['chlorides'])
        data['free sulfur dioxide'] = np.log(data['free sulfur dioxide'])
        data['residual sugar'] = np.log(data['residual sugar'])
        poly = model.get_poly(data)
        binning = model.get_binning(data)
        one_hot = model.get_encoder(binning)
	
        angle = model.get_angle(data, one_hot)
	
        data =  np.hstack((data, one_hot, poly, angle))
	
        data = model.get_standartscaler(data)

        pred = model.clf.predict(data)
        print(pred[0])
        output = json.dumps({'prediction': str(pred[0])})
		
        return output


api.add_resource(Predict, '/')


if __name__ == '__main__':
	app.run(debug=True)
