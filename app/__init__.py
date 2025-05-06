from flask import Flask
import pickle
import json

app = Flask(__name__)

with open('../models/Heart_Disease.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../models/features.json', 'r') as features_file:
    features = json.load(features_file)

from app import main