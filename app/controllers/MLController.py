'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-25 15:19:33
LastEditors: Ian Yang
LastEditTime: 2024-04-26 20:03:29
'''
import joblib
from app.model.TrafficState import TrafficState
from app.model.models import DNN, GRUModel
import torch
import os

class DNNController:
    def __init__(self, device):
        self.filename = '../training/classifier/model.pth'
        self.device = device
        
        
        
    def predict(self,features):
        model = DNN(5,64,2).to(self.device)
        model.load_state_dict(torch.load(self.filename))
        model.eval()
        with torch.no_grad():
            output = model(features)
        # Get the prediction
        _, predicted = torch.max(output.data, 1)
        print(predicted)
        prediction = predicted.item()
 
        return TrafficState(prediction)

    
class GRUController:
    def __init__(self, device):
        self.path = os.path.dirname(__file__)
        self.filename = self.path + '/model_gru.pth'
        self.device = device
        
        
    def predict(self,features):
        model = GRUModel(5,64,3).to(self.device)
        model.load_state_dict(torch.load(self.filename, map_location=self.device))
        model.eval()
        with torch.no_grad():
            output = model(features)
        # Get the prediction
        _, predicted = torch.max(output.data, 1)
        print(predicted)
        prediction = predicted.item()
 
        return TrafficState(prediction)
