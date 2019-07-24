## Prediction of Trees from local model
import utilities
import cv2
import matplotlib.pyplot as plt  

class Tree_Model:
    
    def __init__(self, model_path):
        self.model_path = model_path
        
        #read and load models
        self.read_config()
        self.load_model()
    
    def read_config(self):
        self.config = utilities.read_config()
        
    def load_model(self):
        self.model = utilities.read_model(self.model_path, self.config)
        
    def predict_image(self, image):
        prediction = utilities.predict_image(self.model,image)
        
        #return in RGB order
        prediction = prediction[:,:,::-1]
        
        return prediction
      
        