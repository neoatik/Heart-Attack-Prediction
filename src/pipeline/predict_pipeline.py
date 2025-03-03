import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
       try:
            model_path = 'artifacts\\model.pkl' 
            preprocessor_path = 'artifacts\\proprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
       except Exception as e:
           raise CustomException(e,sys)
       

    

class CustomData:
    def __init__(self,
                 Cholesterol:int, 
                 Systolic:int,
                 Diabetes:int,
                 BP_Ratio:float,
                 Triglycerides:int,
                 Exercise_Hours_Per_Week:float):
        
        self.Cholestrol = Cholesterol
        self.Systolic = Systolic
        self.Diabetes = Diabetes
        self.BP_Ratio = BP_Ratio
        self.Triglycerides = Triglycerides
        self.Exercise_Hours_Per_Week = Exercise_Hours_Per_Week
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Cholesterol":[self.Cholestrol],
                "Systolic":[self.Systolic],
                "Diabetes":[self.Diabetes],
                "BP_Ratio":[self.BP_Ratio],
                "Triglycerides":[self.Triglycerides],
                "Exercise_Hours_Per_Week":[self.Exercise_Hours_Per_Week]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        