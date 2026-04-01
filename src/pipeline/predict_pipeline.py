import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models, load_object

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def CustomData(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error occurred during prediction")
            raise CustomException(e, sys)