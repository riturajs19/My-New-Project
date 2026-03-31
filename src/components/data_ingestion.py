import os
import sys
from dataclasses import dataclass

import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")

        
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Split training and testing input data")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(n_estimators=100, verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            logging.info("Models initialized")

        
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            logging.info(f"Model Report: {model_report}")

          
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name} with score {best_model_score}")

      
            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

         
            best_model.fit(X_train, y_train)

           
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully")

         
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.info("Error occurred in Model Trainer")
            raise CustomException(e, sys)