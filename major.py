#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class IrisDataProcessor:
    def __init__(self):
        self.data = load_iris()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        self.df = pd.DataFrame(
            data=np.c_[self.data['data'], self.data['target']],
            columns=self.data['feature_names'] + ['target']
        )

        features = self.df[self.data['feature_names']]
        scaled_features = self.scaler.fit_transform(features)
        self.df[self.data['feature_names']] = scaled_features

        X = self.df[self.data['feature_names']]
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def get_feature_stats(self):
        # Basic statistical analysis
        return self.df.describe()


# In[3]:


import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
        mlflow.set_experiment("Iris_Classification")

    def run_experiment(self):
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                scores = cross_val_score(
                    model,
                    self.data_processor.X_train,
                    self.data_processor.y_train,
                    cv=5
                )
                accuracy = scores.mean()

                model.fit(self.data_processor.X_train, self.data_processor.y_train)

                mlflow.log_metric("accuracy", accuracy)

    def log_results(self):
        mlflow.end_run()


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn import datasets

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.model = LogisticRegression()

    def quantize_model(self):
        self.model = self.model.fit(
            self.experiment.data_processor.X_train, 
            self.experiment.data_processor.y_train
        )
        self.model.coef_ = np.round(self.model.coef_).astype(int)

    def run_tests(self):
        assert self.model is not None, "Model is not initialized"
        assert len(self.experiment.data_processor.X_test) > 0, "Test set is empty"
        print("All tests passed!")


# In[5]:


def main():
    processor = IrisDataProcessor()
    processor.prepare_data()

    experiment = IrisExperiment(processor)
    experiment.run_experiment()

    optimizer = IrisModelOptimizer(experiment)
    optimizer.quantize_model()
    optimizer.run_tests()

if __name__ == "__main__":
    main()

