

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict
from sklearn.metrics import accuracy_score
from collections import defaultdict
from functools import reduce

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from src.jobs.etl import ETL
from src.utils.utils import get_logger

class Model:
    def __init__(self) -> None:
        self.etl = ETL()
        self.logger = get_logger('Model')
        self.target = 'is_returning_customer'
        self.output = os.path.join(os.getcwd(), 'src/output')
        self.param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 8),
              "min_samples_split": sp_randint(2, 8),
              "min_samples_leaf": sp_randint(2, 8),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
        self.rfs = defaultdict(dict)
        self.dfs_ones = []
        self.dfs_zeros = []

    def train_test(self, df):
        self.logger.info("Splitting into train and test..")
        y=df[self.target]
        X=df.loc[:, df.columns != self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
        return X_train, X_test, y_train, y_test


    def split_df_parts(self, X_train, y_train, for_ones=True):
       
        temp = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        dfy_1 = temp[temp.is_returning_customer == 1].reset_index(drop=True)
        dfy_0 = temp[temp.is_returning_customer == 0].reset_index(drop=True)

        if for_ones:
            self.logger.info("Splitting 1s into multiple dfs..")
            factor = (dfy_1.shape[0]//6)*4
            for i in range(factor, temp.shape[0] + factor, factor):
                if i == factor:
                    self.dfs_ones.append(dfy_0.iloc[:i, :].append(dfy_1).reset_index(drop=True))
                else:
                    self.dfs_ones.append(dfy_0.iloc[i-factor:i, :].append(dfy_1).reset_index(drop=True))
        else:
            self.logger.info("Splitting 0s into multiple dfs..")
            factor = (dfy_1.shape[0]//4)*6
            for i in range(factor, temp.shape[0] + factor, factor):
                if i == factor:
                    self.dfs_zeros.append(dfy_0.iloc[:i, :].append(dfy_1).reset_index(drop=True))
                else:
                    self.dfs_zeros.append(dfy_0.iloc[i-factor:i, :].append(dfy_1).reset_index(drop=True))


    def random_search(self, X_train, y_train) -> Dict:
        self.logger.info("Performing random_search")
        n_iter_search = 20
        clf = RandomForestClassifier(n_estimators=20)
        random_search = RandomizedSearchCV(clf, param_distributions=self.param_dist, n_iter=n_iter_search)
        random_search.fit(X_train, y_train)
        cv_results = pd.DataFrame(random_search.cv_results_)

        req_cols = ['rank_test_score', 'mean_test_score', 'params']
        random_forest_ht = cv_results[cv_results.rank_test_score <= 3][req_cols].sort_values(by='rank_test_score', ascending=True).reset_index(drop=True, inplace=False)
        best_params_rf = random_forest_ht.params[0]
        return best_params_rf

    def find_n_estimators(self, best_params, X_train, y_train, X_test, y_test) -> int:
        self.logger.info("Finding n_estimators..")
        scores_d = {}
        flipped = {} 
        for k in range(25, 325, 25):
            rfc = RandomForestClassifier(n_estimators=k, **best_params)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            scores_d[k] = round(accuracy_score(y_test, y_pred), 4)
  
        for key, value in scores_d.items(): 
            if value not in flipped: 
                flipped[value] = [key] 
            else: 
                flipped[value].append(key)

        best_params['n_estimators'] = min(flipped[sorted(flipped)[-1]])
        self.logger.info(f"n_estimators: {best_params['n_estimators']}")
        return best_params


    def train(self, best_params_rf, X_test, y_test, for_ones=True):
        self.logger.info("Training with mlflow tracking..")
        if for_ones:
            self.logger.info("Working on 1s..")
            dfs = self.dfs_ones.copy()
            key = 'ones'
            self.logger.info(f"Total dataframes: {len(dfs)}")
        else:
            self.logger.info("Working on 0s..")
            dfs = self.dfs_zeros.copy()
            key = 'zeros'
            self.logger.info(f"Total dataframes: {len(dfs)}")
        with mlflow.start_run(run_name='random_forest_dfs'): 
            for i in range(len(dfs)):
                cols = [i for i in dfs[i].columns if i != self.target]
                xtrain = dfs[i].loc[:, cols]
                ytrain = dfs[i][self.target]
                model = RandomForestClassifier(**best_params_rf)
                model.fit(xtrain, ytrain)
            
                scores = cross_val_score(model, xtrain, ytrain, cv=5)
                self.logger.info(f"Score: {scores.mean()}")
                y_pred = model.predict(X_test)
                auc_score = roc_auc_score(y_test, y_pred)

                self.rfs[str(i) + '_' + key]['randomforest'] = model
                self.rfs[str(i) + '_' + key]['ypred'] = y_pred

                # Use the area under the ROC curve as a metric.
                mlflow.log_metric('auc', auc_score)
                mlflow.log_metric('cross_val_score', scores.mean())
   
                signature = infer_signature(xtrain, model.predict(xtrain))
                mlflow.sklearn.log_model(model, f"random_forest_model_{i}", signature=signature)
        
    
    @staticmethod
    def combine_rfs(rf_a, rf_b):
        rf_a.estimators_ += rf_b.estimators_
        rf_a.n_estimators = len(rf_a.estimators_)
        return rf_a

    def ensemble(self):
        rfs_l = []
        for v in self.rfs.values():
            rfs_l.append(v['randomforest'])
        rf_combined = reduce(self.combine_rfs, rfs_l)
        return rf_combined

    def predict(self, rf_combined, X_test):
        self.logger.info("Predicting..")
        y_pred_combined = rf_combined.predict(X_test)
        pd.DataFrame(y_pred_combined, columns=[self.target]).to_csv(os.path.join(self.output, 'all_pred.csv'), index=False)
        return y_pred_combined

    @staticmethod
    def display_crosstab(y_pred_combined, y_test):
        print(pd.crosstab(y_pred_combined, y_test))

    def core(self):
        df = self.etl.core(save_file=True)
        X_train, X_test, y_train, y_test = self.train_test(df)
        self.split_df_parts(X_train, y_train)
        self.split_df_parts(X_train, y_train, for_ones=False)
        best_params_rf = self.random_search(X_train, y_train)
        best_params_rf1 = self.find_n_estimators(best_params_rf, X_train, y_train, X_test, y_test)
        self.logger.info(best_params_rf1)
        self.train(best_params_rf1, X_test, y_test)
        self.train(best_params_rf1, X_test, y_test, for_ones=False)
        rf_combined = self.ensemble()
        ypred = self.predict(rf_combined, df.loc[:, df.columns != self.target])
        self.display_crosstab(ypred, df[self.target])
        return ypred




    