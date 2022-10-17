import math
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import log_loss, mean_squared_error
import scipy.stats as ss
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
import joblib
from feature_selector import FeatureSelector

from scipy.stats import chi2

def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()   


class PPIMBC(TransformerMixin, BaseEstimator):
    def __init__(self, model=None, p_val_thresh=0.05, num_simul=30, cv=0, simul_size=0.2, simul_type=0,
                 sig_test_type="non-parametric", random_state=None, n_jobs=-1, verbose=2):
        self.random_state = random_state
        if model is not None:
            self.model = model
        else:
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.p_val_thresh = p_val_thresh
        self.num_simul = num_simul
        self.simul_size = simul_size
        self.simul_type = simul_type
        self.sig_test_type = sig_test_type
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.MB = None
        self.feature_ranking = list()
        self.feat_imp_scores = list()
        self.col_ranking = list()
        self.score_ranking = list()
    def _feature_importance(self, data1, data2, Y, i):
        Y = np.ravel(Y)
        if self.simul_type == 0:
            x_train1, x_test1, y_train1, y_test1 = train_test_split(data1, Y, test_size=self.simul_size, random_state=i)
            x_train2, x_test2, y_train2, y_test2 = train_test_split(data2, Y, test_size=self.simul_size, random_state=i)
        else:
            x_train1, x_test1, y_train1, y_test1 = train_test_split(data2, Y, test_size=self.simul_size,
                                                                    random_state=i, stratify=Y)
            x_train2, x_test2, y_train2, y_test2 = train_test_split(data2, Y, test_size=self.simul_size,
                                                                    random_state=i, stratify=Y)
        self_labels2 = np.unique(y_train2)
        self_labels1 = np.unique(y_train1)
        if len(self_labels2) <= 2:
            self_labels2 = None
        if len(self_labels1) <= 2:
            self_labels1 = None
        model2 = clone(self.model)
        model1 = clone(self.model)
        model2.fit(x_train2, y_train2)
        if "SVC" in type(model2).__name__ or "RidgeClassifier" in type(model2).__name__:
            preds2 = model2.decision_function(x_test2)
            train_preds2 = model2.decision_function(x_train2)
        elif "RandomForestRegressor" in type(model2).__name__:
            preds2 = model2.predict(x_test2)
            train_preds2 = model2.predict(x_train2)
        else:
            preds2 = model2.predict_proba(x_test2)
            train_preds2 = model2.predict_proba(x_train2)
        train_total_loss = log_loss(y_train2, train_preds2, labels=self_labels2)
        total_loss = log_loss(y_test2, preds2, labels=self_labels2)
        model1.fit(x_train1, y_train1)
        if "SVC" in type(model1).__name__ or "RidgeClassifier" in type(model1).__name__:
            preds1 = model1.decision_function(x_test1)
            train_preds1 = model1.decision_function(x_train1)
        elif "RandomForestRegressor" in type(model1).__name__:
            preds1 = model2.predict(x_test1)
            train_preds1 = model2.predict(x_train1)
        else:
            train_preds1 = model1.predict_proba(x_train1)
            preds1 = model1.predict_proba(x_test1)

        train_left_loss = log_loss(y_train1, train_preds1, labels=self_labels1)
        left_loss = log_loss(y_test1, preds1, labels=self_labels1)

        return [total_loss, left_loss]

    def _PPI(self, X, Y, Z, col):
        X = np.reshape(X, (-1, 1))
        Z = np.array(Z)
        data1 = Z.copy()
        if np.size(Z, 1) == 1:
            data2 = np.concatenate((X, np.reshape(Z, (-1, 1))), axis=1)
        else:
            data2 = np.concatenate((X, Z), axis=1)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        p_values = parallel(delayed(self._feature_importance)(data1, data2, Y, i) for i in range(self.num_simul))
        p_values = np.array(p_values)
        testX, testY = p_values[:,0], p_values[:,1]
        sum_testX = sum(testX)
        sum_testY = sum(testY)
        flag = False
        if sum_testX > sum_testY:
           flag = True
        if self.sig_test_type == "parametric":
            t_stat, p_val = ss.ttest_ind(testX, testY, alternative="less", nan_policy="omit")
        else:
            t_stat, p_val = ss.wilcoxon(testX, testY, alternative="less", zero_method="zsplit")
        if col is None:
            return [p_val, flag]
        else:
            return [col, p_val, flag]
    def _grow(self, data, Y, MB):
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        Z_MB = data[MB].values
        Z_MB = np.array(Z_MB)
        feats_and_pval = parallel(delayed(self._PPI)(data[col].values, Y, Z_MB, col) for col in data.columns if col not in MB)
        feats_and_pval.sort(key = lambda x: x[1], reverse=False)
        return feats_and_pval[0]

    def _shrink(self, data, Y, Z, MB, MB_score):
        original_MB = MB.copy()
        original_MB_score = MB_score.copy()
        if Z[1] < self.p_val_thresh and Z[2] == False:
            original_MB.append(Z[0])
            original_MB_score.append(Z[1])
            remove = list()
            remove_score = list()
            if len(MB) < 1:
                return [i for i in original_MB], np.log(1 / Z[1])
            else:
                for col in MB:
                    MB_to_consider = [i for i in original_MB if i not in [col]]
                    for j in range(len(original_MB)):
                        if original_MB[j] == col:
                            index = j
                            break
                    p_val = self._PPI(data[col].values, Y, data[MB_to_consider].values, col)
                    if p_val[1] >= self.p_val_thresh:
                        remove_score.append(original_MB_score[index])
                        remove.append(col)
                    else:
                        original_MB_score[index] = p_val[1]
                return [i for i in original_MB if i not in remove], [j for j in original_MB_score if j not in remove_score]
        else:
            return [i for i in original_MB], original_MB_score
    def _find_MB(self, data, Y):
        orig_data = data.copy()
        Y = np.reshape(Y, (-1, 1))
        MB = [1]
        MB_score = [1]
        candidate_variable = self._grow(data, Y, MB)
        MB, MB_score = self._shrink(orig_data, Y, candidate_variable, MB, MB_score)
        if len(MB) < 1:
            return list(), list()
        else:
            last_MB = MB
            last_score = MB_score
            candidate_variable = self._grow(data, Y, MB)
            MB, MB_score = self._shrink(orig_data, Y, candidate_variable, MB, MB_score)
            while last_MB != MB:
                last_MB = MB
                last_score = MB_score
                candidate_variable = self._grow(data, Y, MB)
                MB, MB_score = self._shrink(orig_data, Y, candidate_variable, MB, MB_score)
        return last_MB, last_score

    def _ranking(self, data, Y, MB):
        feature_ranking = list()
        for col in range(1, len(data.iloc[0, :]) + 1):
            feat_score = list()
            temp_MB = MB.copy()
            if col in MB:
                for i in range(len(MB)):
                    if col == MB[i]:
                        index = i
                        break
                feat_score.append(col)
                feat_score.append(self.feat_imp_scores[index])
            else:
                tt = self._PPI(data[col].values, Y, data[temp_MB].values, col)
                feat_score.append(tt[0])
                feat_score.append(tt[1])
            feature_ranking.append(feat_score)
        return feature_ranking

    def fit(self, data, Y):
        if self.cv!= 0:
            parallel = Parallel(n_jobs=self.n_jobs)#, verbose=self.verbose)
            if type(self.cv).__name__ == "StratifiedKFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
            elif type(self.cv).__name__ == "KFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
                else:                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
            else:
                kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv)) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))

            for i in range(len(tmp)):
                if i == 0:
                    feature_sets, scores = [tmp[i][0]], [tmp[i][1]]
                else:
                    feature_sets.append(tmp[i][0])
                    scores.append(tmp[i][1])

            final_feats = dict()
            for fs in feature_sets:
                for i in fs:
                    if i not in final_feats:
                        final_feats[i] = 1
                    else:
                        final_feats[i]+= 1
            final_MB, max_score, final_feat_imp = list(), 0, list()
            for fs, feat_imp in zip(feature_sets, scores):
                tmp = [final_feats[i] for i in fs]
                score = sum(tmp)/max(len(tmp), 1)
                if score > max_score:
                    final_MB = fs
                    final_feat_imp = feat_imp
                    max_score = score

            tmp_feats_and_imp = list(zip(final_MB, final_feat_imp))
            tmp_feats_and_imp.sort(key = lambda x: x[1], reverse=True)
            
            self.MB = [i for i, _ in tmp_feats_and_imp]
            self.feat_imp_scores = [i for _, i in tmp_feats_and_imp]

        else:
            final_MB, final_feat_imp = self._find_MB(data.copy(), Y)
            tmp_feats_and_imp = list(zip(final_MB, final_feat_imp))
            tmp_feats_and_imp.sort(key = lambda x: x[1], reverse=False)
            self.MB = [i for i, _ in tmp_feats_and_imp]
            self.feat_imp_scores = [i for _, i in tmp_feats_and_imp]
            feature_ranking = self._ranking(data.copy(), Y, self.MB)
            feature_ranking.sort(key = lambda x: x[1], reverse=False)
            self.feature_ranking = feature_ranking

            for i in range(len(feature_ranking)):
                self.col_ranking.append(feature_ranking[i][0])
                self.score_ranking.append(feature_ranking[i][1])

    def transform(self, data):
        return data[self.MB]
    def Fit(self, data, Y):
        self.fit(data, Y)
    def feature_importance(self):
        y_axis = np.arange(len(self.MB))
        x_axis = self.feat_imp_scores

        sns.barplot(x=x_axis, y=y_axis, orient="h")
        plt.yticks(y_axis, [str(i) for i in self.MB], size='small')
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.show()