from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
import optuna

from typing import Optional

import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: Optional[dict] = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            early_stopping_rounds = 0,
            eval_set = None,
            subsample = 1.0,
            bagging_temperature = 1.0,
            bootstrap_type = 'Bernoulli',
            rsm = 1.0,
            quantization_type = None,
            nbins = 255,
            trial = None
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        if eval_set is None and early_stopping_rounds != 0:
            raise NotImplementedError
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.quant = None
        self.feature_importances_ = None
        self.trial = trial

    def partial_fit(self, X, y):
        b = self.base_model_class(**self.base_model_params)
        if self.rsm != 1:
            mask = np.full(X.shape[1], True)
            mask[:np.ceil(X.shape[1] * self.rsm).astype('int')] = False
            np.random.shuffle(mask)
            X = X.copy()
            X[:, mask] = 0

        if self.bootstrap_type == 'Bernoulli':
            mask = np.full(y.shape[0], False)
            mask[:np.ceil(y.shape[0] * self.subsample).astype('int')] = True
            np.random.shuffle(mask)
            X = X[mask]
            y = y[mask]
            b.fit(X, y)
        if self.bootstrap_type == 'Bayesian':
            weight = np.power(-np.log(np.random.uniform(size = y.shape[0])), self.bagging_temperature)
            b.fit(X, y, sample_weight=weight)
        return b

    def fit(self, X_train, y_train, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        original_X_train = X_train
        self.feature_importances_ = np.zeros(X_train.shape[1])

        if self.quantization_type == 'Uniform':
            self.quant = KBinsDiscretizer(
                n_bins=self.nbins, encode='ordinal', strategy='uniform'
            )
            X_train = self.quant.fit_transform(X_train.toarray())
        elif self.quantization_type == 'Quantile':
            self.quant = KBinsDiscretizer(
                n_bins=self.nbins, encode='ordinal', strategy='quantile'
            )
            X_train = self.quant.fit_transform(X_train.toarray())

        train_predictions = np.zeros(y_train.shape[0])

        if self.early_stopping_rounds != 0 and not(self.eval_set is None):
            X_val, y_val = self.eval_set
            cur_bad_rounds = 0

        for _ in range(self.n_estimators):
            self.models.append(self.partial_fit(X_train, -self.loss_derivative(y_train, train_predictions)))
            new_predictions = self.models[-1].predict(X_train)
            self.gammas.append(self.find_optimal_gamma(y_train, train_predictions, new_predictions))
            train_predictions += self.learning_rate * self.gammas[-1] * new_predictions
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['train_roc_auc'].append(self.score(original_X_train, y_train))
            self.feature_importances_ += self.models[-1].feature_importances_
            if not(self.trial is None):
                self.trial.report(self.history['train_loss'][-1], _)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            if self.early_stopping_rounds != 0:
                self.history['val_loss'].append(self.loss_fn(y_val, self.predict_proba(X_val)[:, 1]))
                self.history['val_roc_auc'].append(self.score(X_val, y_val))
                if _ > 0:
                    if self.history['val_loss'][-2] < self.history['val_loss'][-1]:
                        cur_bad_rounds += 1
                    else:
                        cur_bad_rounds = 0
                    if cur_bad_rounds == self.early_stopping_rounds:
                        break
        self.feature_importances_ = self.feature_importances_/len(self.models)
        self.feature_importances_ = self.feature_importances_ / np.linalg.norm(self.feature_importances_)

        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        if not (self.quant is None):
            X = self.quant.transform(X.toarray())
        predict = np.zeros(X.shape[0])
        for _ in range(len(self.models)):
            predict += self.models[_].predict(X) * self.gammas[_] * self.learning_rate
        p1 = self.sigmoid(predict)
        return np.vstack([np.ones(X.shape[0]) - p1, p1]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self, X, y, label='Passed dataset'):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        fig, axes = plt.subplots(2)
        steps = np.arange(len(self.models))
        predict = np.zeros(X.shape[0])
        loss = []
        auc_roc = []
        for _ in range(len(self.models)):
            predict += self.models[_].predict(X) * self.gammas[_] * self.learning_rate
            loss.append(self.loss_fn(y, predict))
            auc_roc.append(roc_auc_score(y == 1, self.sigmoid(predict)))

        axes[0].plot(steps, loss, label=label)
        axes[0].plot(steps, self.history['train_loss'], label='train')
        axes[0].set_title('Loss')

        axes[1].plot(steps, auc_roc, label=label)
        axes[1].plot(steps, self.history['train_roc_auc'], label='train')
        axes[1].set_title('AUC-ROC')

        axes[0].legend()
        plt.tight_layout()
        fig.show()
