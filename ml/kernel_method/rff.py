import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from scipy.stats import ortho_group


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        i = np.random.choice(X.shape[0], 1000000)
        j = (np.random.choice(X.shape[0] - 1, 1000000) + i + 1) % X.shape[0]

        sigma = np.median(np.sum((X[i] - X[j]) ** 2, axis=1))
        self.w = np.random.normal(0, 1 / np.sqrt(sigma), (self.new_dim, self.n_features))
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        return self.func(X @ self.w + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        i = np.random.choice(X.shape[0], 1000000)
        j = (np.random.choice(X.shape[0] - 1, 1000000) + i + 1) % X.shape[0]

        sigma = np.median(np.sum((X[i] - X[j]) ** 2, axis=1))
        if self.new_dim <= self.n_features:
            self.w = np.concatenate(
                [np.diag(np.sqrt(np.random.chisquare(self.new_dim, self.new_dim))) @ ortho_group.rvs(self.new_dim) for i in
                 range(self.n_features // self.new_dim + 1)],
                axis=1)[:, :self.n_features]
        else:
            self.w = (np.sqrt(np.diag(np.random.chisquare(self.new_dim, self.new_dim))) @ ortho_group.rvs(self.new_dim))[:,
                     :self.n_features]
        self.w = self.w/np.sqrt(sigma)
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

class RandomMaclaurinFeatureCreator(FeatureCreatorPlaceholder):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        super().__init__(n_features, new_dim, func)
        self.sigma = None
        self.p = 10

    def fit(self, X, y=None):
        # p = 10

        i = np.random.choice(X.shape[0], 1000000)
        j = (np.random.choice(X.shape[0] - 1, 1000000) + i + 1) % X.shape[0]

        self.sigma = np.median(np.sum((X[i] - X[j]) ** 2, axis=1))

        self.w = [np.random.choice([-1, 1], (self.new_dim, np.random.geometric(1 - 1/self.p) - 1))
                  for i in range(self.n_features)]
        return self

    def transform(self, X, y=None):
        return np.transpose(np.stack([
            np.sqrt(self.p**(w.shape[1])/(self.sigma**(w.shape[1]) * np.math.factorial(w.shape[1])))*np.prod(X@w, axis=1)
            for w in self.w
        ], axis=0))


class RFFPipeline(BaseEstimator):
    """
    ÐŸÐ°Ð¹Ð¿Ð»Ð°Ð¹Ð½, Ð´ÐµÐ»Ð°ÑŽÑ‰Ð¸Ð¹ Ð¿Ð¾ÑÐ»ÐµÐ´Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð¾ Ñ‚Ñ€Ð¸ ÑˆÐ°Ð³Ð°:
        1. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ PCA
        2. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ RFF
        3. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ ÐºÐ»Ð°ÑÑÐ¸Ñ„Ð¸ÐºÐ°Ñ‚Ð¾Ñ€Ð°
    """

    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: ÐšÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¾Ð², Ð³ÐµÐ½ÐµÑ€Ð¸Ñ€ÑƒÐµÐ¼Ñ‹Ñ… RFF
        :param new_dim: ÐšÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¾Ð², Ð´Ð¾ ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ñ… ÑÐ¶Ð¸Ð¼Ð°ÐµÑ‚ PCA
        :param use_PCA: Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ Ð»Ð¸ PCA
        :param feature_creator_class: ÐšÐ»Ð°ÑÑ, ÑÐ¾Ð·Ð´Ð°ÑŽÑ‰Ð¸Ð¹ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¸, Ð¿Ð¾ ÑƒÐ¼Ð¾Ð»Ñ‡Ð°Ð½Ð¸ÑŽ Ð·Ð°Ð³Ð»ÑƒÑˆÐºÐ°
        :param classifier_class: ÐšÐ»Ð°ÑÑ ÐºÐ»Ð°ÑÑÐ¸Ñ„Ð¸ÐºÐ°Ñ‚Ð¾Ñ€Ð°
        :param classifier_params: ÐŸÐ°Ñ€Ð°Ð¼ÐµÑ‚Ñ€Ñ‹, ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¼Ð¸ Ð¸Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð¸Ñ€ÑƒÐµÑ‚ÑÑ ÐºÐ»Ð°ÑÑÐ¸Ñ„Ð¸ÐºÐ°Ñ‚Ð¾Ñ€
        :param func: Ð¤ÑƒÐ½ÐºÑ†Ð¸Ñ, ÐºÐ¾Ñ‚Ð¾Ñ€ÑƒÑŽ Ð¿Ð¾Ð»ÑƒÑ‡Ð°ÐµÑ‚ feature_creator Ð¿Ñ€Ð¸ Ð¸Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸.
                     Ð•ÑÐ»Ð¸ Ð½Ðµ Ñ…Ð¾Ñ‚Ð¸Ñ‚Ðµ, Ð¼Ð¾Ð¶ÐµÑ‚Ðµ Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ ÑÑ‚Ð¾Ñ‚ Ð¿Ð°Ñ€Ð°Ð¼ÐµÑ‚Ñ€.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        if not self.use_PCA:
            self.new_dim = X.shape[1]
            self.feature_creator.new_dim = self.new_dim
            pipeline_steps: list[tuple] = [('rff', self.feature_creator),
                                           ('classifier', self.classifier)]
        else:
            pipeline_steps: list[tuple] = [('pca', PCA(n_components=self.new_dim)),
                                           ('rff', self.feature_creator),
                                           ('classifier', self.classifier)]
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

class RFFPipelineRegressor(BaseEstimator):
    """
    ÐŸÐ°Ð¹Ð¿Ð»Ð°Ð¹Ð½, Ð´ÐµÐ»Ð°ÑŽÑ‰Ð¸Ð¹ Ð¿Ð¾ÑÐ»ÐµÐ´Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð¾ Ñ‚Ñ€Ð¸ ÑˆÐ°Ð³Ð°:
        1. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ PCA
        2. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ RFF
        3. ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ Ñ€ÐµÐ³Ñ€ÐµÑÑÐ¸Ð¸
    """

    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            regression_class=Ridge,
            regression_params=None,
            func=np.cos,
    ):
        """
        :param n_features: ÐšÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¾Ð², Ð³ÐµÐ½ÐµÑ€Ð¸Ñ€ÑƒÐµÐ¼Ñ‹Ñ… RFF
        :param new_dim: ÐšÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¾Ð², Ð´Ð¾ ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ñ… ÑÐ¶Ð¸Ð¼Ð°ÐµÑ‚ PCA
        :param use_PCA: Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ Ð»Ð¸ PCA
        :param feature_creator_class: ÐšÐ»Ð°ÑÑ, ÑÐ¾Ð·Ð´Ð°ÑŽÑ‰Ð¸Ð¹ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¸, Ð¿Ð¾ ÑƒÐ¼Ð¾Ð»Ñ‡Ð°Ð½Ð¸ÑŽ Ð·Ð°Ð³Ð»ÑƒÑˆÐºÐ°
        :param regression_class: ÐšÐ»Ð°ÑÑ Ñ€ÐµÐ³Ñ€ÐµÑÑÐ¸Ð¸
        :param regression_params: ÐŸÐ°Ñ€Ð°Ð¼ÐµÑ‚Ñ€Ñ‹, ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¼Ð¸ Ð¸Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð¸Ñ€ÑƒÐµÑ‚ÑÑ Ñ€ÐµÐ³Ñ€ÐµÑÑÐ¸Ñ
        :param func: Ð¤ÑƒÐ½ÐºÑ†Ð¸Ñ, ÐºÐ¾Ñ‚Ð¾Ñ€ÑƒÑŽ Ð¿Ð¾Ð»ÑƒÑ‡Ð°ÐµÑ‚ feature_creator Ð¿Ñ€Ð¸ Ð¸Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸.
                     Ð•ÑÐ»Ð¸ Ð½Ðµ Ñ…Ð¾Ñ‚Ð¸Ñ‚Ðµ, Ð¼Ð¾Ð¶ÐµÑ‚Ðµ Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ ÑÑ‚Ð¾Ñ‚ Ð¿Ð°Ñ€Ð°Ð¼ÐµÑ‚Ñ€.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if regression_params is None:
            regression_params = {}
        self.regression = regression_class(**regression_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        if not self.use_PCA:
            self.new_dim = X.shape[1]
            self.feature_creator.new_dim = self.new_dim
            pipeline_steps: list[tuple] = [('rff', self.feature_creator),
                                           ('regression', self.regression)]
        else:
            pipeline_steps: list[tuple] = [('pca', PCA(n_components=self.new_dim)),
                                           ('rff', self.feature_creator),
                                           ('regression', self.regression)]
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)