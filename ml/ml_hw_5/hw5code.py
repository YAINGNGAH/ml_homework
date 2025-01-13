import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def find_best_split(feature_vector, target_vector):
    """
    ÐŸÐ¾Ð´ ÐºÑ€Ð¸Ñ‚ÐµÑ€Ð¸ÐµÐ¼ Ð”Ð¶Ð¸Ð½Ð¸ Ð·Ð´ÐµÑÑŒ Ð¿Ð¾Ð´Ñ€Ð°Ð·ÑƒÐ¼ÐµÐ²Ð°ÐµÑ‚ÑÑ ÑÐ»ÐµÐ´ÑƒÑŽÑ‰Ð°Ñ Ñ„ÑƒÐ½ÐºÑ†Ð¸Ñ:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ â€” Ð¼Ð½Ð¾Ð¶ÐµÑÑ‚Ð²Ð¾ Ð¾Ð±ÑŠÐµÐºÑ‚Ð¾Ð², $R_l$ Ð¸ $R_r$ â€” Ð¾Ð±ÑŠÐµÐºÑ‚Ñ‹, Ð¿Ð¾Ð¿Ð°Ð²ÑˆÐ¸Ðµ Ð² Ð»ÐµÐ²Ð¾Ðµ Ð¸ Ð¿Ñ€Ð°Ð²Ð¾Ðµ Ð¿Ð¾Ð´Ð´ÐµÑ€ÐµÐ²Ð¾,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ â€” Ð´Ð¾Ð»Ñ Ð¾Ð±ÑŠÐµÐºÑ‚Ð¾Ð² ÐºÐ»Ð°ÑÑÐ° 1 Ð¸ 0 ÑÐ¾Ð¾Ñ‚Ð²ÐµÑ‚ÑÑ‚Ð²ÐµÐ½Ð½Ð¾.

    Ð£ÐºÐ°Ð·Ð°Ð½Ð¸Ñ:
    * ÐŸÐ¾Ñ€Ð¾Ð³Ð¸, Ð¿Ñ€Ð¸Ð²Ð¾Ð´ÑÑ‰Ð¸Ðµ Ðº Ð¿Ð¾Ð¿Ð°Ð´Ð°Ð½Ð¸ÑŽ Ð² Ð¾Ð´Ð½Ð¾ Ð¸Ð· Ð¿Ð¾Ð´Ð´ÐµÑ€ÐµÐ²ÑŒÐµÐ² Ð¿ÑƒÑÑ‚Ð¾Ð³Ð¾ Ð¼Ð½Ð¾Ð¶ÐµÑÑ‚Ð²Ð° Ð¾Ð±ÑŠÐµÐºÑ‚Ð¾Ð², Ð½Ðµ Ñ€Ð°ÑÑÐ¼Ð°Ñ‚Ñ€Ð¸Ð²Ð°ÑŽÑ‚ÑÑ.
    * Ð’ ÐºÐ°Ñ‡ÐµÑÑ‚Ð²Ðµ Ð¿Ð¾Ñ€Ð¾Ð³Ð¾Ð², Ð½ÑƒÐ¶Ð½Ð¾ Ð±Ñ€Ð°Ñ‚ÑŒ ÑÑ€ÐµÐ´Ð½ÐµÐµ Ð´Ð²ÑƒÑ… ÑÐ¾ÑÐ´ÐµÐ½Ð¸Ñ… (Ð¿Ñ€Ð¸ ÑÐ¾Ñ€Ñ‚Ð¸Ñ€Ð¾Ð²ÐºÐµ) Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ð¹ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ°
    * ÐŸÐ¾Ð²ÐµÐ´ÐµÐ½Ð¸Ðµ Ñ„ÑƒÐ½ÐºÑ†Ð¸Ð¸ Ð² ÑÐ»ÑƒÑ‡Ð°Ðµ ÐºÐ¾Ð½ÑÑ‚Ð°Ð½Ñ‚Ð½Ð¾Ð³Ð¾ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ° Ð¼Ð¾Ð¶ÐµÑ‚ Ð±Ñ‹Ñ‚ÑŒ Ð»ÑŽÐ±Ñ‹Ð¼.
    * ÐŸÑ€Ð¸ Ð¾Ð´Ð¸Ð½Ð°ÐºÐ¾Ð²Ñ‹Ñ… Ð¿Ñ€Ð¸Ñ€Ð¾ÑÑ‚Ð°Ñ… Ð”Ð¶Ð¸Ð½Ð¸ Ð½ÑƒÐ¶Ð½Ð¾ Ð²Ñ‹Ð±Ð¸Ñ€Ð°Ñ‚ÑŒ Ð¼Ð¸Ð½Ð¸Ð¼Ð°Ð»ÑŒÐ½Ñ‹Ð¹ ÑÐ¿Ð»Ð¸Ñ‚.
    * Ð—Ð° Ð½Ð°Ð»Ð¸Ñ‡Ð¸Ðµ Ð² Ñ„ÑƒÐ½ÐºÑ†Ð¸Ð¸ Ñ†Ð¸ÐºÐ»Ð¾Ð² Ð±Ð°Ð»Ð» Ð±ÑƒÐ´ÐµÑ‚ ÑÐ½Ð¸Ð¶ÐµÐ½. Ð’ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð·ÑƒÐ¹Ñ‚Ðµ! :)

    :param feature_vector: Ð²ÐµÑ‰ÐµÑÑ‚Ð²ÐµÐ½Ð½Ð¾Ð·Ð½Ð°Ñ‡Ð½Ñ‹Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ð¹ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ°
    :param target_vector: Ð²ÐµÐºÑ‚Ð¾Ñ€ ÐºÐ»Ð°ÑÑÐ¾Ð² Ð¾Ð±ÑŠÐµÐºÑ‚Ð¾Ð²,  len(feature_vector) == len(target_vector)

    :return thresholds: Ð¾Ñ‚ÑÐ¾Ñ€Ñ‚Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ñ‹Ð¹ Ð¿Ð¾ Ð²Ð¾Ð·Ñ€Ð°ÑÑ‚Ð°Ð½Ð¸ÑŽ Ð²ÐµÐºÑ‚Ð¾Ñ€ ÑÐ¾ Ð²ÑÐµÐ¼Ð¸ Ð²Ð¾Ð·Ð¼Ð¾Ð¶Ð½Ñ‹Ð¼Ð¸ Ð¿Ð¾Ñ€Ð¾Ð³Ð°Ð¼Ð¸, Ð¿Ð¾ ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¼ Ð¾Ð±ÑŠÐµÐºÑ‚Ñ‹ Ð¼Ð¾Ð¶Ð½Ð¾
     Ñ€Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÑŒ Ð½Ð° Ð´Ð²Ðµ Ñ€Ð°Ð·Ð»Ð¸Ñ‡Ð½Ñ‹Ðµ Ð¿Ð¾Ð´Ð²Ñ‹Ð±Ð¾Ñ€ÐºÐ¸, Ð¸Ð»Ð¸ Ð¿Ð¾Ð´Ð´ÐµÑ€ÐµÐ²Ð°
    :return ginis: Ð²ÐµÐºÑ‚Ð¾Ñ€ ÑÐ¾ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸ÑÐ¼Ð¸ ÐºÑ€Ð¸Ñ‚ÐµÑ€Ð¸Ñ Ð”Ð¶Ð¸Ð½Ð¸ Ð´Ð»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð³Ð¾ Ð¸Ð· Ð¿Ð¾Ñ€Ð¾Ð³Ð¾Ð² Ð² thresholds len(ginis) == len(thresholds)
    :return threshold_best: Ð¾Ð¿Ñ‚Ð¸Ð¼Ð°Ð»ÑŒÐ½Ñ‹Ð¹ Ð¿Ð¾Ñ€Ð¾Ð³ (Ñ‡Ð¸ÑÐ»Ð¾)
    :return gini_best: Ð¾Ð¿Ñ‚Ð¸Ð¼Ð°Ð»ÑŒÐ½Ð¾Ðµ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ðµ ÐºÑ€Ð¸Ñ‚ÐµÑ€Ð¸Ñ Ð”Ð¶Ð¸Ð½Ð¸ (Ñ‡Ð¸ÑÐ»Ð¾)
    """
    if np.all(feature_vector == feature_vector[0]):
        return [], [], None, None
    args = np.argsort(feature_vector)
    fv = feature_vector[args]
    tv = target_vector[args]
    _, args, add = np.unique(fv[::-1], return_index=True, return_counts=True)
    thresholds = (_[:-1] + _[1:]) / 2
    pl = tv.cumsum() / np.arange(1, feature_vector.shape[0] + 1)
    pr = (np.sum(target_vector) - tv.cumsum()) / (np.arange(feature_vector.shape[0] - 1, -1, -1))
    hl = 1 - np.square(pl) - np.square(1 - pl)
    hr = 1 - np.square(pr) - np.square(1 - pr)
    ginis = ((-hl * np.arange(1, feature_vector.shape[0] + 1) - hr * np.arange(feature_vector.shape[0] - 1, -1, -1)) / (feature_vector.shape[0]))[feature_vector.shape[0] - 1 - args[:-1]]
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = np.max(ginis)
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        if not (self._min_samples_split is None) and sub_y.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # [0][0]
            return
        if not (self._max_depth is None) and node['depth'] == self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # [0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):  # range(1, sub_X.shape[1])->range(0, sub_X.shape[1])
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[
                        key] = current_click / current_count  # current_count / current_click -> current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))  # x[1] -> x[0]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is None:
                continue

            if not (self._min_samples_leaf is None) and (
                    np.sum(feature_vector < threshold) < self._min_samples_leaf or np.sum(
                    feature_vector >= threshold) < self._min_samples_leaf):
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # [0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": node['depth'] + 1}, {"depth": node['depth'] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)],
                       node["right_child"])  # split -> np.logical_not(split)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree['depth'] = 1
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


def find_best_split_regression(feature_vector, X, target_vector, model):
    if np.all(feature_vector == feature_vector[0]):
        return [], [], None, None
    best_mse = None
    best_thr = None
    md = model()
    for thr in np.percentile(feature_vector, np.arange(1, 100)):
        fv = X[feature_vector < thr]
        tv = target_vector[feature_vector < thr]
        if fv.shape[0] == 0:
            continue
        md.fit(fv, tv)
        l_mse = mean_squared_error(tv, md.predict(fv)) * fv.shape[0]
        fv = X[feature_vector >= thr]
        tv = target_vector[feature_vector >= thr]
        if fv.shape[0] == 0:
            continue
        md.fit(fv, tv)
        r_mse = mean_squared_error(tv, md.predict(fv)) * fv.shape[0]
        mse = (l_mse + r_mse) / feature_vector.shape[0]
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_thr = thr
    return best_thr, best_mse


class LinearRegressionTree:
    '''
    ÐžÐ±Ñ€Ð°Ð±Ð°Ñ‚Ñ‹Ð²Ð°ÐµÑ‚ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ñ‡Ð¸ÑÐ»Ð¾Ð²Ñ‹Ðµ Ð¿Ñ€Ð¸Ð·Ð½Ð°ÐºÐ¸, Ð´Ð»Ñ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð°Ð»ÑŒÐ½Ñ‹Ñ… Ð½ÐµÐ¾Ð±Ñ…Ð¾Ð´Ð¸Ð¼ Ð¿Ñ€ÐµÐ´Ð²Ð°Ñ€Ð¸Ñ‚ÐµÐ»ÑŒÐ½Ð¾Ðµ ÐºÐ¾Ð´Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ.
    '''
    def __init__(self, base_model_type=LinearRegression, max_depth=None, min_samples_split=None,
                 min_samples_leaf=None):

        self._tree = {}
        self._base_model_type = base_model_type
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(sub_X, sub_y)
            return
        if not (self._min_samples_split is None) and sub_y.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(sub_X, sub_y)
            return
        if not (self._max_depth is None) and node['depth'] == self._max_depth:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(sub_X, sub_y)
            return

        feature_best, threshold_best, mse_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_vector = sub_X[:, feature]
            if np.all(feature_vector == feature_vector[0]):
                continue

            threshold, mse = find_best_split_regression(feature_vector, sub_X, sub_y, self._base_model_type)
            if mse is None:
                continue

            if not (self._min_samples_leaf is None) and (
                    np.sum(feature_vector < threshold) < self._min_samples_leaf or np.sum(
                    feature_vector >= threshold) < self._min_samples_leaf):
                continue

            if mse_best is None or mse > mse_best:
                feature_best = feature
                mse_best = mse
                split = feature_vector < threshold
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {"depth": node['depth'] + 1}, {"depth": node['depth'] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)],
                       node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))
        if x[node["feature_split"]] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree['depth'] = 1
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
