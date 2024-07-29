import numpy as np
import pandas as pd
from collections import Counter
import copy
import numbers
from warnings import warn, catch_warnings, simplefilter

class Node:
    def __init__(self, feature=None, feature_name=None, threshold=None, left=None, right=None,
                 gain=None, id=None, depth=None, leaf_node=False, samples=None, gini=None,
                 value=None, clf_value_dis=None, clf_prob_dis=None, sample_indices=None):
        self.feature = feature
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.gini = gini
        self.value = value
        self.clf_value_dis = clf_value_dis
        self.clf_prob_dis = clf_prob_dis
        self.id = id
        self.depth = depth
        self.samples = samples
        self.leaf_node = leaf_node
        self.sample_indices = sample_indices

    def is_leaf_node(self):
        return self.leaf_node is not False

class DecisionTree:
    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=None, n_features=None, criterion="gini",
                 treetype="classification", k=None, feature_names=None, HShrinkage=False, HS_lambda=0, random_state=None):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_features = n_features
        self.feature_names = feature_names
        self.root = None
        self.criterion = criterion
        self.k = k
        self.treetype = treetype
        self.HShrinkage = HShrinkage
        self.HS_lambda = HS_lambda
        self.random_state = random_state
        self.random_state_ = self._check_random_state(random_state)
        self.n_nodes = 0
        self.oob_preds = None
        self.oob_shap = None
        self.HS_applied = False

    def _check_random_state(self, seed):
        if isinstance(seed, numbers.Integral) or (seed is None):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def fit(self, X, y):
        if not self.n_features:
            self.n_features = X.shape[1]
        elif self.n_features == "sqrt":
            self.n_features = int(np.rint(np.sqrt(X.shape[1])))
        elif isinstance(self.n_features, float):
            self.n_features = max(1, int(self.n_features * X.shape[1]))
        else:
            self.n_features = min(X.shape[1], self.n_features)

        self.features_in_ = range(X.shape[1])
        self.node_list = []
        self.node_id_dict = {}
        self.no_samples_total = y.shape[0]
        self.y = y  # Store y in the class

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.root = self._grow_tree(X, y, np.arange(y.shape[0]), feature_names=self.feature_names)
        self._get_decision_paths()
        self.node_list = sorted(self.node_list, key=lambda o: o.id)

        if self.HShrinkage:
            self._apply_hierarchical_srinkage(treetype=self.treetype)
        self._create_node_dict()
        depth_list = [len(i) for i in self.decision_paths]
        self.max_depth_ = max(depth_list) - 1
        self._get_feature_importance()


    def _create_node_dict(self):
        for node in self.node_list:
            self.node_id_dict[node.id] = {
                "node": node,
                "id": node.id,
                "depth": node.depth,
                "feature": node.feature_name or node.feature,
                "is_leaf_node": node.leaf_node,
                "threshold": node.threshold,
                "gini": node.gini,
                "samples": node.samples,
                "value": node.value,
                "sample_indices": node.sample_indices
            }
            if self.treetype == "classification":
                self.node_id_dict[node.id]["value_distribution"] = node.clf_value_dis
                self.node_id_dict[node.id]["prob_distribution"] = list(node.clf_prob_dis)

    def _grow_tree(self, X, y, sample_indices, depth=0, feature_names=None):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if self.treetype == "classification":
            counter = Counter(y)
            clf_value_dis = [counter.get(0) or 0, counter.get(1) or 0]
            clf_prob_dis = (np.array(clf_value_dis) / n_samples)
            leaf_value = np.argmax(clf_prob_dis)

        elif self.treetype == "regression":
            leaf_value = self._mean_label(y)
            clf_value_dis = None
            clf_prob_dis = None

        if ((self.max_depth is not None) and ((depth >= self.max_depth))
                or (n_labels == 1) or (n_samples < self.min_samples_split)
                or ((self.k != None) and (n_samples <= self.k))):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis, y, depth, n_samples, sample_indices)
            return node

        feat_idxs = self.random_state_.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        if (best_gain == -1) or (best_feature is None) or (best_thresh is None):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis, y, depth, n_samples, sample_indices)
            return node

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if (len(left_idxs) < self.min_samples_leaf) or (len(right_idxs) < self.min_samples_leaf) or (
                (self.k != None) and ((len(left_idxs) <= self.k) or (len(right_idxs) <= self.k))):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis, y, depth, n_samples, sample_indices)
            return node

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], sample_indices[left_idxs], depth + 1, feature_names)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], sample_indices[right_idxs], depth + 1, feature_names)

        best_feature_name = None
        if feature_names is not None:
            best_feature_name = feature_names[best_feature]

        node = Node(best_feature,
                    best_feature_name,
                    best_thresh,
                    left,
                    right,
                    best_gain,
                    gini=self._gini(y),
                    depth=depth,
                    value=leaf_value,
                    clf_value_dis=clf_value_dis,
                    clf_prob_dis=clf_prob_dis,
                    samples=n_samples,
                    sample_indices=sample_indices)
        self.node_list.append(node)
        return node

    def _create_leaf(self, leaf_value, clf_value_dis, clf_prob_dis, y, depth, n_samples, sample_indices):
        node = Node(value=leaf_value,
                    clf_value_dis=clf_value_dis,
                    clf_prob_dis=clf_prob_dis,
                    leaf_node=True,
                    gini=self._gini(y),
                    depth=depth,
                    samples=n_samples,
                    sample_indices=sample_indices)
        self.node_list.append(node)
        return node

    def _best_split(self, X, y, feat_idxs):
        best_gain = np.array([-1])
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            if len(thresholds) == 1:
                gain = self._information_gain(y, X_column, thresholds[0])
                if gain > best_gain.max():
                    best_gain = np.array([gain])
                    split_idx = np.array([feat_idx])
                    split_threshold = thresholds

            for index in range(1, len(thresholds)):
                thr = (thresholds[index] + thresholds[index - 1]) / 2
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain.max():
                    best_gain = np.array([gain])
                    split_idx = np.array([feat_idx])
                    split_threshold = np.array([thr])
                elif gain == best_gain.all():
                    best_gain = np.append(best_gain, gain)
                    split_idx = np.append(split_idx, feat_idx)
                    split_threshold = np.append(split_threshold, thr)

        idx_best = self.random_state_.choice(best_gain.shape[0], 1)[0]
        return split_idx[idx_best], split_threshold[idx_best], best_gain[idx_best]

    def _information_gain(self, y, X_column, threshold):
        criterion = self.criterion

        if criterion == "entropy":
            parent_entropy = self._entropy(y)
        elif criterion == "gini":
            parent_gini = self._gini(y)
        elif criterion == "mse":
            parent_mse = np.mean((y - np.mean(y)) ** 2)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        if criterion == "entropy":
            e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
            information_gain = parent_entropy - child_entropy

        elif criterion == "gini":
            g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
            child_gini = (n_l / n) * g_l + (n_r / n) * g_r
            information_gain = (n / self.no_samples_total) * (parent_gini - child_gini)

        elif criterion == "mse":
            mse_l, mse_r = np.mean((y[left_idxs] - np.mean(y[left_idxs])) ** 2), np.mean((y[right_idxs] - np.mean(y[right_idxs])) ** 2)
            child_mse = (n_l / n) * mse_l + (n_r / n) * mse_r
            information_gain = parent_mse - child_mse

        return information_gain



    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _gini(self, y):
        n = len(y)
        k = self.k

        if self.treetype == "classification":
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / counts.sum()
            impurity = 1 - sum(probabilities ** 2)

        elif self.treetype == "regression":
            if len(y) == 0:
                impurity = 0
            else:
                impurity = np.mean((y - np.mean(y)) ** 2)

        if (k != None) and (n > k):
            impurity = impurity * n / (n - k)
        elif (k != None) and (n <= k):
            impurity = 1
        return impurity

    def _mean_label(self, y):
        return np.mean(y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self._traverse_tree(X, self.root))
        else:
            return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        if self.treetype != "classification":
            message = "This function is only available for classification tasks"
            warn(message)
            return

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self._traverse_tree(X, self.root, pred_proba=True))
        else:
            return np.array([self._traverse_tree(x, self.root, pred_proba=True) for x in X])

    def _traverse_tree(self, x, node, pred_proba=False):
        if node.is_leaf_node():
            if pred_proba:
                return node.clf_prob_dis
            else:
                return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left, pred_proba)
        return self._traverse_tree(x, node.right, pred_proba)

    def _get_feature_importance(self):
        feature_importance = np.zeros(len(self.features_in_))
        features_list = [i.feature for i in self.node_list]
        feat_imp_p_node = np.nan_to_num(np.array([i.gain for i in self.node_list], dtype=float))

        for feat_num, feat_imp in zip(features_list, feat_imp_p_node):
            if feat_num is not None:
                feature_importance[feat_num] += feat_imp

        if np.sum(feature_importance) != 0:
            feature_importance_scaled = np.divide(feature_importance, (np.sum(feature_importance)))
        else:
            feature_importance_scaled = feature_importance

        self.feature_importances_ = feature_importance_scaled

    def _get_decision_paths(self):
        self.decision_paths = list(self._paths(self.root))
        self.decision_paths_str = ["->".join(map(str, path)) for path in self.decision_paths]

    def _paths(self, node, p=()):
        if node.left or node.right:
            if node.id is None:
                node.id = self.n_nodes
                self.n_nodes += 1
            yield from self._paths(node.left, (*p, node.id))
            yield from self._paths(node.right, (*p, node.id))
        else:
            if node.id is None:
                node.id = self.n_nodes
                self.n_nodes += 1
            yield (*p, node.id)


    def prune(self, min_samples_leaf=None):
        """
        Prunes the decision tree by converting nodes with sample counts
        less than or equal to min_samples_leaf into leaf nodes.

        Parameters:
        - min_samples_leaf: The minimum number of samples required at a node
                            for it to remain a decision node. Nodes with
                            fewer samples will be pruned.
        """
        if min_samples_leaf is None:
            min_samples_leaf = self.min_samples_leaf

        pruned_nodes = 0

        # Iterate through all decision paths
        for decision_path in self.decision_paths:
            for l, node_id in enumerate(decision_path):
                node = self.node_list[node_id]
                # Check if the number of samples at the node is less than or equal to the threshold
                if node.samples <= min_samples_leaf:
                    pruned_nodes += 1
                    # Convert this node to a leaf
                    self._make_leaf(node, self._get_y_for_node(node))

        # Print number of pruned nodes for debugging
        # print(f"Pruned {pruned_nodes} nodes.")

        # Recalculate the decision paths, sort node list, and update related attributes
        self._get_decision_paths()
        self.node_list = sorted(self.node_list, key=lambda o: o.id)
        self._create_node_dict()
        
        # Update the max depth of the tree
        depth_list = [len(i) for i in self.decision_paths]
        self.max_depth_ = max(depth_list) - 1
        
        # Recalculate feature importance
        self._get_feature_importance()

    def _make_leaf(self, node, y):
        node.leaf_node = True
        node.left = None
        node.right = None
        if self.treetype == "classification":
            counter = Counter(y)
            node.clf_value_dis = [counter.get(0) or 0, counter.get(1) or 0]
            node.clf_prob_dis = (np.array(node.clf_value_dis) / node.samples)
            node.value = np.argmax(node.clf_prob_dis)
        elif self.treetype == "regression":
            node.value = self._mean_label(y)
        node.gini = self._gini(y)
        node.samples = len(y)

    def _get_y_for_node(self, node):
        sample_indices = node.sample_indices
        return self.y[sample_indices]

    def traverse_explain_path(self, x, node=None, dict_list=None):
        if dict_list is None:
            dict_list = []

        dict_node = {"node_id": node.id}

        if node.is_leaf_node():
            if self.treetype == "classification":
                dict_node.update([("value", node.value), ("prob_distribution", node.clf_prob_dis)])
                dict_list.append(dict_node)
                return [dic.get("node_id") for dic in dict_list], dict_list
            dict_node["value"] = node.value
            dict_list.append(dict_node)
            return [dic.get("node_id") for dic in dict_list], dict_list

        dict_node.update([("feature", node.feature_name or node.feature),
                          ("threshold", np.round(node.threshold, 3)),
                          ("value_observation", x[node.feature].round(3))])

        if x[node.feature] <= node.threshold:
            dict_node["decision"] = "{} <= {} --> left".format(x[node.feature].round(3), np.round(node.threshold, 3))
            dict_list.append(dict_node)
            return self.traverse_explain_path(x, node.left, dict_list)
        dict_node["decision"] = "{} > {} --> right".format(x[node.feature].round(3), np.round(node.threshold, 3))
        dict_list.append(dict_node)
        return self.traverse_explain_path(x, node.right, dict_list)

    def explain_decision_path(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self.traverse_explain_path(X, self.root), dtype="object")

        return np.array([self.traverse_explain_path(x, self.root) for x in X], dtype="object")

    def _apply_hierarchical_srinkage(self, treetype=None, HS_lambda=None, smSHAP_coefs=None, m_nodes=None, testHS=False):
        if treetype == None:
            treetype = self.treetype
        if HS_lambda == None:
            HS_lambda = self.HS_lambda

        if treetype == "regression":
            node_values_HS = np.zeros(len(self.node_list))
            for decision_path in self.decision_paths:
                cum_sum = 0
                for l, node_id in enumerate(decision_path):
                    if l == 0:
                        cum_sum = self.node_list[node_id].value
                        node_values_HS[node_id] = cum_sum
                        continue

                    current_node = self.node_list[node_id]
                    node_id_parent = decision_path[l - 1]
                    parent_node = self.node_list[node_id_parent]

                    if (smSHAP_coefs != None):
                        cum_sum += ((current_node.value - parent_node.value) / (1 + HS_lambda / parent_node.samples)) * np.abs(smSHAP_coefs[parent_node.feature])
                    elif (m_nodes != None):
                        cum_sum += ((current_node.value - parent_node.value) / (1 + HS_lambda / parent_node.samples)) * m_nodes[node_id]
                    else:
                        cum_sum += ((current_node.value - parent_node.value) / (1 + HS_lambda / parent_node.samples))

                    node_values_HS[node_id] = cum_sum

            for node_id, value in enumerate(node_values_HS):
                self.node_list[node_id].value = value

        elif treetype == "classification":
            clf_prob_dist = np.array(copy.deepcopy([node_id.clf_prob_dis for node_id in self.node_list]))
            node_samples = copy.deepcopy([node_id.samples for node_id in self.node_list])
            node_values_HS = np.zeros((len(node_samples), 2))

            for decision_path in self.decision_paths:
                node_values_ = np.zeros((len(node_samples), 2))
                cum_sum = 0
                for l, node_id in enumerate(decision_path):
                    if l == 0:
                        cum_sum = copy.deepcopy(clf_prob_dist[node_id])
                        node_values_[node_id] = cum_sum
                        continue

                    current_node = self.node_list[node_id]
                    node_id_parent = decision_path[l - 1]
                    parent_node = self.node_list[node_id_parent]

                    if (smSHAP_coefs != None):
                        cum_sum += ((clf_prob_dist[node_id] - clf_prob_dist[node_id_parent]) / (1 + HS_lambda / node_samples[node_id_parent])) * np.abs(smSHAP_coefs[parent_node.feature])
                    elif (m_nodes != None):
                        cum_sum += ((clf_prob_dist[node_id] - clf_prob_dist[node_id_parent]) / (1 + HS_lambda / node_samples[node_id_parent])) * m_nodes[node_id]
                    else:
                        cum_sum += ((clf_prob_dist[node_id] - clf_prob_dist[node_id_parent]) / (1 + HS_lambda / node_samples[node_id_parent]))

                    node_values_[node_id] = cum_sum
                for node_id in decision_path:
                    node_values_HS[node_id] = node_values_[node_id]

            for node_id in range(len(self.node_list)):
                self.node_list[node_id].clf_prob_dis = node_values_HS[node_id]
                self.node_list[node_id].value = np.argmax(self.node_list[node_id].clf_prob_dis)

        self.HS_applied = True

    def export_tree_for_SHAP(self, return_tree_dict=False):
        children_left = []
        children_right = []

        for node in self.node_list:
            if node.left is not None:
                children_left.append(node.left.id)
            else:
                children_left.append(-1)
            if node.right is not None:
                children_right.append(node.right.id)
            else:
                children_right.append(-1)

        children_left = np.array(children_left)
        children_right = np.array(children_right)
        children_default = children_right.copy()

        features = np.array([node.feature if node.feature is not None else -2 for node in self.node_list])
        thresholds = np.array([node.threshold if node.threshold is not None else -2 for node in self.node_list])

        if self.treetype == "regression":
            values = np.array([node.value for node in self.node_list])
        elif self.treetype == "classification":
            values = np.array([node.clf_prob_dis[1] for node in self.node_list])
        values = values.reshape(values.shape[0], 1)

        samples = np.array([float(node.samples) for node in self.node_list])

        tree_dict = {
            "children_left": children_left,
            "children_right": children_right,
            "children_default": children_default,
            "features": features,
            "thresholds": thresholds,
            "values": values,
            "node_sample_weight": samples
        }
        model = {"trees": [tree_dict]}

        if return_tree_dict:
            return model, tree_dict
        return model

    def _get_parent_node(self, node_id):
        return [node.id for node in self.node_list if (node.leaf_node == False) if ((node.left.id == node_id) | (node.right.id == node_id))][0]

    def _reestimate_node_values(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        traversed_nodes = self.explain_decision_path(X)[:, 0].copy()
        y_vals_array = np.full((self.n_nodes, X.shape[0]), np.nan)

        for i, (idxs, y) in enumerate(zip(traversed_nodes, y)):
            y_vals_array[list(idxs), [i]] = y

        nan_rows = np.argwhere(np.isnan(y_vals_array).all(axis=1)).flatten()

        if nan_rows.shape[0] != 0:
            for nan_node_id in nan_rows:
                par_node_id = self._get_parent_node(nan_node_id)
                y_vals_array[nan_node_id] = y_vals_array[par_node_id]

        result = {}

        if self.treetype == "regression":
            node_vals = np.nanmean(y_vals_array, axis=1)
            n_samples = np.count_nonzero(~np.isnan(y_vals_array), axis=1)

            for i in range(y_vals_array.shape[0]):
                result[i] = {"samples": n_samples[i], "value": node_vals[i]}
            node_vals = np.nanmean(y_vals_array, axis=1)
            return node_vals, result, nan_rows, y_vals_array

        elif self.treetype == "classification":
            for i in range(y_vals_array.shape[0]):
                n_samples = len(y_vals_array[i, :][~np.isnan(y_vals_array[i, :])])
                val, cnts = np.unique(y_vals_array[i, :][~np.isnan(y_vals_array[i, :])], return_counts=True)
                counts = {k: v for k, v in zip(val, cnts)}

                clf_value_dis = [counts.get(0) or 0, counts.get(1) or 0]
                clf_prob_dis = (np.array(clf_value_dis) / n_samples)
                leaf_value = np.argmax(clf_prob_dis)

                result[i] = {"samples": n_samples, "value": leaf_value, "value_distribution": clf_value_dis, "prob_distribution": clf_prob_dis}
            node_prob = np.array([(1 - val, val) for val in np.nanmean(y_vals_array, axis=1)])
            return node_prob, result, nan_rows, y_vals_array
