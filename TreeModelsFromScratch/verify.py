import numpy as np
from RandomForest import RandomForest

def print_node_info(node):
    print(f"Node ID: {node.id}")
    print(f"Samples: {node.samples}")
    print(f"Sample Indices: {node.sample_indices}")
    print(f"Class Value Distribution: {node.clf_value_dis}")
    print(f"Class Probability Distribution: {node.clf_prob_dis}")
    print(f"Gini Impurity: {node.gini}\n")

    if not node.is_leaf_node():
        print("\nLeft Child Node Info:")
        print_node_info(node.left)

        print("\nRight Child Node Info:")
        print_node_info(node.right)


# Small dataset
X_train = np.array([[2.5, 3.0],
                    [1.5, 2.5],
                    [3.5, 3.5],
                    [3.0, 2.0],
                    [2.0, 1.5]])
y_train = np.array([0, 1, 0, 1, 0])

# Create and fit a RandomForest
rf = RandomForest(n_trees=3, max_depth=2, random_state=42)
rf.fit(X_train, y_train)

# Choose a sample to add to the tree
new_sample_index = 0
new_sample_x = X_train[new_sample_index]
new_sample_y = y_train[new_sample_index]

# Traverse and add the new sample to each tree
for tree in rf.trees:
    print(f"\nBefore adding new sample to tree with root node {tree.root.id}:\n")
    print_node_info(tree.root)
    tree.traverse_add_path(new_sample_x, new_sample_index, new_sample_y)
    print(f"\nAfter adding new sample to tree with root node {tree.root.id}:\n")
    print_node_info(tree.root)

import unittest
import numpy as np
from DecisionTree import DecisionTree
from RandomForest import RandomForest

class TestRandomForest(unittest.TestCase):

    def setUp(self):
        # Small dataset
        self.X_train = np.array([[2.5, 3.0],
                                 [1.5, 2.5],
                                 [3.5, 3.5],
                                 [3.0, 2.0],
                                 [2.0, 1.5]])
        self.y_train = np.array([0, 1, 0, 1, 0])
        self.rf = RandomForest(n_trees=3, max_depth=2, random_state=42)

    def test_fit(self):
        self.rf.fit(self.X_train, self.y_train)
        self.assertEqual(len(self.rf.trees), 3)
        self.assertEqual(self.rf.trees[0].root.samples, len(self.X_train))

    def test_bootstrap_sampling(self):
        X_inbag, y_inbag, idxs_inbag = self.rf._bootstrap_samples(self.X_train, self.y_train, bootstrap=True, random_state=self.rf.random_state_)
        self.assertEqual(len(idxs_inbag), len(self.X_train))
        self.assertTrue(np.all(np.isin(idxs_inbag, np.arange(len(self.X_train)))))

    def test_oob_samples(self):
        X_inbag, y_inbag, idxs_inbag = self.rf._bootstrap_samples(self.X_train, self.y_train, bootstrap=True, random_state=self.rf.random_state_)
        X_oob, y_oob, idxs_oob = self.rf._oob_samples(self.X_train, self.y_train, idxs_inbag)
        self.assertEqual(len(np.setdiff1d(np.arange(len(self.X_train)), idxs_inbag)), len(idxs_oob))

    def test_tree_growth(self):
        self.rf.fit(self.X_train, self.y_train)
        for tree in self.rf.trees:
            self.assertTrue(tree.max_depth is not None)
            self.assertTrue(tree.min_samples_split <= len(self.X_train))

    def test_shap_values(self):
        self.rf.fit(self.X_train, self.y_train)
        shap_values = self.rf.trees[0].predict_proba(self.X_train)
        self.assertEqual(shap_values.shape[0], len(self.X_train))

    def test_random_state_consistency(self):
        rf1 = RandomForest(n_trees=3, max_depth=2, random_state=42)
        rf2 = RandomForest(n_trees=3, max_depth=2, random_state=42)
        rf1.fit(self.X_train, self.y_train)
        rf2.fit(self.X_train, self.y_train)
        self.assertEqual(rf1.trees[0].root.samples, rf2.trees[0].root.samples)

if __name__ == '__main__':
    unittest.main()
