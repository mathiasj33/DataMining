from DecisionTree import DecisionTree
from Dataset import Dataset

from collections import defaultdict, Counter
import math
import random
random.seed(0)

class RandomForest:
    def __init__(self, n_trees, max_depth, data_fraction, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.data_fraction = data_fraction
        self.n_features = n_features
        self.trees = []

    def train_and_classify(self, train, test):
        self.train(train)
        return self.classify(test)

    def train(self, train):
        if self.n_features is None:
            self.n_features = int(math.sqrt(len(train.data[0].keys())))
        for i in range(self.n_trees):
            self.train_random_tree(train)

    def train_random_tree(self, train):
        sample_indices = [random.choice(range(len(train.data))) for _ in range(int(self.data_fraction * len(train.data)))]
        sample = Dataset([train.data[i] for i in sample_indices], [train.labels[i] for i in sample_indices])
        tree = DecisionTree(self.max_depth, self.n_features)
        tree.train(sample)
        self.trees.append(tree)

    def classify(self, test):
        preds = []
        for features in test.data:
            preds.append(self.classify_instance(features))
        return preds

    def classify_instance(self, features):
        votes = []
        for tree in self.trees:
            votes.append(tree.classify_instance(features))
        return Counter(votes).most_common(1)[0][0]
