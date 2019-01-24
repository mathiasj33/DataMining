from Classifier import Classifier
from DecisionTree import DecisionTree
from Dataset import Dataset

from collections import Counter
import math
import random
random.seed(0)

class RandomForest(Classifier):
    def __init__(self, num_trees, max_depth, bagging_data_fraction, num_features = None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.bagging_data_fraction = bagging_data_fraction
        self.num_features = num_features
        self.trees = []

    def train(self, train):
        if self.num_features is None:
            total_num_features = len(train.data[0].keys())
            # considering sqrt(total_num_features) works relatively well empirically
            self.num_features = int(math.sqrt(total_num_features))
        for i in range(self.num_trees):
            self.train_decision_tree(train)

    def train_decision_tree(self, train):
        sample_indices = [random.choice(range(len(train.data))) for _ in
                          range(int(self.bagging_data_fraction * len(train.data)))]
        sample = Dataset([train.data[i] for i in sample_indices], [train.labels[i] for i in sample_indices])
        tree = DecisionTree(self.max_depth, self.num_features)
        tree.train(sample)
        self.trees.append(tree)

    def classify_instance(self, features):
        votes = []
        for tree in self.trees:
            votes.append(tree.classify_instance(features))
        return Counter(votes).most_common(1)[0][0]

    def __str__(self):
        return 'RandomForest'
