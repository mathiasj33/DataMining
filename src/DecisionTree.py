from Classifier import Classifier
import random
random.seed(0)

class DecisionTree(Classifier):
    def __init__(self, max_depth, num_features = None):
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def train(self, train):
        features = list(train.data[0].keys())  # all possible features
        if self.num_features is None: self.num_features = len(features)
        self.root = Node(train)
        self.root.train(features, 1, self.max_depth, self.num_features)

    def classify_instance(self, features):
        node = self.root
        while node.children:
            found_child = False
            for c, v in node.children:
                if v == features[node.split_feature]:
                    node = c
                    found_child = True
                    break
            if not found_child:
                node = random.choice(node.children)[0]  # randomly pick a path and continue classification
        return node.dataset.majority_label()

    def __str__(self):
        return 'DecisionTree'


class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.children = []
        self.split_feature = None

    def train(self, features, depth, max_depth, num_features):
        if depth >= max_depth: return
        if len(set(self.dataset.labels)) == 1: return  # all labels belong to the same class
        if len(self.dataset.data) == 1: return
        if len(features) == 0: return

        if num_features >= len(features):
            considered_features = features
        else:
            considered_features = random.sample(features, num_features)

        current_gini, split_ginis = self.calculate_gini_indices(considered_features)

        if not split_ginis: return
        min_key, min_val = min(split_ginis.items(), key=lambda x: x[1])
        if current_gini < min_val: return

        split_feature = min_key
        self.split_feature = split_feature
        new_features = features.copy()
        new_features.remove(split_feature)
        splits, pos_to_feature_value = self.dataset.split_on_feature(split_feature)
        for i in range(len(splits)):
            subset = splits[i]
            child = Node(subset)
            child.train(new_features, depth + 1, max_depth, num_features)
            self.children.append((child, pos_to_feature_value[i]))

    def calculate_gini_indices(self, considered_features):
        current_gini = self.dataset.calculate_gini_index()
        split_ginis = {}
        for feature in considered_features:
            splits, _ = self.dataset.split_on_feature(feature)
            split_gini = sum((len(subset.data) / len(self.dataset.data)) * subset.calculate_gini_index()
                             for subset in splits)
            split_ginis[feature] = split_gini
        return current_gini, split_ginis
