from Dataset import Dataset
import random
random.seed(0)

class DecisionTree:
    def __init__(self, max_depth, n_features = None):
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def train_and_classify(self, train, test):
        self.train(train)
        return self.classify(test)

    def train(self, train):
        features = list(train.data[0].keys())  # all possible features
        if self.n_features is None: self.n_features = len(features)
        self.root = Node(train)
        self.root.train(features, 1, self.max_depth, self.n_features)

    def classify(self, dataset):
        preds = []
        for features in dataset.data:
            preds.append(self.classify_instance(features))
        return preds

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
                node = node.children[0][0]  # randomly pick the first path and continue classification
        return node.dataset.majority_label()


class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.children = []
        self.split_feature = None

    def train(self, features, depth, max_depth, n_features):
        if depth >= max_depth: return
        if len(set(self.dataset.labels)) == 1: return  # all labels belong to the same class
        if len(self.dataset.data) == 1: return
        if len(features) == 0: return

        if n_features >= len(features):
            considered_features = features
        else:
            considered_features = random.sample(features, n_features)

        current_gini = self.dataset.gini()
        split_ginis = {}
        for feature in considered_features:
            splits, _ = self.dataset.split(feature)
            split_gini = sum((len(subset.data) / len(self.dataset.data)) * subset.gini() for subset in splits)
            split_ginis[feature] = split_gini

        if not split_ginis: return
        min_key, min_val = min(split_ginis.items(), key=lambda x: x[1])
        if current_gini < min_val: return

        split_feature = min_key
        self.split_feature = split_feature
        new_features = features.copy()
        new_features.remove(split_feature)
        splits, pos_to_value = self.dataset.split(split_feature)
        for i in range(len(splits)):
            subset = splits[i]
            child = Node(subset)
            child.train(new_features, depth + 1, max_depth, n_features)
            self.children.append((child, pos_to_value[i]))


if __name__ == '__main__':
    ds = Dataset.from_file('../../data/balance.scale.train')
    print(ds)