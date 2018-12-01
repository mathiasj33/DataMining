from Dataset import Dataset


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def train_and_classify(self, train, test):
        features = list(train.data[0].keys())  # all possible features
        self.root = Node(train)
        self.root.train(features, 1, self.max_depth)
        print('trained')
        return self.classify(test)

    def classify(self, dataset):
        preds = []
        for features in dataset.data:
            node = self.root
            while node.children:
                found_child = False
                for c,v in node.children:
                    if v == features[node.split_feature]:
                        node = c
                        found_child = True
                        break
                if not found_child:
                    node = node.children[0][0]  # randomly pick the first path and continue classification
            preds.append(node.dataset.majority_label())
        return preds


class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.children = []
        self.split_feature = None

    def train(self, features, depth, max_depth):
        if depth >= max_depth: return
        if len(set(self.dataset.labels)) == 1: return  # all labels belong to the same class
        if len(self.dataset.data) == 1: return
        if len(features) == 0: return

        current_gini = self.dataset.gini()
        split_ginis = []
        for feature in features:
            splits, _ = self.dataset.split(feature)
            split_gini = sum((len(subset.data) / len(self.dataset.data)) * subset.gini() for subset in splits)
            split_ginis.append(split_gini)

        if not split_ginis: return
        smallest = split_ginis[0]
        smallest_index = 0
        for i in range(len(split_ginis)):
            if split_ginis[i] < smallest:
                smallest = split_ginis[i]
                smallest_index = i
        if current_gini < smallest: return

        split_feature = features[smallest_index]
        self.split_feature = split_feature
        new_features = features.copy()
        new_features.remove(split_feature)
        splits, pos_to_value = self.dataset.split(split_feature)
        for i in range(len(splits)):
            subset = splits[i]
            child = Node(subset)
            child.train(new_features, depth + 1, max_depth)
            self.children.append((child, pos_to_value[i]))


if __name__ == '__main__':
    ds = Dataset.from_file('../../data/balance.scale.train')
    print(ds)