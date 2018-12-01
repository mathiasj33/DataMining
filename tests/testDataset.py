import unittest
from Dataset import Dataset
from DecisionTree import DecisionTree, Node


class DatasetTest(unittest.TestCase):
    def test_gini(self):
        ds = Dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(ds.gini(), 0.459, places=3)

    def test_split(self):
        ds = Dataset([{1:1, 2:2}, {1:1, 2:3}, {1:2, 2:2}], [1,1,1])
        splits, _ = ds.split(1)
        self.assertEqual([{1:1, 2:2}, {1:1, 2:3}], splits[0].data)
        self.assertEqual([{1:2, 2:2}], splits[1].data)
        splits, _ = splits[0].split(2)
        self.assertEqual([{1: 1, 2: 2}], splits[0].data)
        self.assertEqual([{1: 1, 2: 3}], splits[1].data)
        splits, _ = ds.split(2)
        self.assertEqual([{1: 1, 2: 2}, {1:2, 2:2}], splits[0].data)
        self.assertEqual([{1:1, 2:3}], splits[1].data)

    def test_train_node(self):
        features = [
            {1:1, 2:1, 3:1},
            {1:1, 2:1, 3:2},
            {1:2, 2:1, 3:1},
            {1:3, 2:2, 3:1},
            {1:3, 2:3, 3:1},
            {1:3, 2:3, 3:2},
        ]
        labels = [0,0,1,1,1,0]
        ds = Dataset(features, labels)
        root = Node(ds)
        tree = DecisionTree(max_depth=100)
        root.train([1, 2, 3], 1, 100)
        print(tree)

        # correct_tree = Node(ds)
        # correct_tree.split_feature = 1
        # left = Node(Dataset(features[:2], labels[:2]))
        # middle = Node(Dataset(features[2], labels[2]))
        # right = Node(Dataset(features[3:], labels[3:]))
        # correct_tree.add_child(left, 1)
        # correct_tree.add_child(middle, 2)
        # correct_tree.add_child(right, 3)
        #
        # right.split_feature = 3
        # right_left = Node(Dataset(features[3:4], labels[3:4]))
        # right_right = Node(Dataset(features[5], labels[5]))
        # right.add_child(right_left, 1)
        # right.add_child(right_right, 2)

        # self.assertEqual(correct_tree, root)

    def test_train_and_classify(self):
        features = [
            {1: 1, 2: 1, 3: 1},
            {1: 1, 2: 1, 3: 2},
            {1: 2, 2: 1, 3: 1},
            {1: 3, 2: 2, 3: 1},
            {1: 3, 2: 3, 3: 1},
            {1: 3, 2: 3, 3: 2},
        ]
        labels = [0, 0, 1, 1, 1, 0]
        ds = Dataset(features, labels)
        tree = DecisionTree(max_depth=100)
        self.assertEqual(tree.train_and_classify(ds, ds), [0, 0, 1, 1, 1, 0])