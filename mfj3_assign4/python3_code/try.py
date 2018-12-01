from Dataset import Dataset
from DecisionTree import DecisionTree
import ClassifierStats as stats
# from sklearn.tree import DecisionTreeClassifier as SKTree
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import accuracy_score

dataset = 'balance.scale'
train = Dataset.from_file('/home/mathias/PycharmProjects/DataMiningClassification/data/{}.train'.format(dataset))
test = Dataset.from_file('/home/mathias/PycharmProjects/DataMiningClassification/data/{}.test'.format(dataset))
tree = DecisionTree(100)
predictions = tree.train_and_classify(train, test)
accuracy = stats.accuracy(test.labels, predictions)
conf_matrix = stats.confusion_matrix(test.labels, predictions, len(set(train.labels + test.labels)))


# sktree = SKTree()
# dv = DictVectorizer()
# X = dv.fit_transform(train.data)
# sktree.fit(X, train.labels)
# X_test = dv.transform(test.data)
# skaccuracy = accuracy_score(sktree.predict(X_test), test.labels)
