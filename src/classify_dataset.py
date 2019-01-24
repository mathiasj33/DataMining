from Dataset import Dataset
from RandomForest import RandomForest
import ClassifierStats as stats

dataset_path = 'synthetic.social'
train = Dataset.from_file('../data/{}.train'.format(dataset_path))
test = Dataset.from_file('../data/{}.test'.format(dataset_path))
model = RandomForest(num_trees=100, max_depth=100, bagging_data_fraction=0.4)
model.train(train)
predictions = model.classify(test)
accuracy = stats.accuracy(test.labels, predictions)
print(accuracy)