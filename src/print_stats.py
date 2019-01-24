from DecisionTree import get_model_for_train_file as get_tree
from RandomForest import get_model_for_train_file as get_forest
from Dataset import Dataset
import ClassifierStats as stats
import math

for ds_string in ['balance.scale', 'nursery', 'led', 'synthetic.social']:
    train_path = '../../data/{}.train'.format(ds_string)
    test_path = '../../data/{}.test'.format(ds_string)
    train = Dataset.from_file(train_path)
    test = Dataset.from_file(test_path)
    print(ds_string)
    for model in [get_tree(train_path), get_forest(train_path)]:
        print('\t %s' % model)
        for evaluate_on in [train, test]:
            print('\t \t %s' % ('train' if evaluate_on == train else 'test'))
            predictions = model.train_and_classify(train, evaluate_on)
            print('\t \t \t Accuracy: %.3f' % stats.accuracy(evaluate_on.labels, predictions))
            for c in set(train.labels + test.labels):
                f1_score = stats.f1_score(evaluate_on.labels, predictions, c)
                print('\t \t \t F1 score on class %d: %.3f' % (c, f1_score))


