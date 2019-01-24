from Dataset import Dataset
import ClassifierStats as stats

def print_confusion_matrix(conf):
    for i in range(len(conf)):
        for j in range(len(conf)):
            print(conf[i][j], end ="")
            if j == len(conf) - 1: print('')
            else: print(' ', end ="")

def train_and_print_results(argv, model):
    train_path = argv[1]
    test_path = argv[2]
    train = Dataset.from_file(train_path)
    test = Dataset.from_file(test_path)
    predictions = model.train_and_classify(train, test)
    conf_matrix = stats.confusion_matrix(test.labels, predictions, max(train.labels + test.labels))
    print_confusion_matrix(conf_matrix)