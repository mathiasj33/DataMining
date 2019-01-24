def check_length(true, pred):
    if len(true) != len(pred): raise ValueError('Different lengths of true and predicted')

def accuracy(true, pred):
    check_length(true, pred)
    return sum([1 for i, j in zip(true, pred) if i == j]) / len(true)

def confusion_matrix(true, pred, num_classes):
    check_length(true, pred)
    matrix = [[None for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i][j] = sum(1 for t,p in zip(true, pred) if t == i+1 and p == j+1)
    return matrix

def f1_score(true, pred, positive_class):
    try:
        precision = sum(1 for i,j in zip(true,pred) if i == positive_class and j == i) \
                / sum(1 for i in pred if i == positive_class)
        recall = sum(1 for i, j in zip(true, pred) if i == positive_class and j == i) \
                / sum(1 for i in true if i == positive_class)
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return float('NaN')