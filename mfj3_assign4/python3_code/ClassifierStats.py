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