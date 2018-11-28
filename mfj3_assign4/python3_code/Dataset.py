class Dataset:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    @staticmethod
    def from_file(path):
        data = []
        labels = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(' ')
                labels.append(int(parts[0]))
                features = {}
                for i in range(1, len(parts)):
                    feature_value = parts[i].split(':')
                    features[feature_value[0]] = feature_value[1]
                data.append(features)
        return Dataset(data, labels)
