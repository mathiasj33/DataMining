from collections import defaultdict, Counter


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

    def gini(self):
        class_freqs = self.class_frequencies()
        return 1 - sum([p**2 for p in class_freqs.values()])

    def class_frequencies(self):
        freqs = defaultdict(int)
        for label in self.labels:
            freqs[label] += 1
        for label in freqs.keys():
            freqs[label] /= len(self.labels)
        return freqs

    def split(self, feature):
        value_to_pos = {}
        splits = []
        for i in range(len(self.data)):
            features = self.data[i]
            label = self.labels[i]
            value = features[feature]
            if value not in value_to_pos:
                splits.append(([],[]))
                value_to_pos[value] = len(splits) - 1
            splits[value_to_pos[value]][0].append(features)
            splits[value_to_pos[value]][1].append(label)
        return [Dataset(s[0], s[1]) for s in splits], {v:k for k,v in value_to_pos.items()}

    def majority_label(self):
        return Counter(self.labels).most_common(1)[0][0]