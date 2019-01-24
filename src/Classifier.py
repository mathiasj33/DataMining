from abc import ABC, abstractmethod

class Classifier(ABC):
    def classify(self, dataset):
        preds = []
        for features in dataset.data:
            preds.append(self.classify_instance(features))
        return preds

    @abstractmethod
    def classify_instance(self, features):
        pass