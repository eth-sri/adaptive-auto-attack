import torch
import torch.nn as nn

class CCATDetector(nn.Module):
    def __init__(self, classifier, threshold):
        super(CCATDetector, self).__init__()
        self.classifier = classifier
        self.threshold = threshold

    def forward(self, x):
        logits = self.classifier(x)
        prob = logits.softmax(dim=1)
        confidence = prob.max(dim=1)[0]
        # print(prob)
        # print(prob.shape)
        # print(confidence)
        # print(confidence.shape)
        # exit()
        return confidence > self.threshold
