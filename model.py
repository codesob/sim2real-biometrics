import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features=512, out_features=1000, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label=None):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.s

        cosine = cosine.clamp(-1+1e-7, 1-1e-7)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        output *= self.s
        return output

class Sim2RealBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Sim2RealBackbone, self).__init__()
     
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.dropout = nn.Dropout(p=0.4) 
        self.fc = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x