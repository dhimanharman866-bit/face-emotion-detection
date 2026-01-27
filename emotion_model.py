import torch
import torch.nn as nn

class Emotiondetection(nn.Module):
  def __init__(self):
    super().__init__()

    self.feature=nn.Sequential(
        # first convolutional layer 
        nn.Conv2d(1,32,3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),


        # second convolutional layer 
        nn.Conv2d(32,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),


        # third layer
        nn.Conv2d(64,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128,256,3,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256,7)
    
    )
  def forward(self,x):
    x=self.feature(x)
    return self.classifier(x)