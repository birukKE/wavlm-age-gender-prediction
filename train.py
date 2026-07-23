
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchmetrics import ConcordanceCorrCoef, MeanAbsoluteError
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from wavLm.data_preparation import*


class Net(torch.nn.Module):
    def __init__(self, num_of_gender_labels = 2, input_dim = 768):
        super().__init__()
        self.shared_backbone  = nn.Linear(input_dim, 512)

        self.age = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.gender = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )


    def forward(self, x):
        x = self.shared_backbone(x)
        x_age = self.age(x)
        x_gender = self.gender(x)
        return x_gender, x_age.squeeze()


net = Net()
# criterion = nn.BCEWithLogitsLoss()
cross_entrophy_criterion = nn.CrossEntropyLoss()
# age_criterion = ConcordanceCorrCoef()
age_criterion = nn.L1Loss()

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

#############################################################################################################################################
# Training
#############################################################################################################################################
def check_gender_accuracy(outputs, labels):
    predictions = torch.argmax(outputs, dim=1)
    tot_correct_per_batch = torch.sum(predictions == labels)
    return tot_correct_per_batch

def check_age_accuracy(outputs, labels):
    tot_correct_per_batch = 0
    tolerance = 0.05
    for idx, output in enumerate(outputs):
        if abs(output - labels[idx]) < tolerance:
            tot_correct_per_batch+=1
    return tot_correct_per_batch

def normalized(values):
    return (values - 0.1) / (0.9)

writer = SummaryWriter()
num_epochs = 5
for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    for inputs, labels_gender, labels_age in train_loader:
        optimizer.zero_grad()
        gender_outputs, age_outputs = net(inputs)
        loss_gender = cross_entrophy_criterion(gender_outputs, labels_gender.long())
        
        loss_age = age_criterion(normalized(age_outputs), labels_age)
        loss = (loss_gender + loss_age)/2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels_gender, labels_age in dev_loader:
            gender_outputs, age_outputs = net(inputs)
            loss_gender = cross_entrophy_criterion(gender_outputs, labels_gender.long())
            loss_age = 1 - age_criterion(normalized(age_outputs), labels_age)
            loss = (loss_gender + loss_age)/2
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(dev_loader)

    print(f'Epoch {epoch}: train_loss = {train_loss}    val_loss = {val_loss}')

    # writer.add_scalars('loss',{'train':train_loss}, epoch)  # writer.add_scalars('loss',{'train':train_loss,'val':val_loss},epoch)

torch.save(net.state_dict(), 'trained-net.pt')



