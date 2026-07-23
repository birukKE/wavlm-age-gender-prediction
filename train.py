
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
age_criterion = ConcordanceCorrCoef()
# age_criterion = nn.L1Loss()

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
    # return (values - 0.1) / (0.9)
    return values



writer = SummaryWriter()
num_epochs = 5
for epoch in range(num_epochs):
#     print("In the outer loop")
    net.train()
    train_loss = 0.0
    for inputs, labels_gender, labels_age in train_loader:
        optimizer.zero_grad()
        gender_outputs, age_outputs = net(inputs)
        loss_gender = cross_entrophy_criterion(gender_outputs, labels_gender.long())
        
        loss_age = 1 - age_criterion(normalized(age_outputs), labels_age)
        # loss_age = age_criterion(normalized(age_outputs), labels_age)
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


age_prediction = []
age_true = []
test_loss = 0.0
net.eval()
age_accr = 0
gender_accr = 0
i = 0
for inputs, labels_gender, labels_age in test_loader:
    gender_outputs, age_outputs = net(inputs)
    age_accr += check_age_accuracy(normalized(age_outputs), labels_age)
    gender_accr += check_gender_accuracy(gender_outputs, labels_gender)

    age_prediction.extend(age_outputs.detach().numpy())
    age_true.extend(labels_age.detach().numpy())

# print("len(test_dataset) = ", len(test_dataset))
age_accr = age_accr / len(test_dataset)
gender_accr = gender_accr / len(test_dataset)
print(f'age_true = {type(age_true[0])}  and  age_prediciton = {type(age_prediction)}')
MAE = mean_absolute_error(age_true, age_prediction)
spearman_correaltion = spearmanr(age_true, age_prediction)

print(f'Age accuracy = {age_accr}      gender accuracy = {gender_accr}')
print(f'MAE of age = ', MAE)
print(f'Spearmans correlation of age = ', spearman_correaltion)





























# # my_test_data = np.load("swedish_testdata.npz", allow_pickle=True)

# # x = my_test_data["xtest_data"]

# # y_gender = my_test_data["y_test_gender_labels"]
# # transform_y_gender = le_gender.transform(y_gender) # decoded = le.inverse_transform(encoded)   <-   To do the opposite

# # y_age = my_test_data["y_test_age_labels"]
# # transform_y_age = [y_age_trans[age_group] for age_group in y_age]

# # y_test_age_labels = torch.tensor(transform_y_age)
# # y_test_gender_labels = torch.tensor(transform_y_gender)
# # test_x = torch.tensor(x)

# # print("input = ", input_dim)
# # print("output = ", train_x.shape)
# # print("train_y = ", y_train_age_labels.shape)
# # print("train_y = ", y_train_gender_labels.shape)



# # test_dataset = torch.utils.data.TensorDataset(test_x, y_test_gender_labels, y_test_age_labels)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# # net.eval()

# # test_loss = 0.0
# # for inputs, labels_gender, labels_age in train_loader:
# #     optimizer.zero_grad()
# #     gender_outputs, age_outputs = net(inputs)
# #     loss_gender = cross_entrophy_criterion(gender_outputs, labels_gender.long())
# #     loss_age = 1 - age_criterion(age_outputs, labels_age)
# #     loss = (loss_gender + loss_age)/2
# #     loss.backward()
# #     optimizer.step()
# #     test_loss += loss.item()
        
# #     # print("Done with the inner loop")

# # #     # print the epoch loss
# #     test_loss /= len(train_loader)

# #     print(f'Epoch {epoch}: train_loss={test_loss}')

#     # outputs = net(inputs)
#     # print("outputs = ", outputs)
#     # print("labels = ", labels)
#     # total_accuracy += check_accuracy(outputs, labels)

# # n_samples = len(test_dataset)
# # accuracy_rate = total_accuracy / n_samples




# # # Testing
# # print("Starting with testing!")
# # # output_dim = len(stateList)
# # # my_test_data = np.load("training_data.npz")
# # my_test_data = np.load("noisy_training_data.npz")

# # batch_size_test = 4

# # x = my_train_data["x_test"]
# # y = le.transform(my_train_data["y_test"])

# # test_x = torch.tensor(x)
# # test_y = torch.tensor(y)

# # print("Shape of my test x: ", test_x.shape)
# # print("Shape of my test y: ", test_y.shape)

# # test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# # print("done loading test data")



# # le_age = LabelEncoder()
# # y_age = my_train_data["y_train_gender_labels"]
# # unique_labels = list(set(y_age.tolist()))
# # le_age.fit(unique_labels)



# # def check_accuracy(outputs, labels):
# #     # Turn the one hot encoding in to 1 dimensional arrya with the actual values
# #     # targets = torch.argmax(labels, dim=1)  # dim 1 refers to columns
# #     # outputs =  tensor([[-0.5255,  0.7507],
# #     #                    [-0.9177,  1.2567],
# #     # labels =  tensor([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
# #     # Output returns logits of proabbailities, the column index's are the lables, so in the first row, the model guesses that it i likely index 1 AKA class 1
# #     # So I take the max indices from each row wbecause they represent teh clas
# #     predictions = torch.argmax(outputs, dim=1)
# #     tot_correct_per_batch = torch.sum(predictions == labels)
    
# #     return tot_correct_per_batch

# # total_accuracy = 0.0
# # net.eval()

# # print("before for loop")
# # for inputs, labels in test_loader:
# #     outputs = net(inputs)
# #     # print("outputs = ", outputs)
# #     # print("labels = ", labels)
# #     total_accuracy += check_accuracy(outputs, labels)

# # n_samples = len(test_dataset)
# # accuracy_rate = total_accuracy / n_samples

# # print("Accuracy on the test data: ", accuracy_rate)