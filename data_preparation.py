import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import torch, torchmetrics


def prepare_data_4_training(data_filepath):
    x_data = [] 
    y_gender_labels = []
    y_age_labels = []

    my_loaded_data = np.load(data_filepath, allow_pickle=True)['data']
    random.shuffle(my_loaded_data)
    count_men = 0
    for item in my_loaded_data:
        if item["gender"]=="male_masculine":
            if count_men>=550:
                continue
            count_men+=1
    
        x_data.append(item["embedding"])
        y_gender_labels.append(item["gender"])
        y_age_labels.append(item["age"])
    return x_data, y_gender_labels, y_age_labels


xtrain_data, y_train_gender_labels, y_train_age_labels = prepare_data_4_training('data/traindata_wavLm_modeled-base.npz')
xtest_data, y_test_gender_labels, y_test_age_labels = prepare_data_4_training('data/testdata_wavLm_modeled-base.npz')
xdev_data, y_dev_gender_labels, y_dev_age_labels = prepare_data_4_training('data/devdata_wavLm_modeled-base.npz')

#########################################################################################################
# General data mappings and settings
#########################################################################################################
BATCH_SIZE = 16

# age_groups = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]
# y_age_trans = {age_groups[i-1]: float(10*i + 5)/100 for i in range(1, 10)}
# y_age_trans = { "teens": 15, "twenties": 25, "thirties": 35, "fourties": 45, "fifties": 55, "sixties": 65, "seventies":75, "eighties": 85, "nineties": 95}
# y_age_trans = { "teens": 0.15, "twenties": 0.25, "thirties": 0.35, "fourties": 0.45, "fifties": 0.55, "sixties": 0.65, "seventies":0.75, "eighties": 0.85, "nineties": 0.95}
y_age_trans = { "teens": 15, "twenties": 25, "thirties": 35, "fourties": 45, "fifties": 55, "sixties": 65, "seventies":75, "eighties": 85, "nineties": 95}
y_gender_trans = {"female_feminine":0, "male_masculine": 1}
unique_labels = list(set(y_train_gender_labels))
input_dim = xtrain_data[0].shape[0]


#########################################################################################################
# Train data
#########################################################################################################
transform_y_gender = [y_gender_trans[gender]  for gender in y_train_gender_labels]
transform_y_age = [y_age_trans[age_group] for age_group in y_train_age_labels]

y_train_age_labels = torch.tensor(transform_y_age)
y_train_gender_labels = torch.tensor(transform_y_gender)
train_x = torch.tensor(xtrain_data)

train_dataset = torch.utils.data.TensorDataset(train_x, y_train_gender_labels, y_train_age_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


#########################################################################################################
# Test data
#########################################################################################################
transform_y_gender = [y_gender_trans[gender]  for gender in y_test_gender_labels]
transform_y_age = [y_age_trans[age_group] for age_group in y_test_age_labels]

y_test_age_labels = torch.tensor(transform_y_age)
y_test_gender_labels = torch.tensor(transform_y_gender)
test_x = torch.tensor(xtest_data)

test_dataset = torch.utils.data.TensorDataset(test_x, y_test_gender_labels, y_test_age_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


#########################################################################################################
# Dev data
#########################################################################################################
transform_y_gender = [y_gender_trans[gender]  for gender in y_dev_gender_labels]
transform_y_age = [y_age_trans[age_group] for age_group in y_dev_age_labels]

y_dev_age_labels = torch.tensor(transform_y_age)
y_dev_gender_labels = torch.tensor(transform_y_gender)
dev_x = torch.tensor(xdev_data)

dev_dataset = torch.utils.data.TensorDataset(dev_x, y_dev_gender_labels, y_dev_age_labels)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
