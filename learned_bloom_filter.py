import torch
import numpy as np
from random import sample
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def conv(x):
    a = format(x, '032b')
    l = list(str(a))
    l = np.array(list(map(int, l)))
    return l

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.out = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.out(x)
        return x
net = nn.Sequential(nn.Linear(32,2), nn.Sigmoid())
#net = Net()
print(net)
print(list(net.parameters()))
total = 1000000
x = [i for i in range(999, 2000)] # Positive if it's from 1000 - 2000

not_x = [i for i in range (0, total) if not i in x] # Negative otherwise

some_x = sample(not_x, 1000) # 1000 from outside as well.
x.extend(some_x)
not_x = [i for i in range(0,total) if not i in x]

x_true = np.array(x)
x_false = np.array(not_x)

number_of_examples = 500
x_positive = np.random.choice(x_true, number_of_examples)

lst = []
index = 0
while index < x_positive.size:
    lst.append(conv(x_positive[index]))
    index += 1

x_positive = np.array(lst)

y_positive = np.array([1] * number_of_examples)

x_negative = np.random.choice(x_false, number_of_examples)

index = 0
lst = []
while index < x_negative.size:
    lst.append(conv(x_negative[index]))
    index += 1

x_negative = np.array(lst)

y_negative = np.array([0] * number_of_examples)

x_total = np.concatenate((x_positive, x_negative), 0)
y_total = np.concatenate((y_positive, y_negative), 0)

x_tensor = torch.from_numpy(x_total).long().to('cpu')
y_tensor = torch.from_numpy(y_total).long().to('cpu')

dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [int(.8 * number_of_examples * 2), int(.2 * number_of_examples * 2)]) # 80% training, 20% validation
train_loader = DataLoader(dataset=train_dataset, batch_size=10)
val_loader = DataLoader(dataset=val_dataset, batch_size=10)

lr = 1e-1
n_epochs = 1000

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
losses = []
val_losses = []

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.float().to('cpu')
        y_batch = y_batch.to('cpu')

        net.train()

        #yhat = net(x_batch.float()).unsqueeze(dim=0)
        yhat = net(x_batch.float())
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())


with torch.no_grad():
    correct = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')
        
        net.eval()
        
        yhat = net(x_val.float())
        _, predict = torch.max(yhat.data, 1)
        print("y_val:")
        print(y_val)
        print(yhat.data)
        print(predict)
        total += 10
        for i in range(10):
            print(y_val[i] == predict[i])
            correct += (y_val[i] == predict[i])

    print("Hello")
    print(correct)
    print(total)


print(net.state_dict())
