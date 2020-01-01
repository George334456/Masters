import torch
import pickle
from BloomFilter import BloomFilter
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

def correct_check(x, y):
    if x.item() is True and y.item() == 1:
        return 1
    if x.item() is False and y.item() == 0:
        return 1
    return 0

def generate_data(number = 1000000):
    x = [i for i in range( 999, 2000 )] # positive if it's from 1000 - 2000
    pickle.dump(x, open("in_range_x.dump", "wb"))
    not_x = [i for i in range(0, number) if not i in x] # Negative otherwise
    some_x = sample(not_x, 1000) # 1000 numbers from outside
    pickle.dump(some_x, open("out_range_x.dump", "wb"))

    x.extend(some_x)
    not_x = [ i for i in range(0, number) if not i in x] # Recalculate the negative ones.
    pickle.dump(not_x, open("negative_x.dump", "wb"))

def generate_data_restricted_range(total_range, lst, start, end):
    # Generate data from total_range that has interval [start, end] and 
    # does not appear in lst.
    result = []
    for i in total_range:
        if i <= end and i >= start and i not in lst:
            result.append(i)
    return result

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32,1)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x
    def train_model(self, train_loader, val_loader):
        # Train step
        lr = 1e-1
        n_epochs = 1000
        
        loss_fn = nn.BCELoss(reduction='mean')
        optimizer = optim.SGD(net.parameters(), lr=lr)
        losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float().to('cpu')
                y_batch = y_batch.float().to('cpu')
        
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
                
                self.eval()
                
                yhat = net(x_val.float())
                
                predict = torch.gt(yhat.data, 0.5)
                #_, predict = torch.max(yhat.data, 1)
                total += 10
                for i in range(10):
                    correct += (correct_check(predict[i], y_val[i]))
        
            print("Hello")
            print(correct)
            print(total)
        torch.save(self, "./model.dump")

#net = nn.Sequential(nn.Linear(32,2), nn.Sigmoid())
net = Net()
print(net)
print(list(net.parameters()))

# Dataset generation and load

# generate_data(1000000)
in_range = pickle.load(open("in_range_x.dump", "rb"))
out_range = pickle.load(open("out_range_x.dump", "rb"))
x = in_range + out_range
not_x = pickle.load(open("negative_x.dump", "rb"))

x_true = np.array(x)
x_false = np.array(not_x)

number_of_examples = 1000

# Get exactly half of number of examples from outside as well as inside.
x_positive = []
x_positive.extend(np.random.choice(in_range, int(number_of_examples / 2)))
x_positive.extend(np.random.choice(out_range, int(number_of_examples/2)))
x_positive = np.array(x_positive)

x_positive_size = x_positive.size

# Generate a dataset of all our positive data.
lst = []
index = 0
while index < x_true.size:
    lst.append(conv(x_true[index]))
    index += 1
x_true_data = np.array(lst)
y_true = np.array([1] * x_true.size)

x_true_data = torch.from_numpy(x_true_data).long().to('cpu')
y_true = torch.from_numpy(y_true).long().to('cpu')

total = TensorDataset(x_true_data, y_true)

# All our positive keys are put into this true_loader
true_loader = DataLoader(dataset=total, batch_size = 10)

# Generate our training tensors. Convert to binary as well. They start as numpy arrays.
lst = []
index = 0
while index < x_positive.size:
    lst.append(conv(x_positive[index]))
    index += 1

x_positive = np.array(lst)

y_positive = np.array([1] * number_of_examples)
# These are the keys to insert.

positive_keys_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_positive).long().to('cpu'), torch.from_numpy(y_positive).long().to('cpu')), batch_size = 10)

x_negative = np.random.choice(x_false, number_of_examples)
print("The size of x_negative is {}", x_negative.size)

index = 0
lst = []
while index < x_negative.size:
    lst.append(conv(x_negative[index]))
    index += 1

x_negative = np.array(lst)

y_negative = np.array([0] * number_of_examples)
negative_keys_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_negative).long().to('cpu'), torch.from_numpy(y_negative).long().to('cpu')), batch_size = 10)

x_total = np.concatenate((x_positive, x_negative), 0)
y_total = np.concatenate((y_positive, y_negative), 0)

x_tensor = torch.from_numpy(x_total).long().to('cpu')
y_tensor = torch.from_numpy(y_total).long().to('cpu')
print(x_tensor.size(0))
print(y_tensor.size(0))

dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [int(.8 * number_of_examples * 2), int(.2 * number_of_examples * 2)]) # 80% training, 20% validation
train_loader = DataLoader(dataset=train_dataset, batch_size=10)
val_loader = DataLoader(dataset=val_dataset, batch_size=10)

#net.train_model(train_loader, val_loader)

net = torch.load("./model.dump")
net.eval()

print(net.state_dict())

tau = 0.5
# Find tau
with torch.no_grad():
    number = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')

        net.eval()

        yhat = net(x_val.float())

        for i in range(y_val.size(0)):
            # If it's not supposed to be a key
            if y_val[i] == 0:
                if yhat[i] > tau:
                    number += 1
            total += 1
    print("{} number out of {} total. {}".format( number, total, number/total ))


# Find the false negatives
tau = 0.5
count = 0 
bf_lst = []
# In this case, our key set is x_train.

with torch.no_grad():
    for x_val, y_val in positive_keys_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')

        net.eval()

        yhat = net(x_val.float())
        for i in range(y_val.size(0)):
            if yhat[i] >= tau:
                continue
            else:
                k = x_val[i].tolist()
                out = "".join([str(i) for i in k])
                bf_lst.append(out)
print("The length of bf_lst is {} and the x_positive is {}".format(len(bf_lst), x_positive_size))

with torch.no_grad():
    fp = 0
    total = 0
    for x_val, y_val in negative_keys_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')

        net.eval()

        yhat = net(x_val.float())
        for i in range(y_val.size(0)):
            total += 1
            if yhat[i] < tau:
                continue
            else:
                fp += 1
    print("The false positive rate is {} over {}. {}".format(fp, total, fp/total))
    

bf = BloomFilter(len(bf_lst), 0.01)
for i in bf_lst:
    # Note we add the string of i to the bloom filter
    # REMEMBER TO CHECK THE STRING OF i.
    bf.add(str(i))

print("Current bloom filter hash count is {} and total size {}".format(bf.hash_count, bf.size))

# To find the FPR, we need to sample from the non-known list. "Held-out set of non-keys"
print("--------------- Standard Learned Bloom Filter")
with torch.no_grad():
    false_positive = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')

        net.eval()

        yhat = net(x_val.float())

        for i in range(y_val.size(0)):
            if y_val[i] == 0:
                total += 1
            if yhat[i] > tau:
                # assume I return true
                if y_val[i] == 0:
                   false_positive += 1
            else: # consult the bloom filter
                k = x_val[i].tolist()
                out = "".join([str(i) for i in k])
                answer = bf.check(out)
                if (answer is True and y_val[i] == 0):
                    false_positive += 1
    print("{} false positive out of {} total. {}".format( false_positive, total, false_positive/total ))

# Generate restricted data.

restricted_dataset_0_100000 = generate_data_restricted_range(not_x, bf_lst, 0, 100000)

lst = []
index = 0
restricted_data1 = np.random.choice(restricted_dataset_0_100000, 400)
while index < len(restricted_data1):
    lst.append(conv(restricted_data1[index]))
    index += 1

restricted_data1 = np.array(lst)
y = np.array([0] * len(lst))
restricted_data1 = TensorDataset(torch.from_numpy(restricted_data1).long().to('cpu'), torch.from_numpy(y).long().to('cpu'))
restricted_data1 = DataLoader(dataset=restricted_data1, batch_size = 10)

# Find FPR for a restricted dataset.
with torch.no_grad():
    false_positive = 0
    total = 0
    for x_val, y_val in restricted_data1:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')
        yhat = net(x_val.float())
        for i in range(y_val.size(0)):
            if y_val[i] == 0:
                total += 1
            if yhat[i] > tau:
                if y_val[i] == 0:
                    false_positive +=1
            else:
                k = x_val[i].tolist()
                out = "".join([str(i) for i in k])
                answer = bf.check(out)
                if (answer is True and y_val[i] == 0):
                    false_positive += 1
    print("{} false positive out of {} total. {} for restricted dataset 1".format(false_positive, total, false_positive/total))

# TODO: Add in the sandwiched bloom filter

print("--------------- Sandwiched Learned Bloom Filter")
# The backup filter filters by 8% while inserting bf_lst elements, and the front bloom filter inserts at 38% for 1000 positive_key elements
backup_filter = BloomFilter(len(bf_lst), 0.1)
for i in bf_lst:
    backup_filter.add(str(i))

front_filter_lst = []
for x_val, _ in positive_keys_loader:
    x_val = x_val.to('cpu')

    for i in range(y_val.size(0)):
        k = x_val[i].tolist()
        out = "".join([str(i) for i in k])
        front_filter_lst.append(out)

front_filter = BloomFilter(len(front_filter_lst), 0.43)
for i in front_filter_lst:
    front_filter.add(str(i))

print("Front_filter has size {}, backup_filter has size {}".format(front_filter.size, backup_filter.size))

with torch.no_grad():
    false_positive = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')

        net.eval()

        yhat = net(x_val.float())

        for i in range(y_val.size(0)):
            if y_val[i] == 0:
                total += 1
            k = x_val[i].tolist()
            out = "".join([str(i) for i in k])
            # If it is in the front filter, check it out in the learned function.            
            if front_filter.check(out):
                if yhat[i] > tau:
                    # assume I return true so check for false_positives.
                    # If it returns true when the expected value is false, this is a false positive.
                    if y_val[i] == 0:
                       false_positive += 1
                else: # consult the bloom filter
                    answer = backup_filter.check(out)
                    if (answer is True and y_val[i] == 0):
                        false_positive += 1
    print("{} false positive out of {} total. {}".format( false_positive, total, false_positive/total ))

with torch.no_grad():
    false_positive = 0
    total = 0
    for x_val, y_val in restricted_data1:
        x_val = x_val.to('cpu')
        y_val = y_val.to('cpu')
        yhat = net(x_val.float())
        for i in range(y_val.size(0)):
            if y_val[i] == 0:
                total += 1

            k = x_val[i].tolist()
            out = "".join([str(i) for i in k])
            if front_filter.check(out):
                if yhat[i] > tau:
                    if y_val[i] == 0:
                        false_positive += 1
                else:
                    k = x_val[i].tolist()
                    out = "".join([str(i) for i in k])
                    answer = backup_filter.check(out)
                    if (answer is True and y_val[i] == 0):
                        false_positive += 1
    print("{} false positive out of {} total. {} for restricted dataset 1".format(false_positive, total, false_positive/total))
