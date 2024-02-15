import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import time
from datetime import datetime
from tqdm import tqdm
import math
import os


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--wd', type=float, default=0, help='Weight decay')
parser.add_argument('--mom', type=float, default=0, help='Momentum')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--lrgama', type=float, default=1, help='Learning rate factor of decay')


args = parser.parse_args()



def load_fasta(target, file, label, max_len = -1, location="data/"):
  nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N":-1}
  line_counter = 0

  file = os.path.join(location, file)

  # open the fasta file
  with open(file, "r") as f:
    for line in f:
      if line_counter==0:
        # read the first line, which should contain the header
        header = line.strip()
        chrom, pos = header.split(":")
        chrom = chrom[4:] # remove the ">" symbol
        first_pos, end_pos = map(int, pos.split("-"))
        line_counter = 1- line_counter

      else:
        sequence = ""
        sequence += line.strip()
        # create a dictionary with the sequence and other information
        int_seq = [nuc_map[nuc] for nuc in sequence.upper()]
        one_hot = np.zeros((len(sequence), 4))
        for i, nuc in enumerate(int_seq):
          if nuc != -1:
            one_hot[i, nuc] = 1


        fasta_dict = {
          "chromosome": chrom,
          "first_pos": first_pos,
          "end_pos": end_pos,
          "len": len(sequence),
          "label": label,
          "sequence": sequence,
          "one-hot": one_hot
        }
        # append the dictionary to the target list
        if (max_len == -1) or (max_len >= fasta_dict["len"]):
          target.append(fasta_dict)
        line_counter = 1- line_counter



class FastaDataset(Dataset):
  def __init__(self, files, labels, sample_len):
    # files is a list of fasta file names
    # labels is a list of corresponding label for each file
    assert len(files) == len(labels), "Files and labels must have the same length"

    self.labels = labels
    self. sample_len = sample_len

    self.data = []
    for file, label in zip(files, labels):
      print(label)
      # load each file with the load_fasta function and append to the data list
      load_fasta(self.data, file, label, location="data/")

  def __len__(self):
    # return the length of the data list
    return len(self.data)

  def info(self):
    count = {}
    for d in self.data:
      count[str(d["label"])] = count.get(str(d["label"]), 0) + 1
    for n in count.keys():
       count[n] /= len(self)
       count[n] = np.round(count[n], decimals=4)
    return count

  def __getitem__(self, index):
      # return the one-hot encoded sequence and the label as a tensor
      sample = self.data[index]
      one_hot = torch.tensor(sample["one-hot"], dtype=torch.float)
      label = torch.tensor(sample["label"], dtype=torch.long)
      # get the length of the sequence
      seq_len = one_hot.shape[0]
      # if the length is less than sample_len, pad with zeros
      if seq_len < self.sample_len:
        pad_len = self.sample_len - seq_len
        half_pad_len = math.floor(pad_len/2)
        one_hot = torch.nn.functional.pad(one_hot, (0, 0, half_pad_len, pad_len-half_pad_len))
      # if the length is more than sample_len, randomly truncate
      elif seq_len > self.sample_len:
        start = torch.randint(0, seq_len - self.sample_len + 1, (1,))
        end = start + self.sample_len
        one_hot = one_hot[start:end]

      #accomodate shapes and types
      one_hot = one_hot.unsqueeze(0)
      label = label.float()

      return one_hot, label

def to_labels(tensor):
    binary_tensor = (tensor > 0.5).int()

    mapping = {
        (0, 0, 0): 1,
        (0, 0, 1): 2,
        (0, 1, 0): 3,
        (1, 0, 0): 4,
        (0, 1, 1): 5,
        (1, 0, 1): 6,
        (1, 1, 1): 7
    }

    # Initialize an output tensor with the same shape as the binary tensor
    output_tensor = torch.zeros(binary_tensor.shape[0],1)

    # Iterate over the binary tensor and apply the mapping
    for i in range(binary_tensor.shape[0]):
        pattern = tuple(binary_tensor[i].tolist())
        output_tensor[i] = mapping.get(pattern, 0)  # Default to 0 if pattern is not found

    return output_tensor


random.seed(0)


##Data Loading

batch_size = args.batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("running on", device)


#dataset = FastaDataset(["A.fasta", "B.fasta", "D.fasta", "AB.fasta", "AD.fasta", "BD.fasta", "ABD.fasta"],[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], 249)
#dataset = FastaDataset(["A.fasta", "B.fasta", "D.fasta", "AB.fasta", "AD.fasta", "BD.fasta", "ABD.fasta"],[[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], 249)
#dataset = FastaDataset(["A.fasta", "D.fasta", "AB.fasta", "AD.fasta", "BD.fasta"],[[1,0,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]], 249)
dataset = FastaDataset(["A.fasta", "D.fasta", "AB.fasta", "AD.fasta", "BD.fasta"],[[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0]], 249)

from sklearn.model_selection import train_test_split
train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("\n"*2+"-"*10)
for key, value in dataset.info().items():
    print(f"{key}: {value}")
print("-"*10)




##Model Design

class DeepSTARR(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepSTARR, self).__init__()

        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(7,4), padding=(3,0))
        self.relu1 = nn.LeakyReLU(0.00001)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=60*8, kernel_size=(3,1), padding=(1,0))
        self.relu2 = nn.LeakyReLU(0.00001)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=60*8, out_channels=60*8, kernel_size=(5,1), padding=(2,0))
        self.relu3 = nn.LeakyReLU(0.00001)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=60*3, out_channels=120, kernel_size=(3,1), padding=(1,0))
        self.relu4 = nn.LeakyReLU(0.00001)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(1800*0+7440*2, 256*8)  # 120 channels * the input dims :/
        self.relu5 = nn.LeakyReLU()
        self.fc2 = nn.Linear(256*8, 256)
        self.relu6 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        #print(x.shape)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        #print(x.shape)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        #print(x.shape)

        # Fourth convolutional layer
        #x = self.conv4(x)
        #x = self.relu4(x)
        #x = self.pool4(x)
        #print(x.shape)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x
    
def validate(model, val_dataloader):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_x, val_trg in val_dataloader:
            val_x = val_x.to(device)
            val_trg = val_trg.to(device)

            val_outputs = model(val_x)

            predicted_classes = torch.argmax(val_outputs, dim=1)
            real_classes = torch.argmax(val_trg, dim=1)

            # Compute the loss
            val_loss += criterion(val_outputs, val_trg).item()

            # Calculate accuracy
            val_correct += torch.sum(predicted_classes == real_classes).item()
            val_total += len(predicted_classes)

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return avg_val_loss


##Training

model = DeepSTARR(num_classes=7)
#model = torch.load("pred_saved (7).pt")
model = model.to(device)



training_accs = []
val_accs = []


def train_model(num_epochs):
    best_accuracy = -1
    global training_accs
    global val_accs
    losses = []
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        model.train()
        total_correct = 0
        total = 0

        model.train()
        # Generate the dataset for this epoch
        for  x, trg in train_dataloader:
          x = x.to(device)
          trg = trg.to(device)

          outputs = model(x)

          # Compute the loss
          loss = criterion(outputs, trg)

          # Zero the gradients, backward pass, and update the weights
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()



          # Calculate accuracy
          predicted_classes0 = torch.argmax(outputs, dim=1)
          real_classes0 = torch.argmax(trg, dim=1)
          #total_correct += sum(sum((trg==(outputs>0.5)).T)==model.num_classes)
          total_correct += torch.sum(predicted_classes0 == real_classes0).item()
          total += len(trg)

          # Print statistics
          total_loss += loss.item()
        accuracy = total_correct/total
        scheduler.step()


        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_x, val_trg in val_dataloader:
                val_x = val_x.to(device)
                val_trg = val_trg.to(device)

                val_outputs = model(val_x)

                # Compute the loss
                val_loss += criterion(val_outputs, val_trg).item()

                # Calculate accuracy
                predicted_classes = torch.argmax(val_outputs, dim=1)
                real_classes = torch.argmax(val_trg, dim=1)

                #val_correct += sum(sum((val_trg == (val_outputs > 0.5)).T) == model.num_classes)
                val_correct += torch.sum(predicted_classes == real_classes).item()
                val_total += len(val_trg)

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        training_accs.append(accuracy)
        val_accs.append(val_accuracy)
        losses.append(avg_val_loss)

        if val_accuracy >= best_accuracy:
          if epoch>num_epochs/10:
            best_accuracy = val_accuracy
            best_model = model.state_dict()


    model.load_state_dict(best_model)
    print(f"Best model saved with accuracy: {best_accuracy:.4f}")
    return model, best_accuracy, [training_accs, val_accs, losses]


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.mom, weight_decay = args.wd) #lr = 5e-4 com o do bernardo teve 30% em 6 ou 7 epocas e depois desceu pra 9%. very nice
scheduler = ExponentialLR(optimizer, args.lrgama)

model = model.to(device)
validate(model, val_dataloader)

model, acc, history = train_model(num_epochs=args.epochs)

#saving the model
torch.save(model,"saved_models/pred_saved_"+str(datetime.now().strftime("%Y%m%d_%H%M"))+".pt")



#printing the graphs
history = [training_accs, val_accs, training_accs]

#convert list into np array and send everything to cpu
history_np = np.zeros((3,len(history[0])))

for data_n, data in enumerate(history):
  for n, d in enumerate(data):
    if type(d) == float:
      history_np[data_n, n] = d
    elif type(d) == torch.Tensor:
      history_np[data_n, n] = d.to("cpu")
    else:
      print("sus type")


#plot graphs
plt.plot(history_np[0])
plt.plot(history_np[1])
plt.title("Training and Val Acc \n lr=" + str(args.lr)+ ", momentum =" + str(args.mom)+ ", weight_decay =" + str(args.wd)+ ", lr_gama =" + str(args.lrgama))
plt.savefig("saved_models/graphs/acc_"+str(datetime.now().strftime("%Y%m%d_%H%M"))+".png")

plt.plot(history_np[2])
plt.title("Acc Loss")
#plt.show()
