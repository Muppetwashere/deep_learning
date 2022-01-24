import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import * 
from utils import *
from logs import * 

# Path to train and test data
DATA_TRAIN_PATH = '/mounts/Datasets1/ChallengeDeep/train'
DATA_TEST_PATH = '/mounts/Datasets1/ChallengeDeep/test/imgs'

# Hyperparams
num_epochs = 10
num_workers = 8
batch_size = 32
img_size = 224
num_class = 86
lr = 1e-3
top_logdir = "./logs"

# ratio dataset split
train_ratio = 0.8
eval_ratio = 0.15
test_ratio = 0.05

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
plankton_dataset = PlanktonDataset(DATA_TRAIN_PATH, final_size = img_size)

# Split dataset
nb_train = int(train_ratio * len(plankton_dataset))+1
nb_eval = int(eval_ratio * len(plankton_dataset))
nb_test = int(test_ratio * len(plankton_dataset))
train_dataset, eval_dataset, test_dataset = torch.utils.data.dataset.random_split(plankton_dataset, (nb_train, nb_eval, nb_test))

# DataLoaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
test_dataloader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print("The train set contains {} images, in {} batches".format(len(train_dataloader.dataset), len(train_dataloader)))
print("The evaluation set contains {} images, in {} batches".format(len(eval_dataloader.dataset), len(eval_dataloader)))
print("The test set contains {} images, in {} batches".format(len(test_dataloader.dataset), len(test_dataloader)))

# Model
model_name = "DenseNet169_im_224"
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
# Model last layer update
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_class)
model.to(device)

# loss function and optimizer
f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# log params
logdir = generate_unique_logpath(top_logdir, model_name)
model_checkpoint = ModelCheckpoint(logdir + "/best_model.pt", model)
tensorboard_writer = SummaryWriter(log_dir=logdir)

for epoch in range(1,num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs} : ")
    train_loss, train_acc, train_f1 = train_model(train_dataloader, model, f_loss, optimizer, device)
    print(f"Train loss : {train_loss}, Acc : {train_acc}, F1-score : {train_f1}")

    val_loss, val_acc, val_f1 = eval_model(eval_dataloader, model, f_loss, device)
    scheduler.step(val_loss)
    print(f"Validation loss : {val_loss}, Acc : {val_acc}, F1-score {val_f1}")
    
    tensorboard_writer.add_scalar('metrics/train_loss', train_loss, epoch)
    tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, epoch)
    tensorboard_writer.add_scalar('metrics/train_f1',  train_f1, epoch)
    tensorboard_writer.add_scalar('metrics/val_loss', val_loss, epoch)
    tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, epoch)
    tensorboard_writer.add_scalar('metrics/val_f1',  val_f1, epoch)
    model_checkpoint.update(val_loss)

print("Testing model : ")
test_f1, test_acc = test_model(test_dataloader,model, device)
tensorboard_writer.add_scalar('metrics/test_f1', test_f1, epoch)
tensorboard_writer.add_scalar('metrics/test_acc', test_acc, epoch)

# Predictions
""" print("Loading model : ")
model_path = './logs/DenseNet169_1' + "/best_model.pt"
model_ = load_model(model, model_path)

test_chall_dataset = PlanktonTestDataset(DATA_TEST_PATH, img_size)
test_chall_dataloader = DataLoader(dataset=test_chall_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
make_prediction(model_,test_chall_dataloader, device) """
