import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import csv


def train_model(train_dataloader, model, f_loss, optimizer, device):
    model.train()
    N = 0
    tot_loss = 0.0
    correct = 0.0
    score = 0.0
    y_pred = []
    y_true = []
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for (inputs, targets) in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            N += inputs.shape[0]
            loss = f_loss(outputs, targets)
            tot_loss += inputs.shape[0] *loss.item()
            predicted_targets = outputs.argmax(dim=1)
            y_pred.append(predicted_targets.cpu())
            y_true.append(targets.cpu())
            correct += (predicted_targets == targets).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=tot_loss/N, acc=correct/N)
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        f1 = f1_score(y_pred, y_true, average='macro')
        return tot_loss/N, correct/N, f1

def eval_model(eval_dataloader, model, f_loss, device):
    with torch.no_grad():
        model.eval()
        N=0
        tot_loss = 0.0
        correct = 0.0
        y_pred = []
        y_true = []
        with tqdm(eval_dataloader, unit="batch") as tepoch:
            for (inputs, targets) in tepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                N += inputs.shape[0]
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
                predicted_targets = outputs.argmax(dim=1)
                y_pred.append(predicted_targets.cpu())
                y_true.append(targets.cpu())
                correct += (predicted_targets == targets).sum().item()

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            f1 = f1_score(y_pred, y_true, average='macro')
            return tot_loss/N, correct/N, f1

def scores_f1(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return f1_macro, acc

def test_model(test_dataloader, model, device):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        cmpt = 0
        N = 0
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                N += inputs.shape[0]
                predictions = outputs.argmax(dim=1)
                y_true.append(targets.cpu().numpy())
                y_pred.append(predictions.cpu().numpy())
                if cmpt == 667: # TODO check why errors
                    break
                cmpt += 1
        f1_macro, acc = scores_f1(y_true, y_pred)
        return f1_macro, acc

def load_model(model,model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def make_prediction(model, dataloader, device):
    with torch.no_grad():
        model.eval()
        predictions_dict = {"imgname" : [], "label" : []}
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, names in tepoch:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                for i in range(len(names)):
                    predictions_dict["imgname"].append(names[i])
                    predictions_dict["label"].append(predictions[i].item())
        
        with open("predictions.csv", "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(predictions_dict.keys())
            writer.writerows(zip(*predictions_dict.values()))


    