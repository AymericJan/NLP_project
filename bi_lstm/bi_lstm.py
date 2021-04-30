"""
Importations
"""

import os
import argparse
import time
import json
import random
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy import data

import spacy

"""
Functions
"""


def json_to_config(path):
    """
    Loads the configuration of the network from a json file
    input:
        path (string): path to the json file
    output:
        dict: dict containing the parameters from the json file
    """

    dict_config = json.load(open(path))
    return dict_config

def get_token_from_sentences(text):
    """
    Transform comments into tokens, removing some characters
    input:
        text (string): comment to tokenize
    output:
        list: list of tokens
    """

    special_char="""!"'#$%|}&(,-.@[\\]^_`{/:;<)*+=~>?\t\n"""
    filtered = "".join([char if char not in special_char else "" for char in text])
    return [token.text for token in spacy_load.tokenizer(filtered) if not token.is_space]

def prepare_data(path, text_length, embedding_name = "glove.6B.100d", batch_size= 64, token_fn = get_token_from_sentences, split_ratio = [0.7,0.2]):
    """
    Prepare the data by providing iterators
    inputs:
        path (string): path to the train.csv file
        text_length (int): length of the text
        embedding_name (string): name of the choosen pre-trained embedding
        batch_size (int): size of the batch
        token_fn (function): function to transform comments into tokens
        split_ratio (list[float]): train/val/test split ratios
    outputs:
        train_iterator (BucketIterator): training iterator
        valid_iterator (BucketIterator): validation iterator
        test_iterator (BucketIterator): test iterator
        TEXT (Field): 
        LABEL (Field): 
        PAD_IDX (int): padding length 
    """
    TEXT = data.Field(lower=True, batch_first=True, fix_length=text_length, preprocessing=None, tokenize=token_fn)
    LABEL = data.Field(sequential=False,is_target=True, use_vocab=False, pad_token=None, unk_token=None)

    datafields = [('id', None),
                ('comment_text', TEXT), 
                ("toxic", LABEL), 
                ("severe_toxic", LABEL),
                ('obscene', LABEL), 
                ('threat', LABEL),
                ('insult', LABEL),
                ('identity_hate', LABEL)]


    dataset = data.TabularDataset(
        path=path,
        format='csv',
        skip_header=True,
        fields=datafields)

    train, val, test = dataset.split(split_ratio=split_ratio, random_state=random.getstate())

    TEXT.build_vocab(train, vectors=embedding_name)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train, val, test),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            sort_key=lambda x: len(x.comment_text))

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL, PAD_IDX


class biLSTMnet(nn.Module):
    """
    Neural network architecture for the LSTM model
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embedding_vectors, text_length, lstm_hidden_size, nb_layer_lstm, dropout, bidirectional):
        """
        Init function
        inputs:
            vocab_size (int): size of the vocabulary for the embedding
            embedding_dim (int): dimension of the embedding space
            output_dim (int): output dimension i.e. number of category to predict
            pad_idx (int): padding
            embedding_vectors (NumpyArray): Embedding vectors
            text_length (int): length of the text
            lstm_hidden_size (int): size of the hidden layer of the LSTM
            nb_layer_lstm (int): number of layers in the LSTM
            dropout (float): dropout probability
            bidirectional (bool): LSTM or Bidirectional LSTM
        """
        super().__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(embedding_vectors, freeze=True, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers = nb_layer_lstm, batch_first=True, dropout = dropout, bidirectional = bidirectional)
        self.max_pool = nn.MaxPool2d((text_length,1))
        if bidirectional:
            self.fc1 = nn.Linear(lstm_hidden_size*2, 64)
        else:
            self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, text):
        """
        forward method of the neural network
        """
        x = self.embeddings(text)
        x = self.lstm(x)[0]
        x = self.max_pool(x).squeeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train_epoch(model, training_iterator, optimizer, loss_fn, batch_size = 64, device = "cpu"):
    """
    Training on an epoch
    input:
        model(nn.Module): model to train
        training_iterator (BucketIterator): training iterator
        optimizer (torch optim): optimizer function
        loss_fn (): loss function
        batch_size (int): size of the batch
        device (device): "cpu" or "cuda:0"
    
    output:
        mean_loss (torch Tensor)
        mean_accuracy (torch Tensor)
    """
    n_batch = int(np.ceil(len(training_iterator.dataset)/batch_size))
    total_loss_epoch = torch.zeros(n_batch)
    total_accuracy = torch.zeros(n_batch)
    
    for i, batch in enumerate(training_iterator):
        x, y = batch.comment_text, torch.stack([batch.toxic, batch.severe_toxic, batch.obscene, batch.threat, batch.insult, batch.identity_hate], dim=1).float()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        total_loss_epoch[i] = loss.item()

        total_accuracy[i] = get_accuracy(output, y)
        loss.backward()
        optimizer.step()
    return torch.mean(total_loss_epoch), torch.mean(total_accuracy)

def validation_epoch(model, validation_iterator, loss_fn, batch_size = 64, device = "cpu"):
    """
    Validation on an epoch
    input:
        model(nn.Module): model to train
        validation_iterator (BucketIterator): validation iterator
        loss_fn (): loss function
        batch_size (int): size of the batch
        device (device): "cpu" or "cuda:0"
    
    output:
        mean_loss (torch Tensor)
        mean_accuracy (torch Tensor)
    """
    n_batch = int(np.ceil(len(validation_iterator.dataset)/batch_size))
    total_loss_epoch = torch.zeros(n_batch)
    total_accuracy = torch.zeros(n_batch)
    
    with torch.no_grad():
        for i, batch in enumerate(validation_iterator):
            x, y = batch.comment_text, torch.stack([batch.toxic, batch.severe_toxic, batch.obscene, batch.threat, batch.insult, batch.identity_hate], dim=1).float()
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss_epoch[i] = loss.item()
            total_accuracy[i] = get_accuracy(output, y)

    return torch.mean(total_loss_epoch), torch.mean(total_accuracy)

def test_epoch(model, test_iterator, loss_fn, batch_size = 64, device = "cpu"):
    """
    Test on an epoch
    input:
        model(nn.Module): model to train
        test_iterator (BucketIterator): test iterator
        loss_fn (): loss function
        batch_size (int): size of the batch
        device (device): "cpu" or "cuda:0"
    
    output:
        mean accuracy by category
    """
    n_batch = int(np.ceil(len(test_iterator.dataset)/batch_size))
    total_accuracy = torch.zeros((n_batch,6))

    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            x, y = batch.comment_text, torch.stack([batch.toxic, batch.severe_toxic, batch.obscene, batch.threat, batch.insult, batch.identity_hate], dim=1).float()
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_accuracy[i,:] = get_accuracy(output, y, by_cat=True)
    return torch.mean(total_accuracy, axis = 0)


def get_accuracy(output, y, by_cat = False):
    """
    Computes the accuracy given the output of the network and the groundtruth
    inputs:
        output (torch Tensor): output of the network
        y (torch Tensor): ground truth label
        by_cat (bool): False for overall accuracy, True for accuracy by category

    output:
        accuracy (torch Tensor)
    """
    if by_cat:
        return torch.mean(((torch.sigmoid(output) > 0.5) == (y > 0.5)).float(), axis = 0)
    else:
        return torch.mean(((torch.sigmoid(output) > 0.5) == (y > 0.5)).float())


"""
Main program for the training
"""

if __name__ == "__main__":

    ##Loading the arguments from the config json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest = "config", default = "config/config.json")
    args = parser.parse_args()
    dict_args = json_to_config(args.config)
    

    # hyperparams
    TEXT_LENGTH = 100
    EMBEDDING_SIZE = 100
    BATCH_SIZE = 64
    VOCAB_SIZE=20000
    PATH = dict_args["path_train"]
    PATH_TEST = dict_args["path_test"]
    n_epochs = int(dict_args["n_epochs"])
    name = dict_args["name"]
    bi_lstm = bool(dict_args["bi_lstm"])
    nb_layer_lstm = int(dict_args["nb_layer_lstm"])
    dropout = float(dict_args["dropout"])
    size_hidden_layer = int(dict_args["size_hidden_layer"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIM = 6

    spacy_load = spacy.load("en_core_web_sm")

    train_iterator, valid_iterator, test_iterator, TEXT, LABEL, PAD_IDX = prepare_data(path = PATH, text_length = TEXT_LENGTH,
                                                                        embedding_name = "glove.6B.100d", batch_size= 64,
                                                                        token_fn = get_token_from_sentences, split_ratio = [0.7,0.2, 0.1])
    

    model = biLSTMnet(len(TEXT.vocab), embedding_dim=EMBEDDING_SIZE,
                        output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, embedding_vectors=TEXT.vocab.vectors,
                        text_length=TEXT_LENGTH, lstm_hidden_size= size_hidden_layer, nb_layer_lstm = nb_layer_lstm,
                        dropout = dropout, bidirectional= bi_lstm).to(device)
    
    PATH_MODEL = "{}.pth".format(name)

    df = pd.read_csv(PATH)
    w = torch.Tensor(1/np.array(df[['toxic', 'severe_toxic', 'obscene', 'threat',
    'insult', 'identity_hate']].sum(axis = 0))).to(device)
    loss_fn = nn.BCEWithLogitsLoss(weight = w)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    df = pd.DataFrame(columns = ["epoch", "loss_train", "loss_val","accuracy_train", "accuracy_val"])

    print("device = {}".format(device))

    training_loss = torch.zeros(n_epochs)
    training_accuracy = torch.zeros(n_epochs)
    validation_loss = torch.zeros(n_epochs)
    validation_accuracy = torch.zeros(n_epochs)

    for epoch in tqdm.tqdm(range(n_epochs)):
        print("Starting epoch {} out of {}".format(epoch, n_epochs - 1))
        loss_epoch, accuracy_epoch = train_epoch(model = model, training_iterator = train_iterator, optimizer = optimizer, loss_fn = loss_fn, device = device)
        loss_val, accuracy_val = validation_epoch(model, valid_iterator, loss_fn, batch_size = 64, device = device)
        training_loss[epoch] = loss_epoch
        training_accuracy[epoch] = accuracy_epoch
        validation_loss[epoch] = loss_val
        validation_accuracy[epoch] = accuracy_val
        print("Training/Validation: Loss = {:.6f} vs {:.6f}, accuracy = {:.6f} vs {:.6f}".format(loss_epoch, loss_val, accuracy_epoch, accuracy_val))

        df = df.append({"epoch":epoch, "loss_train":loss_epoch, "loss_val":loss_val,"accuracy_train":accuracy_epoch, "accuracy_val":accuracy_val}, ignore_index = True)

    df.to_csv("{}.csv".format(name))
    torch.save(model.state_dict(), PATH_MODEL)
    
    test_accuracy = test_epoch(model, test_iterator, loss_fn, batch_size = 64, device = device)
    print('toxic, severe_toxic, obscene, threat, insult, identity_hate')
    print(test_accuracy)