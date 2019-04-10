import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM_CRF, prepare_sequence

def train(training_data, dev_data, embedding_dim, hidden_dim, no_epochs):
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim, START_TAG, STOP_TAG)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    avg_losses = []    

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(no_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        print("Epoch - " + str(epoch))
        losses = [] 
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            losses.append(int(loss))
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        # Check predictions after training
        with torch.no_grad():
            precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
            print(model(precheck_sent))
        avg_losses.append(np.mean(losses))
        # We got it!
