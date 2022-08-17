import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random
import numpy as np
from transformers import BertModel
import pdb
from tqdm import tqdm
from abc import ABC, abstractmethod


class Autoencoder_RNN(nn.Module):
    def __init__(self, max_seq_len, batch_size, hidden_size, enc_out_size,
                 vocab_size, embed_size=300, pad_idx=0, local=False,
                 device='cpu'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.enc_out_size = enc_out_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pad_idx = pad_idx

        self.local = local
        self.device = device

    def forward(self):
        raise NotImplementedError


class ADePT(Autoencoder_RNN):
    def __init__(self, pretrained_embeddings, *args, private=False,
                 dp_mechanism='laplace', clipping_constant=1, epsilon=1,
                 delta=1e-5, norm_ord=1, no_clipping=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(pretrained_embeddings, self.embed_size, self.hidden_size,
                               self.enc_out_size, self.vocab_size, batch_size=self.batch_size, num_layers=1, dropout=0.0, pad_idx=self.pad_idx)
        self.decoder = Decoder(pretrained_embeddings, self.embed_size, self.hidden_size,
                               self.enc_out_size, self.vocab_size, device=self.device,
                               batch_size=self.batch_size, num_layers=1, dropout=0.0, pad_idx=self.pad_idx)

        self.private = private
        self.clipping_constant = clipping_constant
        self.epsilon = epsilon
        self.delta = delta
        self.norm_ord = norm_ord
        self.no_clipping = no_clipping
        self.dp_mechanism = dp_mechanism

    def forward(self, inputs, lengths, teacher_forcing_ratio=0.5):
        '''
        Input:
            inputs: batch_size X seq_len
            lengths: batch_size
        Return:
            z: batch_size X seq_len X vocab_size+4
            enc_embed: batch_size X seq_len X embed_size
        '''
        encoder_hidden = self.encoder.initHidden()
        encoder_hidden = (encoder_hidden[0].to(self.device),
                          encoder_hidden[1].to(self.device))

        encoder_out, encoder_hidden = self.encoder(inputs, lengths,
                                                   encoder_hidden)

        ## Privacy module
        context_vec = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=2)
        if self.private:
            context_vec = self.privatize(context_vec)
        else:
            if not self.no_clipping:
                # ADePT model clips without privacy to encourage model
                # representations to be within a radius of C
                context_vec = self.clip(context_vec)
        encoder_hidden = (context_vec[:, :, :self.hidden_size].contiguous(),
                          context_vec[:, :, self.hidden_size:].contiguous())

        target = deepcopy(inputs)
        z = torch.zeros(self.max_seq_len, inputs.shape[0], self.vocab_size)
        z = z.to(self.device)
        prev_x = target[:, 0]
        prev_x = prev_x.unsqueeze(0)

        decoder_hidden = encoder_hidden

        teacher_force = random.random() < teacher_forcing_ratio
        if teacher_force:
            for i in range(1, self.max_seq_len):
                z_i, decoder_hidden = self.decoder(prev_x, decoder_hidden)
                prev_x = target[:, i].unsqueeze(dim=0)
                z[i] = z_i
        else:
            for i in range(1, self.max_seq_len):
                z_i, decoder_hidden = self.decoder(prev_x, decoder_hidden)
                top_idxs = z_i.argmax(1)
                prev_x = top_idxs.unsqueeze(dim=0)
                z[i] = z_i
        z = z.transpose(1, 0)[:, 1:, :].reshape(-1, z.shape[2])

        return z

    def privatize(self, context_vec):
        clipped = self.clip(context_vec)
        noisified = self.noisify(clipped, context_vec)
        return noisified

    def clip(self, context_vec):
        norm = torch.linalg.norm(context_vec, axis=2, ord=self.norm_ord)
        ones = torch.ones(norm.shape[1]).to(self.device)
        min_val = torch.minimum(ones, self.clipping_constant / norm)
        clipped = min_val.unsqueeze(-1) * context_vec
        return clipped

    def noisify(self, clipped_tensor, context_vec):
        if self.norm_ord == 1:
            sensitivity = torch.tensor(2 * self.clipping_constant)
        elif self.norm_ord == 2:
            sensitivity = 2 * self.clipping_constant * torch.sqrt(
                torch.tensor(clipped_tensor.shape[2]))
        else:
            raise Exception("Sensitivity calculation not implemented for norm orders other than 1 or 2.")
        if self.dp_mechanism == 'laplace':
            laplace = torch.distributions.Laplace(0, sensitivity / self.epsilon)
            noise = laplace.sample(
                sample_shape=torch.Size((context_vec.shape[1],
                                         context_vec.shape[2]))).unsqueeze(0)
        elif self.dp_mechanism == 'gaussian':
            scale = torch.sqrt((sensitivity**2 / self.epsilon**2) * 2 * torch.log(torch.tensor(1.25 / self.delta)))
            gauss = torch.distributions.normal.Normal(0, scale)
            noise = gauss.sample(sample_shape=torch.Size((clipped_tensor.shape[1], clipped_tensor.shape[2])))
        else:
            raise Exception(f"No DP mechanism available called '{self.dp_mechanism}'.")
        noise = noise.to(self.device)
        noisified = clipped_tensor + noise
        return noisified


class Encoder(nn.Module):
    def __init__(self, embeds, embed_size, hidden_size, enc_out_size,
                 vocab_size, num_layers=2, batch_size=32, dropout=0.,
                 pad_idx=0):
        super(Encoder, self).__init__()
        if embeds is not None:
            self.pretrained_embeds = True
            self.embeds = nn.Embedding.from_pretrained(
                    torch.from_numpy(embeds).float(),
                    padding_idx=pad_idx)
        else:
            self.pretrained_embeds = False
            self.embeds = nn.Embedding(vocab_size, embed_size,
                                       padding_idx=pad_idx)

        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=False,
                           dropout=dropout)

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, x, lengths, hidden_tup):
        """
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        if self.pretrained_embeds:
            with torch.no_grad():
                x = self.embeds(x)
        else:
            x = self.embeds(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        output, (hidden, cell) = self.rnn(packed, hidden_tup)
        output, _ = pad_packed_sequence(output)
        return output, (hidden, cell)

    def initHidden(self):
        # 1 X batch_size X hidden_size
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        cell = torch.zeros(1, self.batch_size, self.hidden_size)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, embeds, embed_size, hidden_size, enc_out_size,
                 vocab_size, device='cpu', batch_size=32, num_layers=2,
                 dropout=0.0, pad_idx=0):
        super(Decoder, self).__init__()
        if embeds is not None:
            self.pretrained_embeds = True
            self.embeds = nn.Embedding.from_pretrained(
                    torch.from_numpy(embeds).float(),
                    padding_idx=pad_idx)
        else:
            self.pretrained_embeds = False
            self.embeds = nn.Embedding(vocab_size, embed_size,
                                       padding_idx=pad_idx)

        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=False, bidirectional=False,
                           dropout=dropout)

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = device

        self.generator = nn.Linear(hidden_size, vocab_size)

    def forward(self, prev_x, hidden_tup):

        # update rnn hidden state
        if self.pretrained_embeds:
            with torch.no_grad():
                prev_embed = self.embeds(prev_x)
        else:
            prev_embed = F.relu(self.embeds(prev_x))
        output, (hidden, cell) = self.rnn(prev_embed, hidden_tup)
        z_i = self.generator(output[0])

        return z_i, (hidden, cell)
