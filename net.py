import numpy as np
import codecs
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from TorchCRF import CRF
from torch import nn
from keras.preprocessing.sequence import pad_sequences

from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig
from transformers import BertForTokenClassification, RobertaForTokenClassification, XLNetForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score, classification_report
import os
from tqdm import trange
import argparse
import logging

class bertCRF(nn.Module):
    def __init__(self, num_classes, model_name) -> None:
        super(bertCRF, self).__init__()

        if model_name == "bert-base-cased-crf":
            self.bert = BertModel(BertConfig())
        if model_name == "roberta-base-crf":
            self.bert = RobertaModel(RobertaConfig())

        self.dropout = nn.Dropout(0.1)
        self.position_wise_ff = nn.Linear(768, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=None,
                            attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)
        

        log_likelihood, sequence_of_tags = self.crf(emissions, labels,mask=attention_mask), self.crf.viterbi_decode(emissions, attention_mask)
        return -log_likelihood, sequence_of_tags