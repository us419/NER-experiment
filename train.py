import numpy as np
import codecs
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForTokenClassification, RobertaForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score, classification_report
import os
from tqdm import trange
import argparse
import logging
from net import bertCRF

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_number", type=str, default='7')
parser.add_argument("--pretrained_model", type=str, default='roberta-base')
parser.add_argument("--lr", type=int, default=3e-5)
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--output_dir", type=str, default='output')
parser.add_argument("--save_dir", type=str, default='models')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

path_train = "../dataset/train.tsv"
path_test = "../dataset/test.tsv"
MAX_LEN = 100
bs = args.batchsize
device = torch.device("cuda")

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def prepare_input(data_path, MAX_LEN):
    sentences = []
    sentence = []
    labels = []
    label = []
    tag_set = set()

    for line in codecs.open(data_path, 'r', 'utf-8'):
        word = line.strip().split('\t')
        if len(word) > 1:
            sentence.append(word[0])
            label.append(word[-1])
            tag_set.add(word[-1])
        else:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
    if args.pretrained_model == 'bert-base-cased' or args.pretrained_model == 'bert-base-cased-crf':
        print("use BertTokenizer")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    if args.pretrained_model == 'roberta-base' or args.pretrained_model == 'roberta-base-crf':
        print("use RobertaTokenizer")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)            
    if args.pretrained_model == 'xlnet-base-cased':
        print("use XLNetTokenizer")
        tokenizer = XLNetTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False)

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, tokenizer)
        for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    tag_list = list(tag_set)
    tag_list.append("PAD")
    print(tag_list)
    tag2idx = {t: i for i, t in enumerate(tag_list)}

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    input_ids = torch.tensor(input_ids)
    tags = torch.tensor(tags)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks, tags, tag_list

train_inputs, train_masks, train_tags, tag_list = prepare_input(path_train, MAX_LEN)
test_inputs, test_masks, test_tags, _ = prepare_input(path_test, MAX_LEN)
tag2idx = {t: i for i, t in enumerate(tag_list)}

train_data = TensorDataset(train_inputs, train_masks, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

if args.pretrained_model == 'bert-base-cased':
    model = BertForTokenClassification.from_pretrained(
        args.pretrained_model,
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
if args.pretrained_model == 'bert-base-cased-crf':
    model = bertCRF(num_classes=len(tag2idx), model_name=args.pretrained_model)
if args.pretrained_model == 'roberta-base':
    model = RobertaForTokenClassification.from_pretrained(
        args.pretrained_model,
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
if args.pretrained_model == 'roberta-base-crf':
    model = bertCRF(num_classes=len(tag2idx), model_name=args.pretrained_model)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=args.lr,
    eps=1e-8
)

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        loss = loss.mean()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    best_f1 = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        if args.pretrained_model[-3:] == 'crf':
            logits = np.array(outputs[1])
            predictions.extend(logits)
            label_ids = b_labels.to('cpu').numpy()
            true_labels.extend(label_ids)
        else:
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

    eval_loss = eval_loss / len(test_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))

    if args.pretrained_model[-3:] == 'crf':
        predictions = np.array(predictions)
        pred_tags = [tag_list[p_i] for p in predictions for p_i in p]
    else:
        pred_tags = [tag_list[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if tag_list[l_i] != "PAD"]

    valid_tags = [tag_list[l_i] for l in true_labels
                                  for l_i in l if tag_list[l_i] != "PAD"]

    acc = accuracy_score(pred_tags, valid_tags)
    f1 = f1_score(pred_tags, valid_tags)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'models/' + args.pretrained_model + '_best.pt')
        report = classification_report(valid_tags, pred_tags,digits=4)
        output_eval_file = os.path.join(args.output_dir, args.pretrained_model+"_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Eval results *****\n")
            writer.write(report)
    logger.info("\n%s", report)
    print("Validation Accuracy: {}".format(acc))
    print("Validation F1-Score: {}".format(f1))
    print()