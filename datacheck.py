import numpy as np
import codecs
import csv

path_train = "../dataset/train.tsv"
path_test = "../dataset/test.tsv"

sentences = []
sentence = []
labels = []
label = []
tag_set = set()

for line in codecs.open(path_test, 'r', 'utf-8'):
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

len_count = {}
for sentence in sentences:
    if len(sentence) in len_count.keys():
        len_count[len(sentence)] += 1
    else:
        len_count[len(sentence)] = 1

print(len_count)