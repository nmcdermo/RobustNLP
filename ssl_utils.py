from transformers import BertForMaskedLM, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.spatial.distance import cosine
from string import punctuation
from copy import deepcopy
import numpy as np
import nltk
import torch
import sys
import os

def get_adversarial_attacks():

    textfooler_folder = "TextFooler"
    adv_results_folder = "adv_results"
    adv_results_file = "adversaries.txt"
    adv_path = os.path.join(os.pardir, textfooler_folder ,adv_results_folder, adv_results_file)
    adv_sentences = []
    orig_sentences = []
    adv_labels = []
    orig_labels = []
    with open(adv_path) as adv_file:
        for line in adv_file:
            line.strip()
            if ":" in line:
                prefix, sentence = line.split(":")
                sentence = sentence.strip()#.split(" ")
                if "orig" in prefix:
                    orig_sentences.append(sentence)
                    if "0" in prefix:
                        orig_labels.append(0)
                    else:
                        orig_labels.append(1)
                else:
                    adv_sentences.append(sentence)
                    if "0" in prefix:
                        adv_labels.append(0)
                    else:
                        adv_labels.append(1)
    return adv_sentences, orig_sentences, adv_labels, orig_labels

def is_punc(word):
    return word in punctuation

class SentenceEncoder():
    def __init__(self, sentence_dir):
        self.sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_dir)
        self.sentence_model = AutoModel.from_pretrained(sentence_dir)

    def similarity(self, s1, s2):
        text = [s1, s2]
        inputs = self.sentence_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            seq_opt = self.sentence_model(**inputs)[0]

        embeddings = torch.sum(seq_opt * inputs["attention_mask"].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

        semantic_sim = 1 - cosine(embeddings[0],embeddings[1])

        return semantic_sim

class WordEncoder():
    def __init__(self, word_dir, create_new=False):
        self.idx2word = {}
        self.word2idx = {}
        with open(word_dir + "/counter-fitted-vectors.txt", 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in self.idx2word:
                    self.idx2word[len(self.idx2word)] = word
                    self.word2idx[word] = len(self.idx2word) - 1

        if not create_new:
            # load pre-computed cosine similarity matrix if provided
            print('Load pre-computed cosine similarity matrix') #from {}'.format(args.counter_fitting_cos_sim_path))
            cos_sim = np.load(word_dir + "/cos_sim_counter_fitting.npy")
            self.cos_sim = cos_sim
        else:
            # calculate the cosine similarity matrix
            print('Start computing the cosine similarity matrix!')
            embeddings = []
            with open(word_dir, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            product = np.dot(embeddings, embeddings.T)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            cos_sim = product / np.dot(norm, norm.T)
            self.cos_sim = cos_sim
        
        self.sim_mean = np.mean(self.cos_sim)
        self.sim_std = np.std(self.cos_sim)

    def similarity(self, w1, w2):
        i1 = self.word2idx[w1]
        i2 = self.word2idx[w2]
        return self.cos_sim[i1][i2]

    def similarity_threshold(self, w1, w2, alpha_score):
        sim_score = self.similarity(w1, w2)
        return sim_score > self.sim_mean + (alpha_score*self.sim_std)
        


class BertCoreClass:
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir='sentence-transformers/all-MiniLM-L6-v2', cuda_machine=5, max_len=256):
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_dir_upstream)
        self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(pretrained_dir_downstream)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_dir_upstream, fast=True)

        self.bert_classifier.eval()
        self.bert_mlm.eval()

        self.cuda_machine = cuda_machine
        self.max_len = max_len
        
        self.sentence_encoder = SentenceEncoder(sentence_dir)

    def encode(self,text):
        inputs = self.bert_tokenizer.encode_plus(
                                                text,  return_tensors="pt", add_special_tokens = True, #truncation=True, 
                                                #padding=True, 
                                                return_attention_mask = True,  
                                                max_length=512#self.max_len
                                                )
        return inputs

    def decode(self, inputs):
        return self.bert_tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

    def add_mask(self, inputs, mask_pos):
    
        labels = inputs["input_ids"].clone()
        inputs["input_ids"][0][mask_pos] = self.bert_tokenizer.mask_token_id
    
        labels[inputs["input_ids"] != self.bert_tokenizer.mask_token_id] = -100
    
        return labels

    def mlm_compute_loss(self, inputs, labels=None):

        with torch.cuda.device(self.cuda_machine), torch.no_grad():
            mlm_loss = self.bert_mlm(
                    input_ids=inputs["input_ids"],
                    labels=labels,
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"]
                    )
        return mlm_loss


    def classify(self, inputs, label=None, num_classes=2):
        with torch.cuda.device(self.cuda_machine), torch.no_grad():
            if label is not None:
                label = torch.Tensor([label]).long()
            classification_loss = self.bert_classifier(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    labels=label
                    )
        return classification_loss

    def mlm_best_candidates(self, logits, mask_pos, num_tokens=50):

        to_pred = logits[0][mask_pos]

        pred_tokens = torch.argsort(to_pred, descending=True)[:num_tokens]
    
        return pred_tokens

