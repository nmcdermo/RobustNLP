from transformers import BertForMaskedLM, BertTokenizerFast, BertForSequenceClassification, AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from copy import deepcopy
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



class BertCoreClass:
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir='sentence-transformers/all-MiniLM-L6-v2', cuda_machine=5, max_len=128):
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_dir_upstream)
        self.bert_classifier = BertForSequenceClassification.from_pretrained(pretrained_dir_downstream, num_labels=2)
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(pretrained_dir_upstream)

        self.bert_classifier.eval()
        self.bert_mlm.eval()

        self.cuda_machine = cuda_machine
        self.max_len = max_len
        
        self.sentence_encoder = SentenceEncoder(sentence_dir)

    def encode(self,text):
        inputs = self.bert_tokenizer.encode_plus(
                                                text,  return_tensors="pt", add_special_tokens = True, truncation=True, 
                                                padding=True, return_attention_mask = True,  
                                                max_length=self.max_len
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


class CoreDefense(BertCoreClass): 
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir='sentence-transformers/all-MiniLM-L6-v2', cuda_machine=5, max_len=128):
        super().__init__(pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir, cuda_machine=5, max_len=128)

    def get_all_losses(self, inputs):
        self.mlm_losses = []
        sentence_size = len(inputs["input_ids"][0])
        for i in range(1, sentence_size-1):
            adv_inputs = deepcopy(inputs)
            mlm_labels = self.add_mask(adv_inputs, i)
            mlm_loss = self.mlm_compute_loss(adv_inputs, mlm_labels)
            self.mlm_losses.append((i, mlm_loss.loss.item(), mlm_loss.logits))
        self.mlm_losses.sort(key = lambda x:x[1])

    def matching_label(self, inputs, label):
        classification_loss = self.classify(inputs, label)
        print("Classification loss: ", classification_loss.loss)
        chosen_label = torch.argmax(classification_loss.logits).item()
        return chosen_label == label

    def defend(self, sentence, desired_label):
        self.orig_sentence = sentence
        self.adv_sentence = ""
        self.desired_label = desired_label

class HighestLossDefense(CoreDefense):

    def defend(self, sentence, desired_label):
        super().defend(sentence, desired_label)
        inputs = self.encode(sentence)
        assert(not self.matching_label(inputs, self.desired_label))
        self.get_all_losses(inputs)
        token_to_replace = self.mlm_losses[-1][0]
        old_loss = self.mlm_losses[-1][1]
        print("Old MLM Loss", old_loss)
        logits = self.mlm_losses[-1][2]
        new_candidates = self.mlm_best_candidates(logits, token_to_replace)
        print("Old Word: ", self.bert_tokenizer.decode(inputs.input_ids[0][token_to_replace]))
        inputs["input_ids"][0][token_to_replace] = new_candidates[0].item()

        print("New Word: ", self.bert_tokenizer.decode(inputs.input_ids[0][token_to_replace]))
        final_mlm_inputs = deepcopy(inputs)
        final_mlm_labels = self.add_mask(final_mlm_inputs, token_to_replace)
        new_loss = self.mlm_compute_loss(final_mlm_inputs, final_mlm_labels)
        print("New MLM Loss", new_loss.loss.item())
        success = self.matching_label(inputs, self.desired_label)

        new_sentence = self.decode(inputs)

        sim = self.sentence_encoder.similarity(sentence, new_sentence)
        return success, new_sentence, sim

