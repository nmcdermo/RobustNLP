from transformers import BertForMaskedLM, BertTokenizerFast, BertForSequenceClassification
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

class BertCoreClass:
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, cuda_machine=5, max_len=128):
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_dir_upstream)
        self.bert_classifier = BertForSequenceClassification.from_pretrained(pretrained_dir_downstream, num_labels=2)
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(pretrained_dir_upstream)

        self.bert_classifier.eval()
        self.bert_mlm.eval()

        self.cuda_machine = cuda_machine
        self.max_len = max_len

    def encode(self,text):
        inputs = self.bert_tokenizer.encode_plus(
                                                text,  return_tensors="pt", add_special_tokens = True, truncation=True, 
                                                padding=True, return_attention_mask = True,  
                                                max_length=self.max_len
                                                )
        return inputs

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
        print(inputs["input_ids"].shape)
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

    def mlm_best_candidates(self, mlm_loss, mask_pos, num_tokens=50):

        to_pred = mlm_loss.logits[0][mask_pos]

        pred_tokens = torch.argsort(to_pred, descending=True)[:num_tokens]
    
        return pred_tokens


