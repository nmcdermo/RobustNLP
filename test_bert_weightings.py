from ssl_utils import get_adversarial_attacks, BertCoreClass 
import torch
import torch.nn.functional as f
import os

if __name__ == "__main__":

    adv_sentences, orig_sentences, adv_labels, orig_labels = get_adversarial_attacks()

    pretrained_dir_finetuned = os.path.join(os.pardir, "TextFooler","BERT","yelp2")
    pretrained_dir_base = "bert-base-uncased"
    bert_core = BertCoreClass(pretrained_dir_base, pretrained_dir_finetuned)
    
    for i in range(100):
        inputs = bert_core.encode(orig_sentences[i])
        classification_outputs = bert_core.classify(inputs)
        probs = f.softmax(classification_outputs.logits)
        best_choice = torch.argmax(probs).item()
        print(best_choice, orig_labels[i])

    
