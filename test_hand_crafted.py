from ssl_utils import BertCoreClass
import torch
import sys
import os
from copy import deepcopy

if __name__ == "__main__":

    sentence = sys.argv[1]
    correct_label = int(sys.argv[2])
    mask_pos = int(sys.argv[3])
    print("Original Sentence: " + sentence)

    pretrained_dir_downstream = os.path.join(os.pardir,"TextFooler", "BERT","yelp2")
    pretrained_dir_upstream = "bert-base-uncased"
    predictor = BertCoreClass(pretrained_dir_upstream, pretrained_dir_downstream)
    inputs = predictor.encode(sentence)
    mlm_inputs = deepcopy(inputs)

    labels = predictor.add_mask(mlm_inputs, mask_pos)
    
    print(inputs["input_ids"], labels)

    mlm_loss = predictor.mlm_compute_loss(mlm_inputs, labels)

    classification_loss = predictor.classify(inputs, correct_label)

    print("MLM Loss: " + str(round(mlm_loss.loss.item(), 3)))
    print("Classification Loss: " + str(round(classification_loss.loss.item(), 3)))
    print("Predicted Label: " + str(torch.argmax(classification_loss.logits).item()))
    print("Actual Label: " + str(correct_label))
    word_candidates = predictor.mlm_best_candidates(mlm_loss, mask_pos)
    inputs["input_ids"][0][mask_pos] = word_candidates[0].item()

    classification_loss_new = predictor.classify(inputs, correct_label)

    print("New MLM Loss: " + str(round(mlm_loss.loss.item(), 3)))
    print("New classification Loss: " + str(round(classification_loss_new.loss.item(), 3)))
    print("New predicted Label: " + str(torch.argmax(classification_loss_new.logits).item()))
    print(classification_loss_new.logits)
    print(predictor.bert_tokenizer.decode(word_candidates[0]))
    print(predictor.bert_tokenizer.decode(labels[0][mask_pos]))
    #print(word_candidates)
