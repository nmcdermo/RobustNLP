from ssl_utils import BertCoreClass, get_adversarial_attacks
import os
from copy import deepcopy

def add_csv_line(file_obj, sentence, is_adv, pos, loss):
    file_obj.write(",".join([str(sentence), str(is_adv), str(pos), str(loss)]) + "\n")

if __name__ == "__main__":

    adv_sentences, orig_sentences, adv_labels, orig_labels = get_adversarial_attacks()

    pretrained_dir_finetuned = os.path.join(os.pardir, "TextFooler","BERT","yelp2")
    pretrained_dir_base = "bert-base-uncased"
    bert_core = BertCoreClass(pretrained_dir_base, pretrained_dir_finetuned)
    with open("ssl_losses.csv", "w") as f:

        for i in range(50):
            print("Sentence "+str(i))
            adv_sentence = adv_sentences[i]
            orig_sentence = orig_sentences[i]
            input_orig = bert_core.encode(orig_sentence)
            input_adv = bert_core.encode(adv_sentence)
            print(input_orig["input_ids"])
            for j in range(len(input_orig["input_ids"][0])-2):
                input_orig_masked = deepcopy(input_orig)
                labels_orig = bert_core.add_mask(input_orig_masked, j+1)
                ssl_loss = bert_core.mlm_compute_loss(input_orig_masked, labels_orig)
                add_csv_line(f, i, False, j, ssl_loss.loss.item())

            for j in range(len(input_adv["input_ids"][0])-2):
                input_adv_masked = deepcopy(input_adv)
                labels_adv = bert_core.add_mask(input_adv_masked, j+1)
                ssl_loss = bert_core.mlm_compute_loss(input_adv_masked, labels_adv)
                add_csv_line(f, i, True, j, ssl_loss.loss.item())
        
        f.close()
    print("Done!")
