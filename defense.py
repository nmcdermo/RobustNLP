from ssl_utils import get_adversarial_attacks
from defenses import HighestLossDefense, WordSimilarityDefense
import sys
import os

if __name__ == "__main__":

    
    sentence = sys.argv[1] 
    correct_label = int(sys.argv[2])
    
    adv_sentences, orig_sentences, adv_labels, orig_labels = get_adversarial_attacks()

    pretrained_dir_finetuned = os.path.join(os.pardir, "TextFooler","BERT","yelp2")
    pretrained_dir_base = "bert-base-uncased"
    defender = WordSimilarityDefense(pretrained_dir_base, pretrained_dir_finetuned)
    #inc = defender.encode(sentence)
    #dec = defender.bert_tokenizer.decode(inc.input_ids[0], skip_special_tokens=True)
    #print(dec)

    defender.defend(sentence, correct_label, threshold_sigma=2, total_replacements=3)
    """
    num_successes = 0
    tot_difference = 0
    for i in range(50):
        print()
        print("orig_sentence: " + orig_sentences[i])
        #print("adv_sentence: " + adv_sentences[i] + ".")
        suc, new_sentence, _ = defender.defend(adv_sentences[i], orig_labels[i], total_replacements=6)
        if suc:
            num_successes += 1
            tot_difference += _
        print("Success: " + str(suc))
        print("Difference: " + str(_))
        #print("New Sentence: " + new_sentence)
    #print(sim)
    print(num_successes/50)
    print(tot_difference/num_successes)
    
    """

