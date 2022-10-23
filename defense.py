from ssl_utils import HighestLossDefense, get_adversarial_attacks
import sys
import os

if __name__ == "__main__":

    #sentence = sys.argv[1] 
    #correct_label = int(sys.argv[2])
    
    adv_sentences, orig_sentences, adv_labels, orig_labels = get_adversarial_attacks()

    pretrained_dir_finetuned = os.path.join(os.pardir, "TextFooler","BERT","yelp2")
    pretrained_dir_base = "bert-base-uncased"
    defender = HighestLossDefense(pretrained_dir_base, pretrained_dir_finetuned)
    #inc = defender.encode(sentence)
    #dec = defender.bert_tokenizer.decode(inc.input_ids[0], skip_special_tokens=True)
    #print(dec)
    num_successes = 0
    for i in range(100):
        print()
        #print("orig_sentence: " + orig_sentences[i])
        #print("adv_sentence: " + adv_sentences[i] + ".")
        suc, new_sentence, _ = defender.defend(adv_sentences[i], orig_labels[i])
        if suc:
            num_successes += 1
        print("Success: " + str(suc))
        print("Difference: " + str(_))
        #print("New Sentence: " + new_sentence)
    #print(sim)
    print(num_successes/100)
