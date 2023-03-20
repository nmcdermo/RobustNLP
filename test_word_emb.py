from defense import WordSimilarityDefense
import torch
import sys
import os
import numpy as np
import pandas as pd

def get_adversarial_attacks_csv(adv_sentences_file):
    adv_df = pd.read_csv(adv_sentences_file)
    adv_df = adv_df[adv_df["result_type"] == "Successful"]
    return adv_df

def get_clean_sentences_csv(adv_sentences_file):
    adv_df = pd.read_csv(adv_sentences_file)
    adv_df = adv_df[(adv_df["result_type"] == "Successful")|(adv_df["result_type"] == "Failed")]
    adv_df = adv_df[:1000]
    return adv_df

if __name__ == "__main__":
    """
    word_embeddings = WordEncoder("../TextFooler/emb")
    print(word_embeddings.similarity("cheap", "inexpensive"))
    avg_sim = np.mean(word_embeddings.cos_sim[0])
    sim_std = np.std(word_embeddings.cos_sim[0])
    print(avg_sim, sim_std)
    """
    
    clean = True
    if clean:
        adv_df = get_clean_sentences_csv("attacks/fullagnewspwws.csv")
    else:
        adv_df = get_adversarial_attacks_csv("attacks/fullyelppwws.csv")
    print(adv_df.columns)
    print(adv_df[["original_output", "perturbed_output", "ground_truth_output"]])
    print(adv_df[adv_df["original_output"] == adv_df["perturbed_output"]])
    pretrained_dir_base = "bert-base-uncased"
    pretrained_dir_finetuned = "textattack/bert-base-uncased-ag-news"
    defender = WordSimilarityDefense(pretrained_dir_base, pretrained_dir_finetuned, cuda_machine=7)
    successes = 0
    total_sim = 0
    j = 0
    assert(len(adv_df) == 1000)
    with open("results/agnews_clean_full.txt","w") as f: 
        for i, row in adv_df.iterrows():
            if clean:
                adv_sentence = row['original_text']
            else:
                adv_sentence = row['perturbed_text']
            correct_label = row['ground_truth_output']
            if clean:
                assert(defender.matching_label(defender.encode(adv_sentence), correct_label))
            else:
                assert(not defender.matching_label(defender.encode(adv_sentence), correct_label))
            #print(adv_sentence, correct_label)
            
            suc, new_sentence, sim = defender.defend(adv_sentence, correct_label, threshold_sigma=2, total_replacements=3)
            if suc:
                print(suc)
                successes += 1
                total_sim += sim
            f.write(str(suc) + "|" + row["original_text"] + adv_sentence + new_sentence + "|" + str(sim) + "\n")
            
            j += 1
            print("Done sample " + str(j))
    print(successes/j)
    print(total_sim/successes)

