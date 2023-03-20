
from ssl_utils import BertCoreClass
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
    
    clean = False
    if clean:
        adv_df = get_clean_sentences_csv("attacks/fullagnewspwws.csv")
    else:
        adv_df = get_adversarial_attacks_csv("attacks/fullagnewspwws.csv")
    print(adv_df.columns)
    print(adv_df[["original_output", "perturbed_output", "ground_truth_output"]])
    print(adv_df[adv_df["original_output"] == adv_df["perturbed_output"]])
    
    pretrained_dir_base = "bert-base-uncased"
    pretrained_dir_finetuned = "textattack/bert-base-uncased-ag-news"
    classifier = BertCoreClass(pretrained_dir_base, pretrained_dir_finetuned, cuda_machine=7)

    #defender = WordSimilarityDefense(pretrained_dir_base, pretrained_dir_finetuned, cuda_machine=7)
    successes = 0
    total_sim = 0
    j = 0
    #assert(len(adv_df) == 1000)

    with open("results/agnews_pwws_full.txt","r") as f: 
        for l in f.readlines():
            s = l.split("|")
            res = s[0]
            orig = s[1]
            for i, row in adv_df.iterrows():
                if row["original_text"] in orig:
                    orig_clean = row["perturbed_text"]
                    orig_label = row["ground_truth_output"]

            inputs = classifier.encode(orig_clean) 
            lo = classifier.classify(inputs, orig_label)
            print(lo, orig_label, res)
