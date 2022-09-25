from ssl_utils import BertCoreClass
import sys
import os

if __name__ == "__main__":

    sentence = sys.argv[1]
    mask_pos = int(sys.argv[2])
    print(sentence)
    
    
    pretrained_dir = os.path.join(os.getcwd(),"BERT","yelp2")
    
    predictor = BertCoreClass(pretrained_dir)
    inputs = predictor.encode(sentence)
    labels = predictor.add_mask(inputs, mask_pos)
    
    print(inputs["input_ids"], labels)

    mlm_loss = predictor.mlm_compute_loss(inputs, labels)
    print(mlm_loss.loss)

    word_candidates = predictor.mlm_best_candidates(mlm_loss, mask_pos)
    print(word_candidates)
