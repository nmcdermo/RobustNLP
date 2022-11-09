from ssl_utils import BertCoreClass, is_punc
from copy import deepcopy
import torch

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

    def defend(self, sentence, desired_label, ignore_punc = True):
        super().defend(sentence, desired_label)
        inputs = self.encode(sentence)
        self.get_all_losses(inputs)
        assert(not self.matching_label(inputs, desired_label))
        if ignore_punc:
            for i in range(len(self.mlm_losses)):
                token_to_replace = self.mlm_losses[-i][0]
                logits = self.mlm_losses[-i][2] 
                new_candidates = self.mlm_best_candidates(logits, token_to_replace)
                old_input = inputs["input_ids"][0][token_to_replace]
                old_word = self.bert_tokenizer.decode([old_input], remove_special_tokens = True)
                new_word = self.bert_tokenizer.decode([new_candidates[0].item()], remove_special_tokens = True)
                print(old_word, new_word)
                if not is_punc(old_word) and not is_punc(new_word) and old_word != new_word:
                    break
                
        else:
            token_to_replace = self.mlm_losses[-1][0]
            logits = self.mlm_losses[-1][2]
            new_candidates = self.mlm_best_candidates(logits, token_to_replace)

        old_loss = self.mlm_losses[-1][1]
        print("Old MLM Loss", old_loss)
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
