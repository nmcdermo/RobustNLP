from ssl_utils import BertCoreClass, is_punc, WordEncoder
from copy import deepcopy
import torch

class CoreDefense(BertCoreClass): 
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir='sentence-transformers/all-MiniLM-L6-v2', cuda_machine=5, max_len=128):
        super().__init__(pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir, cuda_machine=5, max_len=128)

    def get_all_losses(self, inputs):
        self.mlm_losses = []
        sentence_size = len(inputs["input_ids"][0])
        for i in range(1, sentence_size-1):
            #print(i)
            adv_inputs = deepcopy(inputs)
            mlm_labels = self.add_mask(adv_inputs, i)
            mlm_loss = self.mlm_compute_loss(adv_inputs, mlm_labels)
            self.mlm_losses.append((i, mlm_loss.loss.item(), mlm_loss.logits))
        self.mlm_losses.sort(key = lambda x:x[1])

    def matching_label(self, inputs, label):
        classification_loss = self.classify(inputs, label)
        #print("Classification loss: ", classification_loss.loss)
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
        decoded_sentence = self.decode(inputs)
        self.get_all_losses(inputs)
        #assert(not self.matching_label(inputs, desired_label))
        if ignore_punc:
            for i in range(len(self.mlm_losses)):
                token_to_replace = self.mlm_losses[-i][0]
                logits = self.mlm_losses[-i][2] 
                new_candidates = self.mlm_best_candidates(logits, token_to_replace)
                old_input = inputs["input_ids"][0][token_to_replace]
                old_word = self.bert_tokenizer.decode([old_input], remove_special_tokens = True)
                new_word = self.bert_tokenizer.decode([new_candidates[0].item()], remove_special_tokens = True)
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

        sim = self.sentence_encoder.similarity(decoded_sentence, new_sentence)
        return success, new_sentence, sim

class SentenceSimilarityDefense(CoreDefense):
    def defend(self, sentence, desired_label, threshold_score=0.95, total_replacements=4):
        super().defend(sentence, desired_label)
        inputs = self.encode(sentence)
        decoded_sentence = self.decode(inputs)
        self.get_all_losses(inputs)
        #assert(not self.matching_label(inputs, desired_label))
        for i in range(len(self.mlm_losses)):
            token_to_replace = self.mlm_losses[-i][0]
            logits = self.mlm_losses[-i][2]
            new_candidates = self.mlm_best_candidates(logits, token_to_replace)
            #print(len(new_candidates), new_candidates, new_candidates[0].item())
            for j in range(len(new_candidates)):
                inputs_copy = deepcopy(inputs)
                old_input = inputs["input_ids"][0][token_to_replace]
                inputs_copy["input_ids"][0][token_to_replace] = new_candidates[j]
                old_word = self.bert_tokenizer.decode([old_input], remove_special_tokens = True)
                new_sentence = self.decode(inputs_copy)
                new_word = self.bert_tokenizer.decode([new_candidates[j].item()], remove_special_tokens = True)
                #print(old_word, new_word) 
                sentence_sim = self.sentence_encoder.similarity(decoded_sentence, new_sentence)
                if old_word == new_word:
                    break
                if not is_punc(old_word) and not is_punc(new_word) and sentence_sim > threshold_score:
                    #print("Old Word: ", self.bert_tokenizer.decode(inputs.input_ids[0][token_to_replace]))
                    #print("New Sentence", self.decode(inputs))
                    #print("Sentence similarity", sentence_sim)
                    inputs = inputs_copy
                    total_replacements -= 1
                    break
            if total_replacements == 0 or sentence_sim < threshold_score + 0.025:
                break
        #print("New Sentence", self.decode(inputs))

        success = self.matching_label(inputs, self.desired_label)
        return success, new_sentence, sentence_sim


class WordSimilarityDefense(CoreDefense):
    def __init__(self, pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir='sentence-transformers/all-MiniLM-L6-v2', word_dir='../TextFooler/emb', cuda_machine=5, max_len=128):
        super().__init__(pretrained_dir_upstream, pretrained_dir_downstream, sentence_dir, cuda_machine=5, max_len=128)
        self.word_encoder = WordEncoder(word_dir)
    def defend(self, sentence, desired_label, threshold_sigma=2, total_replacements=1):
        super().defend(sentence, desired_label)
        #print("HERE")
        inputs = self.encode(sentence)
        #print("encoded")
        decoded_sentence = self.decode(inputs)
        #print("decoded")
        self.get_all_losses(inputs)
        #assert(not self.matching_label(inputs, desired_label))
        for i in range(len(self.mlm_losses)):
            token_to_replace = self.mlm_losses[-i][0]
            logits = self.mlm_losses[-i][2]
            new_candidates = self.mlm_best_candidates(logits, token_to_replace)
            #print(len(new_candidates), new_candidates, new_candidates[0].item())
            for j in range(len(new_candidates)):
                inputs_copy = deepcopy(inputs)
                old_input = inputs["input_ids"][0][token_to_replace]
                inputs_copy["input_ids"][0][token_to_replace] = new_candidates[j]
                old_word = self.bert_tokenizer.decode([old_input], remove_special_tokens = True)
                new_word = self.bert_tokenizer.decode([new_candidates[j].item()], remove_special_tokens = True)
                #print(new_word in self.word_encoder.word2idx)
                print(old_word, new_word)
                #if old_word in self.word_encoder.word2idx and new_word in self.word_encoder.word2idx:
                #    print(self.word_encoder.similarity(old_word, new_word))
                if old_word == new_word or old_word not in self.word_encoder.word2idx:
                    break
                if not is_punc(old_word) and not is_punc(new_word) and new_word in self.word_encoder.word2idx and self.word_encoder.similarity_threshold(old_word, new_word, threshold_sigma):
                    #print("Old Word: ", self.bert_tokenizer.decode(inputs.input_ids[0][token_to_replace]))
                    #print("New Sentence", self.decode(inputs_copy))
                    inputs = inputs_copy
                    total_replacements -= 1
                    break
            if total_replacements == 0:
                #or sentence_sim < threshold_score + 0.025:
                break
        #print("New Sentence", self.decode(inputs))

        new_sentence = self.decode(inputs_copy)
        sentence_sim = self.sentence_encoder.similarity(decoded_sentence, new_sentence)
        success = self.matching_label(inputs, self.desired_label)
        return success, new_sentence, sentence_sim
