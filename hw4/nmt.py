import sys
import itertools
import theano
from theano import tensor as T
sys.path.append('./UltraDeep')
from layer import EmbeddingLayer, HiddenLayer
from network import LSTM
from learning_method import LearningMethod
import numpy as np
from compute_bleu import bleu, bleu_stats, get_bleu

reload(sys)
sys.setdefaultencoding('utf8')

def get_data(input_file):
    return map(lambda x:x.strip().split(), open(input_file, 'r').readlines())

def process(train_source_file, train_target_file, dev_source_file, dev_target_file, test_source_file, test_target_predictions):
    
    train_source_data = get_data(train_source_file)
    train_target_data = get_data(train_target_file)
    dev_source_data = get_data(dev_source_file)
    dev_target_data = get_data(dev_target_file)
    test_source_data = get_data(test_source_file)
    
    source_words = set(itertools.chain(*(train_source_data + dev_source_data)))
    target_words = set(itertools.chain(*(train_target_data + dev_target_data)))
    
    source_word_to_idx = dict((v, i) for i, v in enumerate(source_words))
    target_word_to_idx = dict((v, i) for i, v in enumerate(target_words))
    target_idx_to_word = dict((i, v) for i, v in enumerate(target_words))
    
    # Preparing data    
    train_source_data = [[source_word_to_idx[word] for word in sentence] for sentence in train_source_data]
    dev_source_data = [[source_word_to_idx[word] for word in sentence] for sentence in dev_source_data]
    train_target_data = [[target_word_to_idx[word] for word in sentence] for sentence in train_target_data]
    dev_target_data = [[target_word_to_idx[word] for word in sentence] for sentence in dev_target_data]
    test_source_data = [[source_word_to_idx[word] for word in sentence] for sentence in test_source_data]
    
    # Changing the input numpy arrays to tensor vectors
    source_sentence = T.ivector()
    target_sentence = T.ivector()
    target_gold = T.ivector()
    
    source_word_embedding = 128
    target_word_embedding = 128
    source_hidden_embedding = 256
    target_hidden_embedding = 256
        
    hyper_params = []
    
    vocab_source_size = len(source_words)
    vocab_target_size = len(target_words)
    
    source_lookup = EmbeddingLayer(vocab_source_size, source_word_embedding) 
    target_lookup = EmbeddingLayer(vocab_target_size, target_word_embedding) 
    hyper_params += source_lookup.params + target_lookup.params

    source_lstm_forward = LSTM(source_word_embedding, source_hidden_embedding, with_batch=False)
    
    target_lstm = LSTM(256, target_hidden_embedding, with_batch=False)
    hyper_params += source_lstm_forward.params + target_lstm.params[:-1] # Removing the last output

    tanh_layer = HiddenLayer(source_hidden_embedding, target_word_embedding, activation='tanh')
    softmax_layer = HiddenLayer(target_hidden_embedding + source_hidden_embedding, vocab_target_size, activation='softmax')
    hyper_params += softmax_layer.params

    # Getting the source and target embeddings
    source_sentence_emb = source_lookup.link(source_sentence)
    target_sentence_emb = target_lookup.link(target_sentence)
    last_h = source_lstm_forward.link(source_sentence_emb)

    # Repeating the last encoder_output for target word length times
    # First changing the last encoder_output into a row and vector and repeating target word length times
    broadcast_source_context = T.repeat(last_h.dimshuffle('x', 0), target_sentence_emb.shape[0], axis=0)
    broadcast_source_context = tanh_layer.link(broadcast_source_context)
    target_sentence_emb = T.concatenate((target_sentence_emb, broadcast_source_context), axis=1)
    target_lstm.h_0 = last_h
    target_lstm.link(target_sentence_emb)
    
    # Attention
    ht = target_lstm.h.dot(source_lstm_forward.h.transpose())
    # Normalizing across rows to get attention probabilities
    attention_weights = T.nnet.softmax(ht)
    # Weighted source_context_vector based on attention probabilities
    attention_weighted_vector = attention_weights.dot(source_lstm_forward.h)
    # Concatenating the hidden state from lstm and weighted source_context_vector
    pred = T.concatenate([attention_weighted_vector, target_lstm.h], axis=1)
    # Final softmax to get the best translation word
    prediction = softmax_layer.link(pred)
    
    # Computing the cross-entropy loss
    loss = T.nnet.categorical_crossentropy(prediction, target_gold).mean()
    
    updates = LearningMethod(clip=5.0).get_updates('adam', loss, hyper_params)
    
    # For training
    train_function = theano.function(
        inputs=[source_sentence, target_sentence, target_gold],
        outputs=loss,
        updates=updates
    )

    # For prediction
    predict_function = theano.function(
        inputs=[source_sentence, target_sentence],
        outputs=prediction,
    )
        
    def get_translations(source_sentences):
        translated_sentences = []
        for i, sentence in enumerate(source_sentences):
            source_sentence = np.array(sentence).astype(np.int32)
            translated_so_far = [target_word_to_idx['<s>']]
            while True:
                next_word = predict_function(source_sentence, translated_so_far).argmax(axis=1)[-1] # Get the last translated word
                translated_so_far.append(next_word)
                if next_word == target_word_to_idx['</s>']:
                    translated_sentences.append([target_idx_to_word[x] for x in translated_so_far])
                    break
        return translated_sentences
    
    iterations = 100
    batch_size = 10000
    c = 0
    best_score = -1.0 * sys.maxint
    dev_preds = []
    test_preds = []
    dev_best_preds = []
    test_best_preds = []
    for i in xrange(iterations):
        print 'Iteration {}'.format(i)
        random_indexes = range(len(train_source_data))
        np.random.shuffle(random_indexes)
        loss = []
        for sent_no, index in enumerate(random_indexes):
            src_vector = np.array(train_source_data[index]).astype(np.int32)
            tgt_vector = np.array(train_target_data[index]).astype(np.int32)
            c = train_function(src_vector, tgt_vector[:-1], tgt_vector[1:])                  
            loss.append(c)
            if sent_no % batch_size == 0 and sent_no > 0:
                dev_preds = get_translations(dev_source_data)
                dev_bleu_score = get_bleu(dev_preds)
                if dev_bleu_score > best_score:
                    best_score = dev_bleu_score
                    dev_best_preds = dev_preds[:]
                    # Decoding the test once the dev reaches the baseline
                    if dev_bleu_score >= 28:
                        test_preds = get_translations(test_source_data)
                        test_best_preds = test_preds[:]
                    print 'Dev bleu score {}'.format(dev_bleu_score)
                
        print 'Iteration: {} Loss {}'.format(i, 1.0 * (sum(loss))/len(loss))

            
    dev_output_fp = open('dev_output.txt', 'w')
    test_output_fp = open(test_target_predictions, 'w')
    
    for pred in dev_best_preds:
        dev_output_fp.write(' '.join(pred) + '\n')
    dev_output_fp.close()
    
    for pred in test_best_preds:
        test_output_fp.write(' '.join(pred) + '\n')
    test_output_fp.close()

def main():
    train_source_file = sys.argv[1]
    train_target_file = sys.argv[2]
    dev_source_file = sys.argv[3]
    dev_target_file = sys.argv[4]
    test_source_file = sys.argv[5]
    test_target_predictions = sys.argv[6]
    process(train_source_file, train_target_file, dev_source_file, dev_target_file, test_source_file, test_target_predictions)

if __name__ == "__main__":
    main()
