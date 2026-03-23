from torch import nn
import torch

class Vocab():
    def __init__(self, pad_unk='<PAD>'):
        """
        A convenience class that can help store a vocabulary
        and retrieve indices for inputs.
        """
        self.pad_unk = pad_unk
        self.word2idx = {self.pad_unk: 0}
        self.idx2word = [self.pad_unk]

    def getIdx(self, word, add=False):
        if word not in self.word2idx:
            if add:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
            else:
                return self.word2idx[self.pad_unk]
        return self.word2idx[word]

    def getWord(self, idx):
        return self.idx2word[idx]

def convert_data(data, max_len, word_vocab, label_vocab, is_training=False):
    number_of_instances = len(data)

    word_tensor = torch.zeros((number_of_instances, max_len), dtype=torch.long)
    tag_tensor = torch.zeros((number_of_instances, max_len), dtype=torch.long)

    for i, (words, tags) in enumerate(data):
        for j, word in enumerate(words):
            word_tensor[i, j] = word_vocab.getIdx(word, add=is_training)
            tag_tensor[i, j] = label_vocab.getIdx(tags[j], add=is_training)
    
    return word_tensor, tag_tensor

class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        self.word_embedding = nn.Embedding(nwords, DIM_EMBEDDING)
        self.rnn = nn.RNN(DIM_EMBEDDING, RNN_HIDDEN, batch_first=True)
        self.hidden_to_tag = nn.Linear(RNN_HIDDEN, ntags)
        
    def forward(self, inputData):
        word_vectors = self.word_embedding(inputData)
        rnn_out, _ = self.rnn(word_vectors)
        tag_space = self.hidden_to_tag(rnn_out)
        return tag_space
    
torch.manual_seed(0)
DIM_EMBEDDING = 100
RNN_HIDDEN = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 10

max_len= max([len(x[0]) for x in train_data ])