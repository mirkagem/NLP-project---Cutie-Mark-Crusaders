import torch
from torch import nn

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

def read_iob2_file(path):
    """
    Read CoNLL-style IOB2 file
    :param path: path to read from
    :returns: list of (words, labels) per sentence
    """
    data = []
    current_words = []
    current_tags = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue  # skip comments

            tok = line.split('\t')

            current_words.append(tok[1])  # word
            current_tags.append(tok[2])   # label
                                        #tok[0] is a numbered position where word is in the sentence
                                        #other parts are irrelevant (eg. it is either '-	stephen' OR -	-)
        else:
            if current_words:
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []
    if current_words:
        data.append((current_words, current_tags))  #this is here just in case lust sentence does not end with '' (which.. it does )
    return data

def convert_data(data, max_len, word_vocab, label_vocab, is_training=False):
    number_of_instances = len(data)

    word_tensor = torch.zeros((number_of_instances, max_len), dtype=torch.long)
    tag_tensor = torch.zeros((number_of_instances, max_len), dtype=torch.long)

    for i, (words, tags) in enumerate(data):
        for j, word in enumerate(words):
            word_tensor[i, j] = word_vocab.getIdx(word, add=is_training)
            tag_tensor[i, j] = label_vocab.getIdx(tags[j], add=is_training)
    
    return word_tensor, tag_tensor

def create_batches(x, y, batch_size):
    num_batches = len(x) // batch_size
    x_batches = x[:batch_size*num_batches].view(num_batches, batch_size, max_len)
    y_batches = y[:batch_size*num_batches].view(num_batches, batch_size, max_len)
    return x_batches, y_batches

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

train_data = read_iob2_file('en_ewt-ud-train.iob2')
dev_data = read_iob2_file('en_ewt-ud-dev.iob2')
test_data = read_iob2_file('en_ewt-ud-test-masked.iob2')

word_vocab = Vocab()
label_vocab = Vocab()

max_len= max([len(x[0]) for x in train_data ])

train_x, train_y = convert_data(train_data, max_len, word_vocab, label_vocab, True)
dev_x, dev_y = convert_data(dev_data, max_len, word_vocab, label_vocab)
test_x, test_y = convert_data(test_data, max_len, word_vocab, label_vocab)

train_x_batches, train_y_batches = create_batches(train_x, train_y, BATCH_SIZE)
