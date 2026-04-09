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

def create_batches(x, y, batch_size, max_len):
    num_batches = len(x) // batch_size
    x_batches = x[:batch_size*num_batches].view(num_batches, batch_size, max_len)
    y_batches = y[:batch_size*num_batches].view(num_batches, batch_size, max_len)
    return x_batches, y_batches

def predict(model, x_data):
    model.eval()
    with torch.no_grad():
        outputs = model(x_data)
        predictions = torch.argmax(outputs, dim=-1)
    return predictions

def decode_predictions(preds, label_vocab, original_data):
    decoded = []

    for i, (words, _) in enumerate(original_data):
        sent = []
        for j, word in enumerate(words):
            pred_tag_idx = preds[i][j].item()
            pred_tag = label_vocab.idx2word[pred_tag_idx]
            sent.append((word, pred_tag))
        decoded.append(sent)

    return decoded

def save_predictions(decoded_data, filename):
    """
    Saves predictions in the same format as training data: index \t word \t tag \t - \t -
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sentence in decoded_data:
            for i, (word, tag) in enumerate(sentence, start=1):
                f.write(f"{i}\t{word}\t{tag}\t-\t-\n")
            f.write("\n")

class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        self.word_embedding = nn.Embedding(nwords, DIM_EMBEDDING)
        self.dropout = nn.Dropout(0.3)         #edit
        self.rnn = nn.LSTM(DIM_EMBEDDING, RNN_HIDDEN, batch_first=True, bidirectional=True) #edited -- yes, we call it RNN but it is LSTM
        self.hidden_to_tag = nn.Linear(RNN_HIDDEN*2, ntags)             #edited (*2)
    
    def forward(self, inputData):                               #also edited
        word_vectors = self.dropout(self.word_embedding(inputData))
        lstm_out, _ = self.rnn(word_vectors)  # ignore (h, c)
        tag_space = self.hidden_to_tag(self.dropout(lstm_out))  #edit
        return tag_space
    
torch.manual_seed(0)
DIM_EMBEDDING = 100
RNN_HIDDEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 0.01        #One directional: Epoch 9: Loss = 9.0330583427276 -- for learning rate 0.001 , # Epoch 9: Loss = 7.238125507701625 -- for 0.01
                            #Biderectional: LEARNING_RATE = 0.001 ---> 
EPOCHS = 10

train_data = read_iob2_file('en_ewt-ud-train.iob2')
dev_data = read_iob2_file('en_ewt-ud-dev.iob2')
#test_data = read_iob2_file('en_ewt-ud-test-masked.iob2')

word_vocab = Vocab()
label_vocab = Vocab()

max_len= max([len(x[0]) for x in train_data ])

train_x, train_y = convert_data(train_data, max_len, word_vocab, label_vocab, True)
dev_x, dev_y = convert_data(dev_data, max_len, word_vocab, label_vocab)
# test_x, test_y = convert_data(test_data, max_len, word_vocab, label_vocab)

train_x_batches, train_y_batches = create_batches(train_x, train_y, BATCH_SIZE, max_len)

model = TaggerModel(len(word_vocab.idx2word), len(label_vocab.idx2word))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    # loop over batches
    for x_batch, y_batch in zip(train_x_batches, train_y_batches):
        model.zero_grad()
        predicted_values = model(x_batch)

        flat_predictions = predicted_values.view(BATCH_SIZE * max_len, -1)
        flat_targets = y_batch.view(BATCH_SIZE * max_len)

        # calculate loss (and print)
        loss = loss_function(flat_predictions, flat_targets)
        
        # update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss}")


dev_preds = predict(model, dev_x)
decoded_test = decode_predictions(dev_preds, label_vocab, dev_data)

save_predictions(decoded_test, "dev_predictions_LSTM.iob2")
