import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
nltk.download('punkt')

# Dataset 정의
class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, vocab=None):
        df = pd.read_csv(csv_file, names=["text", "label"])
        self.texts = [word_tokenize(t.lower()) for t in df["text"]]
        self.labels = df["label"].tolist()

        if vocab is None:
            self.vocab = build_vocab_from_iterator(self.texts, specials=["<pad>", "<unk>"])
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab

        self.texts = [self.vocab(tokens) for tokens in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

# Collate 함수
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, lengths, torch.tensor(labels)

# LSTM 분류기
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, output_dim=3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        forward = output[range(len(output)), lengths - 1, :self.lstm.hidden_size]
        backward = output[:, 0, self.lstm.hidden_size:]
        combined = torch.cat((forward, backward), dim=1)
        return self.fc(self.dropout(combined))

# 학습 함수
def train(model, train_loader, valid_loader, optimizer, loss_fn, device, epochs=1):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for text, lengths, labels in train_loader:
            text, lengths, labels = text.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(text, lengths)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for text, lengths, labels in valid_loader:
                text, lengths, labels = text.to(device), lengths.to(device), labels.to(device)
                output = model(text, lengths)
                loss = loss_fn(output, labels)
                valid_loss += loss.item()
        print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}")

# 메인 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "NLP_practice2/practice7/train.csv"
    valid_path = "NLP_practice2/practice7/valid.csv"

    train_dataset = TextClassificationDataset(train_path)
    valid_dataset = TextClassificationDataset(valid_path, vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = LSTMClassifier(len(train_dataset.vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, train_loader, valid_loader, optimizer, loss_fn, device, epochs=5)

    torch.save(model.state_dict(), "lstm_model.pt")
