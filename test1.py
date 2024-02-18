import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define your language model architecture
class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SmallLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

# Define a custom dataset class to handle large text files
class LargeTextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.read()
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+self.seq_length]
        return input_seq, target

# Hyperparameters
vocab_size = 256  # Assuming ASCII characters
embedding_dim = 64
hidden_dim = 128
seq_length = 100  # Length of input sequences
batch_size = 128
num_epochs = 10
learning_rate = 0.001

# Create dataset and dataloaders
dataset = LargeTextDataset('output.txt', seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = SmallLanguageModel(vocab_size, embedding_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        inputs = torch.tensor([list(map(ord, s)) for s in inputs])  # Convert characters to ASCII codes
        inputs = inputs.long()
        targets = torch.tensor(list(map(ord, targets))).long()  # Convert characters to ASCII codes
        hidden = torch.zeros(1, inputs.size(0), hidden_dim)  # Initialize hidden state
        output, _ = model(inputs, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# Save the trained model
torch.save(model.state_dict(), 'small_language_model.pth')
