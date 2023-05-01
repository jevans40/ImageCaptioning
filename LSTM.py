import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioningLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(ImageCaptioningLSTM, self).__init__()

        # Embedding layer to convert word indices to word vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM-based decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer to convert LSTM outputs to vocabulary scores
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions, features, mask = None):
        # Extract features from the images using the pre-trained CNN

        # Pass captions through the embedding layer
        embeddings = self.embedding(captions)

        # Combine features and embeddings
        inputs = torch.cat((features, embeddings), dim=1)

        # Pass the combined inputs through the LSTM
        lstm_outputs, _ = self.lstm(inputs)

        # Convert LSTM outputs to vocabulary scores
        vocab_scores = self.fc(lstm_outputs)

        return vocab_scores