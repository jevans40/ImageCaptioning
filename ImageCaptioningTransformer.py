import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0,os.getcwd())

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import Transformer
import numpy as np
import math
import random
from tqdm import tqdm

import json
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pycocoevalcap.cider.cider import Cider


# Defs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("/home/jevans40/fiftyone/coco-2014/train/data"):
    import fiftyone.zoo as foz
    print(foz.list_zoo_datasets())
    dataset = foz.load_zoo_dataset("coco-2014")

###############################################
#             Class Defs                      #
###############################################


# Feature extractor (CNN)
class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last two layers (avgpool and fc)
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Transformer model for caption generation
class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super(ImageCaptioningTransformer, self).__init__()

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = torch.zeros(max_seq_len, d_model).to(device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([Transformer.TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Linear layer for generating tokens
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, mask=None):
        seq_length = x.size(1)
        pos_encoding = self.pos_encoding[:seq_length].unsqueeze(0).repeat(x.size(0), 1, 1)

        x = self.embedding(x) + pos_encoding
        x = torch.cat((image_features, x), dim=1)

        for layer in self.transformer_layers:
            x = layer(x, mask)

        x = self.fc(x)
        return x
    

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, captions, threshold):
        counter = Counter()
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        words = [word for word, count in counter.items() if count >= threshold]
        print(f"Vocab size: {len(words)}")


        for word in words:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def numericalize(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
    

class CustomCocoCaptions(datasets.CocoCaptions):
    def __init__(self, root, annFile, transform=None, max_length=20, vocab = None):
        super(CustomCocoCaptions, self).__init__(root, annFile, transform)
        if vocab == None:
            vocab = Vocabulary()
        captions = get_coco_captions(annFile)
        vocab.build_vocab(captions, threshold=20)
        self.vocab = vocab
        self.max_length = max_length

    def pad_or_trim(self, caption):
        caption = caption + [self.vocab.word2idx["<EOS>"]]
        if len(caption) > self.max_length:
            caption = caption[:self.max_length]
            caption[-1] = self.vocab.word2idx["<EOS>"]
            return caption[:self.max_length]
        else:
            return caption +  [self.vocab.word2idx["<PAD>"]] * (self.max_length - len(caption))
    
    def __getitem__(self, index):
        img, captions = super(CustomCocoCaptions, self).__getitem__(index)
        caption = random.choice(captions)
        caption = self.vocab.numericalize(caption)
        caption = self.pad_or_trim(caption)
        return img, torch.tensor(caption)


###############################################
#             Function Defs                   #
###############################################

def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask==1, float('-inf')).unsqueeze(0)


    
def train(model, feature_extractor, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    dataiter = iter(dataloader)
    for img, captions in tqdm(dataiter, desc="Batch", leave=False):
        img = img.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        # Extract features from the image using the CNN
        features = feature_extractor(img)
        features = features.unsqueeze(1)

        # Generate captions using the transformer model
        input_captions = captions[:, :-1]
        target_captions = captions[:, 1:].reshape(-1)
        mask = create_causal_mask(20).to(device)

        outputs = model(input_captions, features, mask)
        outputs = outputs[:, 1:, :].reshape(-1, outputs.size(2))

        loss = criterion(outputs, target_captions)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def get_coco_captions(annFile):
    with open(annFile) as f:
        data = json.load(f)
    captions = [annotation["caption"] for annotation in data["annotations"]]
    return captions



def show_images(images: List[torch.Tensor],
                real_captions: List[str],
                generated_captions: List[str],
                ncols: int = 3,
                num_images: int = 9,
                seed: Optional[int] = None) -> None:

    if seed is not None:
        random.seed(seed)

    nrows = (num_images + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

    indices = random.sample(range(len(images)), num_images)
    selected_images = [images[i].cpu() for i in indices]
    selected_real_captions = [real_captions[i] for i in indices]
    selected_generated_captions = [generated_captions[i] for i in indices]
    real_cap = []
    gene_cap = []
    for ref, hyp in zip(selected_real_captions, selected_generated_captions):
        ref_caption = [coco_test.vocab.idx2word[idx] for idx in ref if idx not in {coco_test.vocab.word2idx["<SOS>"], coco_test.vocab.word2idx["<EOS>"], coco_test.vocab.word2idx["<PAD>"]}]
        hyp_caption = [coco_test.vocab.idx2word[idx] for idx in hyp if idx not in {coco_test.vocab.word2idx["<SOS>"], coco_test.vocab.word2idx["<EOS>"], coco_test.vocab.word2idx["<PAD>"]}]
        real_cap.append(" ".join(ref_caption))
        gene_cap.append(" ".join(hyp_caption))

    for ax, img, caption, generated_caption in zip(axes.flat, selected_images, real_cap, gene_cap):
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Real: {caption}\nGenerated: {generated_caption}")
        ax.axis('off')
    plt.show()




def test(model, feature_extractor, dataloader, device):
    model.eval()

    references = []
    hypotheses = []
    iterator = iter(dataloader)
    limit = len(iterator)*.10
    passes = 0
    with torch.no_grad():
        for img, captions in tqdm(iterator, desc="Batch", leave=False):
            passes += 1
            if passes > limit:
                break
            img = img.to(device)
            captions = captions.to(device)

            # Extract features from the image using the CNN
            features = feature_extractor(img)
            features = features.unsqueeze(1)

            # Generate captions using the transformer model
            input_captions = torch.full((captions.size(0), 1), coco_test.vocab.word2idx["<SOS>"]).to(device)
            eos_flag = torch.zeros(input_captions.size(0), dtype=torch.bool).to(device)
            pad_idx = coco_test.vocab.word2idx["<PAD>"]
            eos_idx = coco_test.vocab.word2idx["<EOS>"]
            for i in range(1, max_seq_len):
                outputs = model(input_captions, features)
                _, predicted = outputs[:, -1, :].max(1)

                # Check if an <EOS> token is encountered and update the eos_flag
                eos_flag |= (predicted == eos_idx)

                # Replace predicted tokens with <PAD> tokens if the corresponding eos_flag is True
                predicted = torch.where(eos_flag, torch.tensor(pad_idx).to(device), predicted)

                input_captions = torch.cat((input_captions, predicted.unsqueeze(1)), dim=1)
            # Remove <SOS> and convert tensor to list of strings
            input_captions = input_captions.tolist()
            captions = captions.tolist()

            # Convert indices to words and store references and hypotheses
            for ref, hyp in zip(captions, input_captions):
                ref_caption = [coco_test.vocab.idx2word[idx] for idx in ref if idx not in {coco_test.vocab.word2idx["<SOS>"], coco_test.vocab.word2idx["<EOS>"], coco_test.vocab.word2idx["<PAD>"]}]
                hyp_caption = [coco_test.vocab.idx2word[idx] for idx in hyp if idx not in {coco_test.vocab.word2idx["<SOS>"], coco_test.vocab.word2idx["<EOS>"], coco_test.vocab.word2idx["<PAD>"]}]

                references.append([ref_caption])
                hypotheses.append(hyp_caption)
            if passes == 1:
                show_images(img,captions,input_captions,num_images=1,seed=42)

    # Compute BLEU score
    smoothing_function = SmoothingFunction().method1
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)

    # Compute CIDEr-D score
    cider_scorer = Cider()
    references_cider = {i: [" ".join(ref[0])] for i, ref in enumerate(references)}
    hypotheses_cider = {i: [" ".join(hyp)] for i, hyp in enumerate(hypotheses)}
    cider_score, _ = cider_scorer.compute_score(references_cider,hypotheses_cider)

    return bleu_score, cider_score









nltk.download('punkt')








# Hyperparameters
d_model = 512
num_heads = 8
d_ff = 512
num_layers = 4
max_seq_len = 20
dropout = 0.1
num_epochs = 100
learning_rate = 0.001
batch_size = 128





transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

coco_train = CustomCocoCaptions(root = '/home/jevans40/fiftyone/coco-2014/train/data',
                                annFile='/home/jevans40/fiftyone/coco-2014/raw/captions_train2014.json',
                                transform=transform,
                                max_length=max_seq_len)

coco_test = CustomCocoCaptions(root = '/home/jevans40/fiftyone/coco-2014/validation/data',
                                annFile='/home/jevans40/fiftyone/coco-2014/raw/captions_val2014.json',
                                transform=transform,
                                max_length=max_seq_len,
                                vocab=coco_train.vocab)
vocab_size = coco_train.vocab.vocab_size

dataloader = torch.utils.data.DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(coco_test, batch_size=batch_size*3, shuffle=True, num_workers=4)

targetModel = "LSTM"
feature_extractor = ImageFeatureExtractor(d_model).to(device)
caption_generator = None
criterion = None
optimizer = None
latest_model = 0

model_dir = "./Test4/"
if not os.path.exists(model_dir):
        os.makedirs(model_dir)

if targetModel == "Transformer":

    # Create models
    caption_generator = ImageCaptioningTransformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=coco_train.vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(caption_generator.parameters(), lr=learning_rate)

    test1 = iter(dataloader)
    test2 = next(test1)

    print(test2[0].size())






    # Training loop
    latest_model = max([int(fname.split("_")[-1]) for fname in os.listdir(model_dir)], default=0)

    if latest_model >= num_epochs:
        caption_generator.load_state_dict(torch.load(os.path.join(model_dir, f"model_{latest_model}")))
import LSTM

#LSTM hyperparameters
hidden_dim = 512
num_layers = 2
if targetModel == "LSTM":
    caption_generator = LSTM.ImageCaptioningLSTM(vocab_size, d_model, hidden_dim, num_layers).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=coco_train.vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(caption_generator.parameters(), lr=learning_rate)


loss_history = []



plt.ion()
for epoch in tqdm(range(latest_model,latest_model + num_epochs), desc="Epochs"):

    # Plot the loss history
    plt.clf()  # Clear the plot before the next update
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.draw()  # Draw the current plot

    # Test the model and print examples
    bleu_score, cider_score = test(caption_generator, feature_extractor, test_dataloader, device)
    print(f"Bleu Score: {bleu_score:.4f}, CIDEr-D Score: {cider_score:.4f}")

    epoch_loss = train(caption_generator, feature_extractor, dataloader, criterion, optimizer, device)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{latest_model + num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(caption_generator.state_dict(), os.path.join(model_dir, f"model_{epoch+1}"))
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot














'''
AUTO TRAINING STUFF HERE COOLIO
'''
