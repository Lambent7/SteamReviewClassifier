import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset, concatenate_datasets
import numpy as np
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# https://huggingface.co/datasets/SirSkandrani/steam_reviews_clean
data = load_dataset("SirSkandrani/steam_reviews_clean", split="train")
# Cleaning dataset
data = data.remove_columns("__index_level_0__")

def fix_labels(t):
    t['label'] = 0 if t['label'] == -1 else 1
    return t
data = data.map(fix_labels)

pos = data.filter(lambda x: x['label'] == 1).select(range(10000))
neg = data.filter(lambda x: x['label'] == 0).select(range(10000))
# Splitting dataset
temp_data = concatenate_datasets([pos, neg]).shuffle(seed = 38258208)
clean_data = temp_data.train_test_split(test_size=0.2)

train_texts = clean_data["train"]["text"]
test_texts  = clean_data["test"]["text"]
train_labels = clean_data["train"]["label"]
test_labels = clean_data["test"]["label"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors  = vectorizer.transform(test_texts)

X_train = torch.tensor(train_vectors.toarray(), dtype=torch.float32).to(device)
X_test = torch.tensor(test_vectors.toarray(), dtype=torch.float32).to(device)

print(train_labels)

y_train = torch.tensor(np.array(train_labels), dtype=torch.long).to(device)
y_test = torch.tensor(np.array(test_labels), dtype=torch.long).to(device)

print(y_train)

# LAYERS AND NODES
model = nn.Sequential(
    nn.Linear(5000, 256),
    nn.GELU(),
    nn.Dropout(.2),
    nn.Linear(256, 2)
).to(device)
loss_fn = nn.CrossEntropyLoss()
# LEARNING RATE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# EPOCHS
for epoch in range(301):
    optimizer.zero_grad()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")


with torch.no_grad():
    logits = model(X_test)
    predicted_labels = logits.argmax(dim=1) # get index of largest value in each row, dim=1 means apply to each row
    accuracy = (predicted_labels == y_test).float().mean()

print("Accuracy:", accuracy.item())

def format_output(predicted):
    if predicted == 1:
        return "🎮 Positive".center(60," ")
    elif predicted == 0:
        return "🗑️ Negative".center(60," ")

os.system('cls' if os.name == 'nt' else 'clear')

print("\n\n\n")
print(" 🎮 Steam Review Sentiment Predictor 🗑️ ".center(60,"="))
print("\n")

user_input = input("     Type your game review: ")

vectorP = vectorizer.transform([user_input])
tensorP = torch.tensor(vectorP.toarray(), dtype=torch.float32).to(device)

with torch.no_grad():
    predictor = model(tensorP)
    predicted = predictor.argmax(dim=1).item()

output = format_output(predicted)
print("\n " + output + "\n")
print("=" * 60)