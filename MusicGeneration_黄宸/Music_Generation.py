# PT_Part2_Music_Generation_Comet.py
# MIT Deep Learning Lab 1 - Part 2: Music Generation with Comet.ml
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mitdeeplearning as mdl
from scipy.io.wavfile import write
from tqdm import tqdm
import subprocess
import os
from IPython.display import Audio
def abc2wav(abc_text, wav_path="song.wav", abc2midi_path="abc2midi.exe", timidity_path="timidity.exe"):
    temp_abc = wav_path.replace(".wav", ".abc")
    with open(temp_abc, "w", encoding="utf-8") as f:
        f.write(abc_text)
    temp_mid = wav_path.replace(".wav", ".mid")
    try:
        subprocess.run([abc2midi_path, temp_abc, "-o", temp_mid], check=True)
    except subprocess.CalledProcessError as e:
        print("abc2midi 出错:", e)
        return None
    try:
        subprocess.run([timidity_path, temp_mid, "-Ow", "-o", wav_path], check=True)
    except subprocess.CalledProcessError as e:
        print("timidity 出错:", e)
        return None
    os.remove(temp_abc)
    os.remove(temp_mid)
    
    return Audio(wav_path)
# =========================
# 0. Initialize Comet Experiment
# =========================
params = {
    "num_training_iterations": 1000,
    "embedding_dim": 256,
    "hidden_size": 1024,
    "seq_length": 100,
    "batch_size": 8,
    "learning_rate": 5e-3
}
experiment = Experiment(
    api_key="6s5P1JhFis7aNDJYD6BJauIu4",
    project_name="music-generation",
    workspace="hanling114"
)
experiment.set_name("LSTM_music_generation")
experiment.log_parameters(params)

# =========================
# 1. Check Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
experiment.log_text(f"Using device: {device}")

# =========================
# 2. Load and Preview Data
# =========================
songs = mdl.lab1.load_training_data()
print(f"Loaded {len(songs)} songs.")
experiment.log_text(f"Loaded {len(songs)} songs.")
example_song = songs[0]

# =========================
# 3. Character Mapping
# =========================
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
print(f"Vocabulary size: {len(vocab)}")
experiment.log_text(f"Vocabulary size: {len(vocab)}")

def vectorize_string(string):
    return np.array([char2idx[c] for c in string])

vectorized_songs = vectorize_string(songs_joined)

# =========================
# 4. Generate Batches
# =========================
def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - seq_length - 1
    idx = np.random.choice(n, batch_size)
    input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(output_batch, dtype=torch.long)

# =========================
# 5. Define LSTM Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            output, state = self.lstm(x)
        else:
            output, state = self.lstm(x, state)
        logits = self.fc(output)
        if return_state:
            return logits, state
        else:
            return logits

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        return (weight.new(1, batch_size, 1024).zero_().to(device),
                weight.new(1, batch_size, 1024).zero_().to(device))

# =========================
# 6. Loss Function
# =========================
def compute_loss(labels, logits):
    labels = labels.view(-1)
    logits = logits.view(-1, logits.size(-1))
    return nn.functional.cross_entropy(logits, labels)

# =========================
# 7. Training Setup
# =========================
vocab_size = len(vocab)
model = LSTMModel(vocab_size, embedding_dim=params["embedding_dim"], hidden_size=params["hidden_size"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

seq_length = params["seq_length"]
batch_size = params["batch_size"]

print("\nTraining started...\n")
for iteration in range(params["num_training_iterations"]):
    x, y = get_batch(vectorized_songs, seq_length, batch_size)
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(y, logits)
    loss.backward()
    optimizer.step()
    if (iteration + 1) % 50 == 0:
        print(f"Iteration {iteration+1}, Loss: {loss.item():.4f}")
        experiment.log_metric("loss", loss.item(), step=iteration)


print("\nTraining finished!\n")
experiment.log_text("Training finished")

# =========================
# 9. Text Generation
# =========================
def generate_text(model, start_string, generation_length=1000, temperature=1.5):
    model.eval()
    input_idx = [char2idx[s] for s in start_string]
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)
    state = model.init_hidden(1, device)
    text_generated = []

    with torch.no_grad():
        for _ in range(generation_length):
            predictions, state = model(input_idx, state, return_state=True)
            predictions = predictions[:, -1, :] / temperature
            probs = torch.softmax(predictions, dim=-1)
            input_idx = torch.multinomial(probs, num_samples=1)
            text_generated.append(idx2char[input_idx.item()])

    return start_string + ''.join(text_generated)

# =========================
# 10. Generate Music
# =========================
print("Generating music...")
generated_text = generate_text(model, start_string="X", generation_length=1000)
experiment.log_text("Generated text preview:\n" + generated_text[:500])

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
    print(f"\nGenerated Song {i+1} Preview:\n", song[:300], "...\n")
    waveform = abc2wav(song)
    os.remove("song.wav")
    if waveform:
        wav_file = f"generated_song_{i}.wav"
        write(wav_file, 88200, np.frombuffer(waveform.data, dtype=np.int16))
        experiment.log_asset(wav_file)
        print(f"Saved: {wav_file}")

print("\n✅ Music generation complete!")
experiment.end()
