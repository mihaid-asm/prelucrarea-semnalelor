import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('household_hourly_dataset.csv')

print(f"Total inregistrari: {len(df)}")
print(f"\nPrimele randuri:")
print(df.head())
print(f"\nInfo dataset:")
print(df.info())

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')
df.set_index('Datetime', inplace=True)

feature_names = df.columns.tolist()

print(f"\nValori lipsa per coloana:")
print(df.isnull().sum())

df = df.dropna()

print(f"\nDupa curatare: {len(df)} inregistrari")
data_array = df.values.astype(np.float32)
total_len = len(data_array)

train_size = int(0.70 * total_len)
val_size = int(0.15 * total_len)
test_size = total_len - train_size - val_size

raw_train = data_array[:train_size]
raw_val = data_array[train_size:train_size + val_size]
raw_test = data_array[train_size + val_size:]

scaler = MinMaxScaler()
scaler.fit(raw_train)

train_normalized = scaler.transform(raw_train)
val_normalized = scaler.transform(raw_val)
test_normalized = scaler.transform(raw_test)

train_data = torch.FloatTensor(train_normalized)
val_data = torch.FloatTensor(val_normalized)
test_data = torch.FloatTensor(test_normalized)

print(f"\n{'=' * 50}")
print(f"SPLIT DATASET:")
print(f"{'=' * 50}")
print(f"Train:      {len(train_data):,} | Range: [{train_data.min():.2f}, {train_data.max():.2f}]")
print(f"Validation: {len(val_data):,} | Range: [{val_data.min():.2f}, {val_data.max():.2f}]")
print(f"Test:       {len(test_data):,} | Range: [{test_data.min():.2f}, {test_data.max():.2f}]")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]  # Global_active_power (prima coloana)
        return x, y

class LSTMUnit(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)

        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev, C_prev):
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        C_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))

        C_t = f_t * C_prev + i_t * C_tilde

        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        h_t = o_t * torch.tanh(C_t)

        h_t = self.dropout(h_t)

        return h_t, C_t


class DeepManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(LSTMUnit(input_size, hidden_size, dropout))

        for _ in range(1, num_layers):
            self.layers.append(LSTMUnit(hidden_size, hidden_size, dropout))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]
        C = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx], C[layer_idx] = layer(x_t, h[layer_idx], C[layer_idx])
                x_t = h[layer_idx]

        y = self.fc(h[-1])
        return y

seq_len = 48
batch_size = 64
num_epochs = 100

train_dataset = TimeSeriesDataset(train_data, seq_len)
val_dataset = TimeSeriesDataset(val_data, seq_len)
test_dataset = TimeSeriesDataset(test_data, seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

model = DeepManualLSTM(
    input_size=len(feature_names), # 7 feature-uri total
    hidden_size=64,
    output_size=1,
    num_layers=3,
    dropout=0.2
).to(device)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

best_val_loss = float('inf')
patience = 7
patience_counter = 0
train_losses = []
val_losses = []

"""print(f"\n{'=' * 60}")
print(f"START TRAINING")
print(f"{'=' * 60}\n")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pt')
        print(f"Model imbunatatit salvat (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping la epoch {epoch + 1}")
            break

print(f"\n{'=' * 60}")
print(f"TRAINING GATA")
print(f"{'=' * 60}")
print(f"Cel mai bun Val Loss: {best_val_loss:.6f}\n")"""

model.load_state_dict(torch.load('best_lstm_model.pt'))
model.eval()

test_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item()

        all_preds.extend(y_pred.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

test_loss /= len(test_loader)

print(f"{'=' * 60}")
print(f"TEST RESULTS")
print(f"{'=' * 60}")
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test RMSE: {np.sqrt(test_loss):.6f}")

plt.figure(figsize=(15, 5))

"""plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)"""

plt.figure(figsize=(12, 6))
plot_points = min(200, len(all_preds))
plt.plot(all_targets[:plot_points], label='Actual', alpha=0.7, linewidth=1.5)
plt.plot(all_preds[:plot_points], label='Predicted', alpha=0.7, linewidth=1.5, color="red", linestyle=':')
plt.xlabel('Sample')
plt.ylabel('Normalized Power')
plt.title(f'Predictions vs Actual (primele {plot_points} samples)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nGrafic salvat ca 'lstm_results.pdf'")
