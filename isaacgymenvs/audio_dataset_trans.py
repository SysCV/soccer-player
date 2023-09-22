import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import wandb

# Directory containing saved dataset files
dataset_dir = "./dataset"
plot_dataset = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

leg_data_list = []
rms_data_list = []


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


# Iterate over the saved dataset files
for filename in os.listdir(dataset_dir):
    if filename.endswith(".pkl"):
        file_path = os.path.join(dataset_dir, filename)
        loaded_data = torch.load(file_path)

        # Extract the necessary data from loaded_data
        leg_data = loaded_data["leg"]
        rms_data = loaded_data["rms"]

        # Append the data to the lists
        leg_data_list.append(leg_data[:, :, :])  # 6 - 30
        rms_data_list.append(rms_data)

# Concatenate the lists to form a larger dataset
concatenated_leg_data = torch.cat(leg_data_list, dim=0)
concatenated_rms_data = torch.cat(rms_data_list, dim=0)


def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


concatenated_rms_data = min_max_normalize(concatenated_rms_data)

if plot_dataset:
    # Plot leg_data
    plt.figure(figsize=(10, 5))
    plt.plot(concatenated_leg_data[:, 1, 6], label="Leg Data")  # 6 - 30
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Leg Data")
    plt.show()

    # Plot rms_data
    plt.figure(figsize=(10, 5))
    plt.plot(concatenated_rms_data, label="RMS Data")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("RMS Data")
    plt.show()

# Initialize WandB
# wandb.init(project="sequence-to-number")


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        dropout=0.0,
        input_dropout=0.0,
    ):
        super(TransformerPredictor, self).__init__()
        self._create_model(
            input_dim,
            model_dim,
            num_classes,
            num_heads,
            num_layers,
            dropout,
            input_dropout,
        )

    def _create_model(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        dropout,
        input_dropout,
    ):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim),
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(model_dim * 15, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x.flatten(1))
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


# Hyperparameters
input_size = 24  # Size of each time step in the sequence
hidden_size = 24  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
num_heads = 4  # Number of attention heads
output_size = 1  # Output size (single number)
learning_rate = 1e-4
batch_size = 32
num_epochs = 200

# Create the model
model = TransformerPredictor(
    input_size,
    hidden_size,
    output_size,
    num_heads,
    num_layers,
).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

# Load your concatenated dataset
# Assuming you have 'concatenated_leg_data' and 'concatenated_rms_data' as tensors
dataset = TensorDataset(
    concatenated_leg_data.to(device), concatenated_rms_data.to(device)
)

# Split the dataset into training and validation sets (adjust the split ratio as needed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


train_losses = []  # To store training losses
val_losses = []  # To store validation losses

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())  # Append training loss

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)  # Append validation loss

    # Log metrics using WandB
    # wandb.log({"Train Loss": loss.item(), "Validation Loss": val_loss})

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}"
    )

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.show()

# After training, you can use the trained model for predictions.
with torch.no_grad():
    output_batch = model(concatenated_leg_data[1:400, :, :].to(device))

# Assuming 'output_batch' contains the model's predictions in the shape [batch_size, output_size]
# Plot the output in order

plt.figure()
plt.plot(output_batch.to("cpu").numpy(), label="Predicted Value")
plt.plot(concatenated_rms_data[1:400].to("cpu").numpy(), label="RMS Data")
plt.xlabel("Time Step")
plt.ylabel("Predicted Value")
plt.title(f"Output for Sample 1-400")
plt.legend()
plt.show()
