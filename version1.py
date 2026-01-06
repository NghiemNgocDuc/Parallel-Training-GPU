import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

TRAIN_SIZE = 10000
epochs = 10
learning_rate = 1e-2
batch_size = 32

torch.set_float32_matmul_precision("high")

#Load data in binary format
X_train_np = np.fromfile("data/X_train.bin", dtype = np.float32).reshape(60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)

#Apply normalization
mean, std = np.mean(X_train_np), np.std(X_train_np)
X_train_np = (X_train_np - mean) / std
X_test_np = (X_test_np - mean) / std

#Transform to Pytorch tensor and send to GPU
train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(-1, 1, 28, 28)).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda")
test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda")
test_labels = torch.from_numpy(y_test_np).long().to("cuda")

iters_per_epoch = TRAIN_SIZE // batch_size

#Main model
class MLP(nn.Module):
    def __init__ (self, in_layers, hidden_layers, out_layers):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_layers, hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, out_layers)

    def forward(self, x):
        x = x.reshape(batch_size, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#No fixed seed
model = MLP(784, 1024, 10).to("cuda")

# He initialization to prevent gradient explore or vanish and set up manually to match the C implementation
with torch.no_grad():
    # He initialization for Layer 1
    fan_in = model.fc1.weight.size(1)
    scale = (2.0 / fan_in) ** 0.5
    model.fc1.weight.uniform_(-scale, scale)
    model.fc1.bias.zero_()

    # He initialization for Layer 1
    fan_in = model.fc2.weight.size(1)
    scale = (2.0 / fan_in) ** 0.5
    model.fc2.weight.uniform_(-scale, scale)
    model.fc2.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses):
    model.train()
    epoch_loss = 0.0

    for i in range(iters_per_epoch):

        # Load the data first and calculate the amount of time needed to load those data
        data_start = time.time()
        data = train_data[i * batch_size: (i + 1) * batch_size]
        target = train_labels[i * batch_size: (i + 1) * batch_size]
        data_end = time.time()
        timing_stats['data_loading'] += data_end - data_start

        optimizer.zero_grad()

        # Calculate the time needed for the forward pass
        forward_start = time.time()
        outputs = model(data)
        forward_end = time.time()
        timing_stats['forward'] += forward_end - forward_start

        # Calculate the Loss computation timing
        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation'] += loss_end - loss_start

        # Calculate the time needed for backward pass
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start

        # Caluculate Weight updates timing
        update_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        update_end = time.time()
        timing_stats['weight_updates'] += update_end - update_start

    # Store average loss for this epoch
    epoch_losses.append(epoch_loss / iters_per_epoch)

# Calculate average batch accuracy

def evaluate(model, test_data, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0

    with torch.no_grad():
        for i in range(len(test_data) // batch_size):
            data = test_data[i * batch_size: (i + 1) * batch_size]
            target = test_labels[i * batch_size: (i + 1) * batch_size]
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)
            if total_batch != 0:  # Check to avoid division by zero
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1

    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%")


# Main, start training process
if __name__ == "__main__":
    # Initialize timing stats and loss tracking
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    epoch_losses = []

    # Start total timing
    total_start = time.time()

    for epoch in range(epochs):
        train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses)
        print(f"Epoch {epoch} loss: {epoch_losses[epoch]:.4f}")

    # End total timing
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start

    # Print detailed timing breakdown
    print("\n=== PYTORCH CUDA IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")

    print("Detailed Breakdown:")
    print(
        f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(
        f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(
        f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(
        f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(
        f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")

    print("Finished Training")