import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PINNErrorEstimator(nn.Module):
    """
    Tiny 3-layer CNN Physics-Informed Neural Network (PINN) error estimator.
    Predicts the physics violation score of a given simulation grid state.
    """
    def __init__(self):
        super(PINNErrorEstimator, self).__init__()
        # Tiny CNN architecture: 3 conv layers with 16/32/32 channels, accepting 6 input channels: [CO, NO, NO2, O3, u, v]
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Input x shape: (batch_size, 6, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # flatten to (batch_size, 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1) # output shape (batch_size,)

def compute_numerical_divergence(u, v):
    """
    Computes numerical divergence du/dx + dv/dy of a 2D velocity grid
    using central differences for internal cells and one-sided differences at borders.
    u, v are tensors of shape (batch_size, 64, 64).
    """
    du_dx = torch.zeros_like(u)
    dv_dy = torch.zeros_like(v)
    
    # Interior central differences
    du_dx[:, :, 1:-1] = 0.5 * (u[:, :, 2:] - u[:, :, :-2])
    dv_dy[:, 1:-1, :] = 0.5 * (v[:, 2:, :] - v[:, :-2, :])
    
    # Boundary one-sided differences
    du_dx[:, :, 0] = u[:, :, 1] - u[:, :, 0]
    du_dx[:, :, -1] = u[:, :, -1] - u[:, :, -2]
    dv_dy[:, 0, :] = v[:, 1, :] - v[:, 0, :]
    dv_dy[:, -1, :] = v[:, -1, :] - v[:, -2, :]
    
    return du_dx + dv_dy

stop_training = False

def train_pinn(buffer, device, save_path="data/pinn_estimator.pt"):
    """
    Trains a local instance of the PINN model on a copy of the memory sample buffer.
    Run asynchronously in a separate OS thread to avoid blocking the main server.
    """
    global stop_training
    try:
        if len(buffer) < 10:
            print("[PINN Trainer] Not enough samples to start training.")
            return

        print(f"[PINN Trainer] Starting training pass on {len(buffer)} samples...")
        
        # Instantiate a new local model to avoid cross-thread device/state conflicts
        model = PINNErrorEstimator()
        if os.path.exists(save_path):
            try:
                model.load_state_dict(torch.load(save_path, map_location='cpu'))
            except Exception as load_err:
                print(f"[PINN Trainer] Error loading existing weights: {load_err}")

        model.to(device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Convert buffer list of (state_array, residual_val) to PyTorch tensors
        X_list = []
        Y_list = []
        for state, residual in buffer:
            X_list.append(state)
            Y_list.append(residual)
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)
        
        batch_size = min(32, len(buffer))
        dataset_size = len(buffer)
        
        # Run training epochs
        for epoch in range(5):
            if stop_training:
                print("[PINN Trainer] Training aborted due to server shutdown.")
                return
                
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            
            for i in range(0, dataset_size, batch_size):
                if stop_training:
                    print("[PINN Trainer] Training aborted due to server shutdown.")
                    return
                indices = permutation[i:i+batch_size]
                batch_x = X[indices]
                batch_y = Y[indices]
                
                optimizer.zero_grad()
                
                # Model predictions
                outputs = model(batch_x)
                
                # MSE loss relative to true solver divergence residuals
                loss_mse = F.mse_loss(outputs, batch_y)
                
                # Physics Loss: compute numerical divergence of input velocity field
                # channels: 0=CO, 1=NO, 2=NO2, 3=O3, 4=u, 5=v
                u_batch = batch_x[:, 4, :, :]
                v_batch = batch_x[:, 5, :, :]
                div = compute_numerical_divergence(u_batch, v_batch)
                
                # Enforce that predicted score is penalized if divergence is non-zero
                mean_abs_div = torch.mean(torch.abs(div), dim=(1, 2))
                loss_physics = F.mse_loss(outputs, mean_abs_div)
                
                # Combined Loss (lambda = 0.1)
                loss = loss_mse + 0.1 * loss_physics
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(indices)
                
            mean_loss = epoch_loss / dataset_size
            print(f"[PINN Trainer] Epoch {epoch+1}/5 | Combined Loss: {mean_loss:.6f}")
            
        # Save model checkpoint
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Move back to CPU for saving
        model.to('cpu')
        torch.save(model.state_dict(), save_path)
        print(f"[PINN Trainer] Training complete. Saved model to {save_path}")
        
    except Exception as e:
        print(f"[PINN Trainer] Error during training: {e}")
