import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# -----------------------------------------------------------------------------
# Part 1: The "Prunable" Linear Layer
# -----------------------------------------------------------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores, same shape as weights
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate_scores to something slightly positive
        # Initialize gate_scores to 0.0 so initial sigmoid(gate_scores) is 0.5
        # This allows gates to reach sparsity (< 0.01 threshold) much faster
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        # Apply sigmoid to turn scores into values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Pruned weights calculation
        pruned_weights = self.weight * gates
        
        # Standard linear operation
        return F.linear(x, pruned_weights, self.bias)

# -----------------------------------------------------------------------------
# Self-Pruning Network Definition
# -----------------------------------------------------------------------------
class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        # CIFAR-10 contains 3 channel images of size 32x32 = 3072 features
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# Part 2 & 3: Training and Evaluation loop
# -----------------------------------------------------------------------------
def get_sparsity_loss(model):
    """Calculate the L1 norm of all gate values after sigmoid."""
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss

def calculate_sparsity(model, threshold=1e-2):
    """Calculate percentage of weights with gate value below threshold."""
    total_weights = 0
    pruned_weights = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_weights += gates.numel()
                pruned_weights += (gates < threshold).sum().item()
    return (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0

def train_and_evaluate(lam, epochs=5):
    print(f"\n--- Training with lambda = {lam} ---")
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model, Loss, Optimizer
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # Data loading (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            
            # Add custom sparsity regularization
            sp_loss = get_sparsity_loss(model)
            total_loss = cls_loss + lam * sp_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss / len(trainloader):.4f}")
        
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = 100 * correct / total
    sparsity_level = calculate_sparsity(model)
    
    print(f"Test Accuracy: {test_acc:.2f}% | Sparsity Level (< 1e-2): {sparsity_level:.2f}%")
    
    return model, test_acc, sparsity_level

def plot_gate_distribution(model, filename="C:/Users/rahul/.gemini/antigravity/brain/5d0dbca7-bdd1-46f2-aaaa-c5beb6369080/artifacts/gate_distribution.png"):
    model.eval()
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
                
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Final Gate Values')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save the plot securely to the artifact directory so it can be viewed in markdown later
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    # Test cases: None (Baseline), Medium Lambda, High Lambda
    lambdas = [0.0, 1e-4, 5e-4] 
    
    results = []
    best_model = None
    best_sparsity = -1
    
    for lam in lambdas:
        model, acc, sparsity = train_and_evaluate(lam, epochs=3)
        results.append({'Lambda': lam, 'Test Accuracy': acc, 'Sparsity Level (%)': sparsity})
        
        # Save model with good sparsity behavior for plotting (usually highest lambda test)
        if sparsity > best_sparsity and acc > 35.0: # Ensure valid convergence
            best_sparsity = sparsity
            best_model = model
            
    if best_model is None: # fallback
        best_model = model
            
    print("\n=== Final Results ===")
    for res in results:
        print(f"Lambda: {res['Lambda']} | Test Acc: {res['Test Accuracy']:.2f}% | Sparsity: {res['Sparsity Level (%)']:.2f}%")
        
    plot_gate_distribution(best_model)
