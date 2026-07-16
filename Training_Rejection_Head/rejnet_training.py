"""
Train and evaluate rejection-head MLP architectures on extracted tile features.

The script performs repeated training runs for multiple network configurations,
evaluates sensitivity and rejection-rate trade-offs, saves summary results.

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from pathlib import Path
import random
from torchinfo import summary
import itertools
import pandas as pd
from tqdm import tqdm

torch.manual_seed(400)
root = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureDataset(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.npz"))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = np.load(self.files[index])

        x = torch.from_numpy(data['feature']).float()
        y = torch.from_numpy(data['labels']).long()

        return x, y


class RejHeadNet(nn.Module):
    def __init__(self, input_features, hidden_layers = None, 
                 tail = 'simple', dropout_prob = 0.5):
        super().__init__()
        
        self.in_features = input_features
        self.tail = tail

        if hidden_layers is None:
            hidden_layers = [128, 64, 32, 10]

        layers = []
        current_dim = input_features
        for i_layer, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if (i_layer == 0) and (dropout_prob > 0): 
                layers.append(nn.Dropout(dropout_prob))
            current_dim = h_dim
        
        if tail == 'simple':
            layers.append(nn.Linear(current_dim, 2))
        elif tail == 'sigmoid':
            layers.append(nn.Linear(current_dim, 1))
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(-1, self.in_features)
        return self.network(x)

def get_accuracy(outputs, targets):
    if outputs.dim() > 1 and outputs.shape[-1] == 2:
        predicted = torch.argmax(outputs, dim=1)
    else:
        predicted = (outputs.view(-1) >= 0.5).long()

    correct = (predicted == targets).sum().item()
    return 100 * correct / targets.numel()

def get_rejection_rate(outputs, thresholds = None):
    
    if thresholds is None:
        thresholds = [0.5]

    if outputs.dim() > 1 and outputs.shape[-1] == 2:  # softmax tail
        probs = torch.softmax(outputs, dim=-1)
        prob_positive = probs[:, 1]
    else: 
        prob_positive = outputs.view(-1)
    
    rejections = []
    for threshold in thresholds:
        negative_count = (prob_positive < threshold).sum().item()
        rejection_rate = negative_count / len(prob_positive)
        rejections.append(rejection_rate)
    
    return rejections


def get_sensitivity(outputs, target, thresholds = None):

    if thresholds is None:
        thresholds = [0.5]

    if outputs.dim() > 1 and outputs.shape[-1] == 2:  # softmax tail
        probs = torch.softmax(outputs, dim=-1)
        prob_positive = probs[:, 1]
    else: # sigmoid tail
        prob_positive = outputs.view(-1)

    positive_idx = (target == 1)

    if not positive_idx.any():
        return [0.0] * len(thresholds)

    relevant_probs = prob_positive[positive_idx]
    total_actual_positives = positive_idx.sum().item()

    sensitivities = []
    for threshold in thresholds:
        true_positive_count = (relevant_probs > threshold).sum().item()
        sensitivity = true_positive_count / total_actual_positives
        sensitivities.append(sensitivity)

    return sensitivities

def train(model, train_loader, test_loader, criterion, epochs = 10, val_every = 10, sensitivity_thresholds = None, testing=True):
    
    if sensitivity_thresholds is None:
        sensitivity_thresholds = [0.5]

    optimizer = torch.optim.Adam( model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    model.train()

    for epoch in tqdm(range(epochs), desc='Epochs'):
        train_acc_list = []

        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features, targets = features.to(device), targets.to(device).view(-1)

            optimizer.zero_grad()
            outputs = model(features)

            if model.tail == 'sigmoid':
                loss = criterion(outputs.view(-1), targets.float())
            elif model.tail == 'simple':
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_acc = get_accuracy(outputs, targets)
            train_acc_list.append(train_acc)

            if testing:
                if (batch_idx % val_every == 0)  and (batch_idx>0):
                    model.eval()
                    test_accuracies = []
                    test_sensitivities = {t: [] for t in sensitivity_thresholds}

                    with torch.no_grad():
                        sample_size = min(100, len(test_loader.dataset))
                        indices = random.sample(range(len(test_loader.dataset)), sample_size)
                        for idx in indices:
                            test_input, test_target = test_loader.dataset[idx]
                            test_input, test_target = test_input.to(device), test_target.to(device).view(-1)

                            test_output = model(test_input)
                            test_accuracies.append(get_accuracy(test_output, test_target))

                            sensitivities = get_sensitivity(test_output, test_target, sensitivity_thresholds)
                            for i_t, t in enumerate(sensitivity_thresholds):
                                test_sensitivities[t].append(sensitivities[i_t])

                    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
                    recent_train_acc = train_acc_list[-100:]
                    avg_recent_train_acc = sum(recent_train_acc) / len(recent_train_acc)
                    print(f"Epoch {epoch+1}, Batch {batch_idx} | lr {scheduler.get_last_lr()[0]} | Train Acc (recent) {avg_recent_train_acc:0.2f} | Test Acc {avg_test_acc:0.2f}")

                    
                    for t in sensitivity_thresholds:
                        avg_sens = sum(test_sensitivities[t]) / len(test_sensitivities[t])
                        print(f"Threshold {t:.2f} -> Sensitivity: {avg_sens:.2%}")

                    model.train()
        
        scheduler.step()


def evaluate_model(model, sensitivity_thresholds, train_dataloader, test_dataloader):
    
    model.eval()
    test_accuracies = []
    train_accuracies = []
    test_sensitivities = {t: [] for t in sensitivity_thresholds}
    test_rejections = {t: [] for t in sensitivity_thresholds}
    train_sensitivities = {t: [] for t in sensitivity_thresholds}
    train_rejections = {t: [] for t in sensitivity_thresholds}

    with torch.no_grad():
        for (features, targets) in test_dataloader:
            features, targets = features.to(device), targets.to(device).view(-1)
            test_output = model(features)
            test_accuracies.append(get_accuracy(test_output, targets))
            sensitivities = get_sensitivity(test_output, targets, sensitivity_thresholds)
            rejections = get_rejection_rate(test_output, sensitivity_thresholds)
            for i_t, t in enumerate(sensitivity_thresholds):
                test_sensitivities[t].append(sensitivities[i_t])
                test_rejections[t].append(rejections[i_t])
        
        for (features, targets) in train_dataloader:
            features, targets = features.to(device), targets.to(device).view(-1)
            train_output = model(features)
            train_accuracies.append(get_accuracy(train_output, targets))
            sensitivities = get_sensitivity(train_output, targets, sensitivity_thresholds)
            rejections = get_rejection_rate(train_output, sensitivity_thresholds)
            for i_t, t in enumerate(sensitivity_thresholds):
                train_sensitivities[t].append(sensitivities[i_t])
                train_rejections[t].append(rejections[i_t])
        
    avg_test_acc = sum(test_accuracies) / (len(test_accuracies) )
    avg_train_acc = sum(train_accuracies) / (len(train_accuracies) )
    
    Average_test_sensitivities = []
    Average_test_rejections = []
    Average_train_sensitivities = []
    Average_train_rejections = []
    for t in sensitivity_thresholds:
        avg_sens = sum(test_sensitivities[t]) / len(test_sensitivities[t])
        avg_reject = sum(test_rejections[t]) / len(test_rejections[t])
        Average_test_sensitivities.append(avg_sens)
        Average_test_rejections.append(avg_reject)
        avg_sens = sum(train_sensitivities[t]) / len(train_sensitivities[t])
        avg_reject = sum(train_rejections[t]) / len(train_rejections[t])
        Average_train_sensitivities.append(avg_sens)
        Average_train_rejections.append(avg_reject)
        # print(f"Threshold {t:.2f} -> Sensitivity: {avg_sens:.2%} | Rejection rate: {avg_reject}")

    Average_test_sensitivities = np.array(Average_test_sensitivities)
    Average_test_rejections = np.array(Average_test_rejections)
    Average_train_sensitivities = np.array(Average_train_sensitivities)
    Average_train_rejections = np.array(Average_train_rejections)

    return Average_test_sensitivities, Average_test_rejections,Average_train_sensitivities, Average_train_rejections, avg_test_acc, avg_train_acc


def architectural_search_run_multirun(n_runs=3, num_epochs = 10):
    dataset_folder = root / "Features"
    
    batch_size = 10
    training_dataset = FeatureDataset(dataset_folder / "train")
    test_dataset = FeatureDataset(dataset_folder / "test")
    
    # Configurations
    tails = ['sigmoid'] # Add 'simple' if needed
    dropouts = [0, 0.5]
    architectures = [
        [128, 64, 32, 16], # Net1
        [64, 32, 16, 8],   # Net2
        [64, 32, 16],      # Net3
        [32, 16, 8],       # Net4
        [32, 16],          # Net5
        [16, 8],           # Net6
        [8, 4]             # Net7
    ]
    
    # Naming map for the LaTeX table later
    arch_map = {str(a): f"Net{i+1}" for i, a in enumerate(architectures)}

    # Fixed X-axis for consistency across runs
    sensitivity_thresholds = np.linspace(0, 0.5, 50) 
    
    summary_data = [] # For CSV
    curves_data = {}  # For NPZ

    print(f"Starting Search with {n_runs} runs per config...")

    for arch, tail, drop in itertools.product(architectures, tails, dropouts):
        arch_str = str(arch)
        config_name = f"{arch_map[arch_str]}-{tail}-drop{drop}"
        print(f"\n=== Processing: {config_name} ===")

        # Temporary storage for runs
        run_test_accs = []
        run_train_accs = []
        run_test_sens = [] # Shape will be (N_runs, 50)
        run_test_rejs = []
        
        # Calculate FLOPs once (architecture doesn't change)
        dummy_model = RejHeadNet(input_features=16, hidden_layers=arch, 
                               dropout_prob=drop, tail=tail).to(device)
        info = summary(dummy_model, input_size=(1, 16), verbose=0)
        flops = 2 * info.total_mult_adds

        # --- N-Runs Loop ---
        for run_id in range(n_runs):
            # Re-init loaders to ensure shuffling randomness each run
            train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = RejHeadNet(input_features=16, hidden_layers=arch, 
                            dropout_prob=drop, tail=tail).to(device)
            
            criterion = nn.CrossEntropyLoss() if tail == 'simple' else nn.BCELoss()

            # Train
            train(model, train_loader, test_loader, criterion=criterion, 
                  epochs=num_epochs, sensitivity_thresholds=[], testing=False)
            
            # Evaluate (Get curves)
            # Ensure evaluate_model returns arrays matching len(sensitivity_thresholds)
            t_sens, t_rejs, tr_sens, tr_rejs, t_acc, tr_acc = evaluate_model(
                model, sensitivity_thresholds, train_loader, test_loader
            )

            run_test_accs.append(t_acc)
            run_train_accs.append(tr_acc)
            run_test_sens.append(t_sens)
            run_test_rejs.append(t_rejs)

        # --- Aggregate Results (Mean +/- Std) ---
        # 1. Scalars
        avg_test_acc = np.mean(run_test_accs)
        std_test_acc = np.std(run_test_accs)
        avg_train_acc = np.mean(run_train_accs)
        
        # 2. Curves (Axis 0 is the run dimension)
        # Result shape: (50,)
        mean_test_sens = np.mean(run_test_sens, axis=0) 
        std_test_sens  = np.std(run_test_sens, axis=0)
        min_test_sens  = np.min(run_test_sens, axis=0)
        max_test_sens  = np.max(run_test_sens, axis=0)
        mean_test_rejs = np.mean(run_test_rejs, axis=0)
        std_test_rejs  = np.std(run_test_rejs, axis=0)
        min_test_rejs  = np.min(run_test_rejs, axis=0)
        max_test_rejs  = np.max(run_test_rejs, axis=0)

        # --- Save Summary to List ---
        summary_data.append({
            "config_id": config_name,
            "net_name": arch_map[arch_str],
            "architecture": arch_str,
            "tail": tail,
            "dropout": drop,
            "flops": flops,
            "test_acc_mean": avg_test_acc,
            "test_acc_std": std_test_acc,
            "train_acc_mean": avg_train_acc
        })

        # --- Save Curves to Dict ---
        # We save using the config_name as a key prefix
        curves_data[f"{config_name}_test_sens_mean"] = mean_test_sens
        curves_data[f"{config_name}_test_sens_std"] = std_test_sens
        curves_data[f"{config_name}_test_sens_min"] = min_test_sens
        curves_data[f"{config_name}_test_sens_max"] = max_test_sens
        curves_data[f"{config_name}_test_rejs_mean"] = mean_test_rejs
        curves_data[f"{config_name}_test_rejs_std"] = std_test_rejs
        curves_data[f"{config_name}_test_rejs_min"] = min_test_rejs
        curves_data[f"{config_name}_test_rejs_max"] = max_test_rejs
        # (Add train curves if needed)

    # --- Final Saving ---
    # 1. Save Scalar Summary as CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(root / "search_summary.csv", index=False)
    
    # 2. Save Curves as Compressed Numpy
    # We also save the thresholds so we know the X-axis later
    np.savez_compressed(root / "search_curves.npz", 
                        thresholds=sensitivity_thresholds, 
                        **curves_data)


def main():
    architectural_search_run_multirun(
        n_runs=10,
        num_epochs=10,
    )

if __name__ == '__main__':
    main()