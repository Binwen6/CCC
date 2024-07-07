import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(file_path, columns_to_drop=None):
    df = pd.read_csv(file_path)
    X = df.loc[:, 'X1':'X24'] if 'X24' in df.columns else df.loc[:, 'X1':'X14']
    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)
    y = df['Y(1=default, 0=non-default)']
    return train_test_split(X, y, test_size=0.3, random_state=42)

class BPNN_CBCE(nn.Module):
    def __init__(self, input_features, output_features):
        super(BPNN_CBCE, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2)) for _ in range(4)],
            nn.Linear(256, output_features)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(model, criterion, optimizer, train_loader, epochs=500):
    model.train()
    history = {'loss': []}
    for epoch in range(epochs):
        with tqdm(total=len(train_loader.dataset)) as pbar:
            running_loss = 0.0
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update(data.size(0))
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            history['loss'].append(epoch_loss)
            
            if epoch % 50 == 0:
                pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return history

def evaluate_model_with_metrics(model, test_loader, y_test):
    model.eval()
    outputs_all = torch.empty(0).to(y_test.device)
    labels_all = torch.empty(0).to(y_test.device)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            outputs_all = torch.cat((outputs_all, outputs), 0)
            labels_all = torch.cat((labels_all, labels), 0)
    
    _, predicted = torch.max(outputs_all, 1)
    
    fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), outputs_all[:, 1].cpu().numpy())
    roc_auc = auc(fpr, tpr)
    
    cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())
    report = classification_report(y_test.cpu().numpy(), predicted.cpu().numpy())
    
    print(f'Accuracy: {100 * (cm[1, 1] / np.sum(cm))}%')
    print(f'AUC-ROC: {roc_auc:.2f}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')

    plt.figure()
    plt.plot(fpr, tpr, color='lightseagreen', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='darkslategrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig('../Data/roc_curve.png')
    plt.show()

def run_experiment(file_path, columns_to_drop=None, save_suffix=''):
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, columns_to_drop)

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_features = X_train.shape[1]
    model = BPNN_CBCE(input_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, criterion, optimizer, train_loader, epochs=500)

    plt.plot(history['loss'], color='olivedrab', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('损失曲线')
    plt.legend()
    plt.savefig(f'../Data/loss_curve{save_suffix}.png')
    plt.show()

    evaluate_model_with_metrics(model, test_loader, y_test)

# Run experiments
run_experiment("../Data/data1.csv", columns_to_drop=['X13', 'X15', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23'])
run_experiment("../Data/data2.csv", save_suffix='_2')