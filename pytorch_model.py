import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class PyTorchNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model_torch = PyTorchNN(X_train.shape[1], 16)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.01)
for epoch in range(1000):
    outputs = model_torch(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
with torch.no_grad():
    y_pred_torch = model_torch(X_test_tensor)
    y_pred_labels = (y_pred_torch.numpy() > 0.5).astype(int)

print("PyTorch Accuracy:", accuracy_score(y_test, y_pred_labels))
print("PyTorch F1 Score:", f1_score(y_test, y_pred_labels))
print("PyTorch Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))
precision, recall, _ = precision_recall_curve(y_test, y_pred_labels)
print("PyTorch PR-AUC:", auc(recall, precision))
