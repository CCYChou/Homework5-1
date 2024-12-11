import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

# Step 1: Business Understanding
# Goal: Predict the species of iris flowers based on their features.

# Step 2: Data Understanding
# Load dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(data_url, header=None, names=columns)
print(iris_data.head())

# Step 3: Data Preparation
# Encode target variable
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(iris_data["species"].values.reshape(-1, 1))

# Split data into features and labels
X = iris_data.iloc[:, :-1].values
y = y_encoded

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 4: Modeling
class IrisClassifier(LightningModule):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, torch.argmax(targets, dim=1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, torch.argmax(targets, dim=1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)
        acc = accuracy_score(targets.cpu(), preds.cpu())
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Instantiate the model
model = IrisClassifier()

# Step 5: Evaluation
trainer = Trainer(max_epochs=50)
trainer.fit(model, train_loader, test_loader)

# Evaluate on test set
trainer.test(model, test_loader)

# Step 6: Deployment
# Save the model
torch.save(model.state_dict(), "iris_classifier_model.pth")
print("Model saved as 'iris_classifier_model.pth'")
