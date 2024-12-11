import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Modeling
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Evaluation
# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=8, verbose=1)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))

# Step 6: Deployment
# Save the model
model.save("iris_classifier_model.h5")
print("Model saved as 'iris_classifier_model.h5'")
