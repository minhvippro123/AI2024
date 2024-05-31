import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


# Function to generate synthetic images of shapes
def generate_shape(shape, size=28):
    img = np.zeros((size, size))
    if shape == 'square':
        img[7:21, 7:21] = 1
    elif shape == 'circle':
        rr, cc = np.ogrid[:size, :size]
        mask = (rr - size / 2) ** 2 + (cc - size / 2) ** 2 <= (size / 3) ** 2
        img[mask] = 1
    elif shape == 'triangle':
        for i in range(size // 2):
            img[size // 2 - i, 7 + i:21 - i] = 1
    return img


# Generate dataset
def generate_dataset(num_samples_per_class=1000):
    shapes = ['square', 'circle', 'triangle']
    X = []
    y = []
    for i, shape in enumerate(shapes):
        for _ in range(num_samples_per_class):
            X.append(generate_shape(shape))
            y.append(i)
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = np.array(y)
    return X, y


# Generate data
X, y = generate_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")


# Plot some test samples with their predicted labels
def plot_sample_predictions(X, y, model, num_samples=10):
    indices = np.random.choice(len(X), num_samples)
    X_sample, y_sample = X[indices], y[indices]
    y_pred = np.argmax(model.predict(X_sample), axis=-1)
    shapes = ['square', 'circle', 'triangle']

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_sample[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {shapes[y_sample[i]]}\nPred: {shapes[y_pred[i]]}")
        plt.axis('off')
    plt.show()


plot_sample_predictions(X_test, y_test, model)