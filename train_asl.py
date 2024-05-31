import os  # For file and directory operations
import tensorflow as tf  # For TensorFlow operations

from tensorflow.keras.models import Sequential  # Import Sequential model class from Keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout  # Import commonly used layers from Keras
from tensorflow.keras.optimizers import Adam  # Import the Adam optimizer from Keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # Import callbacks for saving the model and early stopping during training

# Define your data directories
train_data_dir = '/path/to/training/data/directory'  # Training data directory
val_data_dir = '/path/to/validation/data/directory'  # Validation data directory

# Model parameters
batch_size = 25  # Batch size for training
img_height, img_width = 200, 200  # Image dimensions

# Convert to 3-channel grayscale image
def to_3channel_gray(img):
    gray = tf.image.rgb_to_grayscale(img)  # Convert image to grayscale
    return tf.repeat(gray, 3, axis=-1)  # Repeat grayscale image to make it 3-channel

# Dataset from directory with augmentation for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # Path training data directory
    train_data_dir,
    # OptionalSplitting validation data from training data, 20% of the data is used for validation
    # validation_split=0.2,
    subset="training",
    # Seed for reproducibility / consistency across multiple runs
    seed=123,
    # Resize images to a specific size
    image_size=(img_height, img_width),
    # Batch size for training
    batch_size=batch_size
).map(
    # Applying grayscale conversion function to input images
    lambda x, y: (to_3channel_gray(x), y)
)

# Dataset from directory for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # Path to validation data directory
    val_data_dir,
    # Split training data from validation data, 20% of the data is added to val directory
    # validation_split=0.2,
    subset="validation",
    # Seed for reproducibility / consistency across multiple runs
    seed=123,
    # Resize images to a specific size
    image_size=(img_height, img_width),
    # Batch size for validation
    batch_size=batch_size
).map(
    # Applying grayscale conversion function to input images
    lambda x, y: (to_3channel_gray(x), y)
)

# Apply data augmentation to the training dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),  # Randomly rotate images
    tf.keras.layers.RandomWidth(0.1),  # Randomly shift images horizontally
    tf.keras.layers.RandomHeight(0.1),  # Randomly shift images vertically
    tf.keras.layers.RandomZoom(0.1),  # Randomly zoom images
])

# Function to augment images
def augment(image, label):
    return data_augmentation(image), label

# Augment the training dataset
train_ds = train_ds.map(augment)

# Prefetch data for performance optimization
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(train_ds) // batch_size
validation_steps = len(val_ds) // batch_size

# Model checkpoint and early stopping callbacks
checkpoint_path = '/path/to/model/directory/asl-detection-{epoch:02d}-{accuracy:.2f}.keras'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
# Optional stopping when validation loss stop improving: early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Load or create the model
model_path = '/path/to/pretrained/model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully. Resuming training.")
else:
    print("Model file not found. Starting training from scratch.")
    # Define the model
    model = tf.keras.Sequential([
        # First convolutional layer with 32 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        # First max pooling layer to reduce spatial dimensions by a factor of 2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.25),
        # Second convolutional layer with 64 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Second max pooling layer to further reduce spatial dimensions by a factor of 2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten layer to convert 3D feature maps to 1D feature vectors
        tf.keras.layers.Flatten(),
        # Fully connected layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        # Output layer with softmax activation for 26 classes (1 per letter)
        tf.keras.layers.Dense(26, activation='softmax')
        # Dropout layer to prevent overfitting (commented out)
        # tf.keras.layers.Dropout(0.5),
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Compile with Adam optimizer and categorical crossentropy loss

# Start training
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[checkpoint]  # Add callbacks for model checkpointing
    )

# Save the final model
final_model_dir = '/path/to/save/final/model/directory'
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)
model.save(os.path.join(final_model_dir, 'final_model.keras'), save_format='tf')
