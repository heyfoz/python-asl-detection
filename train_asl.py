import datetime # For timestamp creation
import os # For file and directory operations
import tensorflow as tf # For TensorFlow deep learning framework operations
from tensorflow.keras.regularizers import Regularizer, L2 # Regularization for neural networks
from tensorflow.keras.models import Sequential # Sequential model for stacking layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization # Neural network layers
from tensorflow.keras.optimizers import Adam # Training model optimizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau # For callbacks and checkpoint saving
import matplotlib.pyplot as plt # For graph plotting
import numpy as np # Numerical operations
import pandas as pd # Data manipulation and analysis

# Define your data directories and model parameters
data_dir = '/path/to/training/dir'
val_data_dir = '/path/to/validation/dir'
model_dir = '/path/to/model/dir'
trained_model_filename = 'filename.keras'
learning_curves_path = os.path.join(model_dir, 'model_learning_curves.png')
batch_size = 25
img_height, img_width = 200, 200
quick_steps_per_epoch = 10 # Adjust  value as needed for a quick test of multiepoch functionality

def plot_learning_curves(history, save_path):
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Create the figure with three subplots: two for accuracy and loss, and one for the table
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot training & validation accuracy values
    axes[0].plot(epochs, history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_xticks(epochs)
    axes[0].legend(loc='lower right')

    # Add padding between plots
    axes[0].margins(y=0.3)

    # Plot training & validation loss values
    axes[1].plot(epochs, history['loss'], label='Training Loss', color='red', linestyle='-', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_xticks(epochs)
    axes[1].legend(loc='upper right')

    # Add padding between plots
    axes[1].margins(y=0.3)
    
    # Create a dataframe for the table
    data = {
        'Epoch': epochs,
        'Accuracy': [f"{val:.4f}" for val in history['accuracy']],
        'Loss': [f"{val:.4f}" for val in history['loss']],
        'Val Accuracy': [f"{val:.4f}" for val in history['val_accuracy']],
        'Val Loss': [f"{val:.4f}" for val in history['val_loss']]
    }
    df = pd.DataFrame(data)
    
    # Add a table below the plots
    cell_text = []
    for row in range(len(df)):
        cell_text.append(df.iloc[row].tolist())
    table = axes[2].table(cellText=cell_text, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    table_title = axes[2].set_title('Epoch History', fontsize=14)  # Add title to the table
    
    # Thicken table edges and add padding inside the cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(2)
        cell.set_height(0.1)  # Add vertical padding inside cells
        cell.set_width(0.2)  # Add horizontal padding inside cells

    # Adjust the padding/margins
    axes[2].axis('off')
    axes[2].margins(x=0.1)  # Add side padding/margins for the table

    # Adjust layout and save the plot
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.6, top=0.9)  # Set spacing between subplots and top margin

    # Reduce vertical space between the second graph and the table
    plt.subplots_adjust(hspace=0.6)
    
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to prevent displaying it in the console

# Define a function for random zoom
def random_zoom(image, label, zoom_range=(-0.1, 0.1)):
    # Generate a random zoom factor
    zoom = np.random.uniform(zoom_range[0], zoom_range[1])
    # Zoom the image
    image = tf.image.resize_with_crop_or_pad(image, 
                                             int(img_height * (1 + zoom)), 
                                             int(img_width * (1 + zoom)))
    image = tf.image.resize(image, [img_height, img_width])
    return image, label

# Define a function for data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

# Load the training data using image_dataset_from_directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',  # Ensure labels are one-hot encoded
).map(lambda x, y: (x / 255.0, y))  # Rescale images

# Load the validation data using image_dataset_from_directory
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',  # Ensure labels are one-hot encoded
).map(lambda x, y: (x / 255.0, y))  # Rescale images

# Apply augmentation to the training dataset
train_ds = train_ds.map(random_zoom, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # New augmentation step
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Load the most recent model if it exists, otherwise create a new one
trained_model_path = os.path.join(model_dir, trained_model_filename)
try:
    model = tf.keras.models.load_model(trained_model_path)
    print(f"Loaded model from {trained_model_path}")
except ValueError as e:
    print(f"Could not load model from {trained_model_path}: {e}")
    print("Training model from scratch.")

    model = Sequential([
        # Initial image shape layer using 3 to specifiy color channels (RGB)
        Input(shape=(img_height, img_width, 3)), 
        # First convolutional layer with 32 filters of size 3x3,  ReLU activation, and L2 regularization
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(0.001)),
        # First max pooling layer to reduce spatial dimensions by a factor of 2
        MaxPooling2D(2, 2),
        # Initial dropout layer to prevent overfitting
        Dropout(0.25),
        # Second convolutional layer with 64 filters of size 3x3 and ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        # Second max pooling layer to further reduce spatial dimensions by a factor of 2
        MaxPooling2D(2, 2),
        # Flatten layer to convert 3D feature maps to 1D feature vectors
        Flatten(),
        # Fully connected layer with 128 units and ReLU activation
        Dense(128, activation='relu'),
        # Final dropout layer to prevent overfitting 
        Dropout(0.5),
        # Output layer with softmax activation for 26 classes (1 per letter)
        Dense(26, activation='softmax') 
    ])

    # Compile the model with reduced learning rate (default is 0.001)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

class PlotLearningCurves(Callback):
    def __init__(self, save_path, model_dir):
        super(PlotLearningCurves, self).__init__()
        self.save_path = save_path
        self.model_dir = model_dir
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'training_time': []}

    def on_epoch_end(self, epoch, logs=None):
        # Append current epoch metrics to history
        self.history['accuracy'].append(logs['accuracy'])
        self.history['val_accuracy'].append(logs['val_accuracy'])
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        
        # Extract training time for the epoch from the status bar
        training_time_str = self.model.history.history['elapsed_time'][epoch] if 'elapsed_time' in self.model.history.history else 'NA'
        self.history['training_time'].append(training_time_str)

        # Plot and save learning curves
        plot_learning_curves(self.history, save_path=self.save_path)
        
        # Get the current date in mm-dd-yyyy format
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")

        # Update checkpoint filename with epoch, accuracy, and date
        accuracy_str = f"val-acc-{logs['val_accuracy']:.4f}"
        checkpoint_filename = f'asl-detect-test-{epoch+1:02d}-{accuracy_str}-{timestamp}.keras'
        checkpoint_path = os.path.join(self.model_dir, checkpoint_filename)
        self.checkpoint_filepath = checkpoint_path  # Update filepath for ModelCheckpoint callback
        self.model.save(checkpoint_path)  # Save model with updated filename
        # Print the updated checkpoint filepath
        print()
        print(f"Checkpoint .keras file saved: {self.checkpoint_filepath}")

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='max', verbose=1)
plot_curves_callback = PlotLearningCurves(learning_curves_path, model_dir)

# Start training
final_total_epochs = 10

# Train the model for the specified number of epochs
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=final_total_epochs,
    callbacks=[plot_curves_callback],
    verbose=1  # Setting verbose to 1 to display the progress bar
)

# Get the final validation accuracy from the training history
final_val_accuracy = history.history['val_accuracy'][-1]

# Generate the current datetime string
final_time = datetime.datetime.now()
final_time_str = final_time.strftime("%m-%d-%Y-%H-%M")

# Define the final model filename with the datetime string
final_model_filename = f'asl-detect-final-test-val-acc-{final_val_accuracy:.4f}-{final_time_str}.keras'
final_model_path = os.path.join(model_dir, final_model_filename)

model.save(final_model_path)
