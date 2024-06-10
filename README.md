# ASL Detection and TensorFlow Image Classification Model Training Project

<p align="center">
  <img src="/misc/ASL N Detection.png" alt="ASL Letter N Detected">
</p>

This project involves training a keras image recognition model to predict American Sign Language (ASL) letters displayed via a video stream. The project consists of three main components:

1. [mediapipe_preprocess.py](/preprocessing/mediapipe_preprocess.py): A script to preprocess images to have a black background, hand centered on the screen in a consistent format.
2. [train_asl.py](train_asl.py): A script to train the ASL alphabet model using TensorFlow.
3. [detect_asl.py](detect_asl.py): A live video stream application that detects the ASL alphabet letter displayed using the trained model.

## Neural Network Architecture Overview

This section provides insights into the architecture of the neural network utilized during training. It was designed to detect and classify American Sign Language (ASL) alphabet gestures. Key components:

1. **Input Layer**: The neural network takes as input grayscale images of ASL gestures. Each image is resized to a standard dimension of 200x200 pixels.

2. **Convolutional Layers**: The neural network begins with two convolutional layers. The first layer consists of 32 filters of size 3x3, followed by a ReLU activation function. Subsequently, a max-pooling layer reduces spatial dimensions by a factor of 2. The second convolutional layer comprises 64 filters of size 3x3, also followed by a ReLU activation function and max-pooling.

3. **Flatten Layer**: After the convolutional layers, a flatten layer converts the 3D feature maps into 1D feature vectors, preparing the data for further processing.

4. **Fully Connected Layer**: The flattened features are then fed into a fully connected layer with 128 units, each employing a ReLU activation function.

5. **Output Layer**: The final layer consists of 26 units, representing each letter in the American Sign Language alphabet. A softmax activation function is applied to produce probability distributions over these classes.

6. **Dropout**: Dropout layers are employed after the first convolutional layer and the fully connected layer to mitigate overfitting during training.

7. **Model Compilation**: The model is compiled using the Adam optimizer and categorical crossentropy loss function, suitable for multi-class classification tasks.

This architecture is trained using a combination of training and validation datasets, with data augmentation techniques applied to enhance model generalization. Throughout training, model checkpoints are saved as .keras files, enabling the restoration of training progress and the selection of the best-performing model. Additionally, comprehensive visualizations, including graphs depicting accuracy over epochs and loss over epochs, along with a table summarizing the training data, are rendered to a single PNG file. Finally, the trained model is saved for future use or deployment.

## Project Structure

- [training](training): Directory containing subdirectories `A` through `Z`, where training images will be stored.
- [validatio](/validation): Directory containing subdirectories `A` through `Z`, where validation images will be stored.
- [preprocessing](/preprocessing): Directory containing preprocessing files used to manipulate data before training.
- [misc](/misc): Directory cointaining misc utility files and a training terminal output example. 

## Getting Started

### Prerequisites

- Python 3.9.x - core programming language
- TensorFlow 2.15.0 - deep learning framework
- OpenCV (CV2) 4.9.2 - computer vision library
- MediaPipe 0.10.9 - detecting and cropping hands
- MatPlotLib 3.9.0 - plotting training history values
- Pandas 2.2.2 - formatting plots history table

### Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/ffm5113/python-asl-detection.git
   cd python-asl-detection
   ```

## Dataset
The dataset of ASL images was taken from the [Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) dataset by Lexset. After mediapipe preprocessing and removal of unwanted images, the model learned from 750 images per letter in training and used 75 images per letter in validation. You can download the dataset and move the images in the respective subdirectories under 'training' and 'test' (renamed validation in this project).

### Preprocessing
To preprocess images using MediaPipe, run the following command in the preprocessing directory:

```bash
python mediapipe_preprocess.py
```

### Training
To train the ASL alphabet model, run the following command in the project directory:
```bash
nohup python train_asl.py &
```
This script will use the images stored in the training and validation directories to train the keras model. The nohup command will execute the training script in the background, and the output will be redirected to a file named nohup.out. You can view the output file in the current directory using:
```bash
cat nohup.out
```

Or to view the last 10 lines of the nohup.out file:
```bash
tail -n 10 nohup.out
```

Using this data preprocessing approach, over 97% validation accuracy and a loss value under .15 were achieved by the end of training. 

| Epoch | Accuracy | Loss   | Val_Accuracy | Val_Loss |
|-------|----------|--------|--------------|----------|
| 1     | 0.7746   | 0.8365 | 0.9477       | 0.2380   |
| 2     | 0.9125   | 0.3420 | 0.9574       | 0.1841   |
| 3     | 0.9372   | 0.2506 | 0.9646       | 0.1679   |
| 4     | 0.9476   | 0.2028 | 0.9636       | 0.1638   |
| 5     | 0.9565   | 0.1655 | 0.9682       | 0.1620   |
| 6     | 0.9630   | 0.1420 | 0.9703       | 0.1523   |
| 7     | 0.9657   | 0.1315 | 0.9718       | 0.1455   |
| 8     | 0.9678   | 0.1166 | 0.9713       | 0.1469   |
| 9     | 0.9711   | 0.1044 | 0.9703       | 0.1503   |
| 10    | 0.9742   | 0.0910 | 0.9728       | 0.1410   |

## Model Learning Curves

![Model Learning Curves](/misc/model_learning_curves.png)

The above graph illustrates the custom learning curves rendered to model_learning_cruves.png during the training script for the ASL alphabet model. It shows the training and validation accuracy and loss over epochs, providing valuable insights into the model's performance and convergence.

## Detecting ASL Alphabet Letters
To start the live video stream application and detect ASL alphabet letters, run the following command from the project directory:
```bash
python detect_asl.py
```

## Video Demonstration
Watch a demonstration of the ASL detection system v1 in action on LinkedIn: [ASL Detection Demonstration](https://www.linkedin.com/posts/moulinf_ai-artificialintelligence-data-activity-7150833680297406465-A43y?utm_source=share&utm_medium=member_desktop)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Conclusion

In this project, we developed an intermediate ASL alphabet detection and classification system using TensorFlow. Through training and validation processes, the model achieved promising results, as demonstrated by the high validation accuracy up to 97%. The architecture of the neural network, along with the incorporation of data augmentation techniques and model checkpoints, contributed to enhancing model generalization and performance.

The automated visualizations, including the plotted graph of learning curves, offer valuable insights into the training progress and performance metrics. These visualizations aid in understanding the model's behavior, identifying potential areas for improvement, and making informed decisions during the training process.

With the live video stream application for detecting ASL alphabet letters, this project showcases the practical application of deep learning in real-time gesture recognition tasks. Moving forward, further refinements and optimizations could be explored to enhance the model's accuracy and efficiency, potentially extending its applications to broader domains within the field of computer vision and human-computer interaction.
