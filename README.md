# ASL Detection and TensorFlow Image Classification Model Training Project

<p align="center">
  <img src="https://github.com/ffm5113/python-asl-detection/blob/main/misc/ASL%20N%20Detection.png" alt="ASL Letter N Detected">
</p>
https://github.com/ffm5113/python-asl-detection/blob/main/misc/ASL%20N%20Detection.png

This project involves training a keras image recognition model to predict American Sign Language (ASL) letters displayed via a video stream. The project consists of three main components:

1. [mediapipe_preprocess.py](/preprocessing/mediapipe_preprocess.py): A script to preprocess images to have a black background, hand centered on the screen in a consistent format.
2. [train_asl.py](train_asl.py): A script to train the ASL alphabet model using TensorFlow.
3. [detect_asl.py](detect_asl.py): A live video stream application that detects the ASL alphabet letter displayed using the trained model.

## Neural Network Architecture Overview

This section provides insights into the architecture of the neural network utilized during training. The neural network architecture employed in this project is designed to detect and classify American Sign Language (ASL) gestures. Below is an overview of the key components and layers comprising the neural network:

1. **Input Layer**: The neural network takes as input grayscale images of ASL gestures. Each image is resized to a standard dimension of 200x200 pixels.

2. **Convolutional Layers**: The neural network begins with two convolutional layers. The first layer consists of 32 filters of size 3x3, followed by a ReLU activation function. Subsequently, a max-pooling layer reduces spatial dimensions by a factor of 2. The second convolutional layer comprises 64 filters of size 3x3, also followed by a ReLU activation function and max-pooling.

3. **Flatten Layer**: After the convolutional layers, a flatten layer converts the 3D feature maps into 1D feature vectors, preparing the data for further processing.

4. **Fully Connected Layer**: The flattened features are then fed into a fully connected layer with 128 units, each employing a ReLU activation function.

5. **Output Layer**: The final layer consists of 26 units, representing each letter in the American Sign Language alphabet. A softmax activation function is applied to produce probability distributions over these classes.

6. **Dropout**: Dropout layers are employed after the first convolutional layer and the fully connected layer to mitigate overfitting during training. However, the dropout layer after the fully connected layer is commented out, allowing flexibility for experimentation.

7. **Model Compilation**: The model is compiled using the Adam optimizer and categorical crossentropy loss function, suitable for multi-class classification tasks.

This architecture is trained using a combination of training and validation datasets, with data augmentation techniques applied to enhance model generalization. Throughout training, model checkpoints are saved, allowing for the restoration of training progress and the selection of the best-performing model. Finally, the trained model is saved for future use or deployment.

## Project Structure

- `training/`: Directory containing subdirectories `A` through `Z`, where training images will be stored.
- `validation/`: Directory containing subdirectories `A` through `Z`, where validation images will be stored.

## Getting Started

### Prerequisites

- Python 3.9.x
- TensorFlow 2.15.0
- OpenCV (CV2) 4.9.2
- MediaPipe 0.10.9

### Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/ffm5113/python-asl-detection.git
   cd yourrepository
   ```

## Dataset
The dataset of ASL images is taken from the [Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) dataset by Lexset. 
You can download the dataset and move the images in the respective subdirectories under 'Training' and 'Test' (for validation).

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

With this data preprocessing approach, I was able to achieve 96% validation accuracy in 4 hours of training (10 epochs). Note that these results were achieved using a deprecated ImageDataGenerator for image augmentation, but the same image preprocessing and configurations can be applied.

| Epoch | Accuracy | Loss   | Val_Accuracy | Val_Loss |
|-------|----------|--------|--------------|----------|
| 1     | 0.3991   | 2.0398 | 0.8337       | 0.6126   |
| 2     | 0.7804   | 0.7374 | 0.9021       | 0.3501   |
| 3     | 0.8535   | 0.5157 | 0.9179       | 0.2564   |
| 4     | 0.8854   | 0.3921 | 0.9347       | 0.2191   |
| 5     | 0.9001   | 0.3505 | 0.9442       | 0.1947   |
| 6     | 0.9194   | 0.2975 | 0.9505       | 0.1753   |
| 7     | 0.9267   | 0.2744 | 0.9600       | 0.1438   |
| 8     | 0.9314   | 0.2576 | 0.9653       | 0.1350   |
| 9     | 0.9386   | 0.2310 | 0.9674       | 0.1317   |
| 10    | 0.9396   | 0.2132 | 0.9642       | 0.1159   |

## Detecting ASL Alphabet Letters
To start the live video stream application and detect ASL alphabet letters, run the following command from the project directory:
```bash
python detect_asl.py
```

## Video Demonstration
Watch a demonstration of the ASL detection system v1 in action on LinkedIn: [ASL Detection Demonstration](https://www.linkedin.com/posts/moulinf_ai-artificialintelligence-data-activity-7150833680297406465-A43y?utm_source=share&utm_medium=member_desktop)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
