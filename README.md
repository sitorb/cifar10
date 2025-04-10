# CIFAR-10 Image Classification with CNN

This project demonstrates image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Code Structure and Logic

1. **Data Loading and Preprocessing:**
   - The code starts by loading the CIFAR-10 dataset using `keras.datasets.cifar10.load_data()`.
   - The images are then preprocessed by scaling pixel values to the range of 0-1 by dividing by 255.

2. **Data Augmentation:**
   - `ImageDataGenerator` is used to augment the training data with random width and height shifts and horizontal flips.
   - This helps to increase the diversity of the training data and prevent overfitting.

3. **CNN Model Architecture:**
   - A sequential CNN model is built using `keras.models.Sequential`.
   - It consists of multiple convolutional layers with ReLU activation, max pooling layers for downsampling, a flattening layer to convert the output to a vector, and dense layers for classification.
   - Dropout is added for regularization to further prevent overfitting.

4. **Model Compilation and Training:**
   - The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function (suitable for multi-class classification), and accuracy metric.
   - The model is trained using `cnn.fit()` with the augmented training data for a specified number of epochs.

5. **Model Evaluation and Prediction:**
   - The model's performance is evaluated on the test data using `cnn.evaluate()`.
   - Predictions are made using `cnn.predict()`.
   - A classification report is generated to provide detailed performance metrics, including precision, recall, F1-score, and support for each class.

6. **Visualization:**
   - A `plot_sample()` function is defined to display images with their corresponding labels.
   - This function is used to visualize predictions and inspect the model's performance on individual samples.

## Technology and Algorithms

- **TensorFlow/Keras:** A popular deep learning framework used for building and training neural networks.
- **CNN (Convolutional Neural Network):** A deep learning algorithm specifically designed for image recognition and classification.
- **Data Augmentation:** Techniques to artificially increase the size and diversity of the training data.
- **Adam Optimizer:** An optimization algorithm used to update the model's weights during training.
- **Sparse Categorical Cross-entropy:** A loss function used for multi-class classification.
- **Classification Report:** A performance evaluation metric that provides detailed information about the model's accuracy for each class.
- **Matplotlib:** A plotting library used for visualizing data and results.

![image](https://github.com/user-attachments/assets/1d35c20c-43db-4a4f-b704-a3bc8ebfb13c)


## Usage

- Install required libraries: `tensorflow`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`
- Download the CIFAR-10 dataset if not already present.
- Run the code cells in the notebook to load the data, train the model, evaluate its performance, and make predictions.

## Conclusion

This project demonstrates how to build and train a CNN for image classification using the CIFAR-10 dataset. It covers data preprocessing, model architecture, training, evaluation, and visualization steps. You can further explore and extend this project by experimenting with different model architectures, hyperparameters, or data augmentation techniques to achieve improved performance.
