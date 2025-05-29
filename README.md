# Cat vs Dog Image Classifier

This project implements a convolutional neural network (CNN) using PyTorch to classify images as either a cat or a dog.

## Project Overview

The project involves:

1.  **Setting up the environment**: Importing necessary libraries and setting up the device for training (GPU if available).
2.  **Data Loading and Preprocessing**: Downloading the Microsoft Cats vs Dogs dataset using Kaggle API, extracting it, and applying transformations to the images. The code also includes a step to identify and remove corrupted images from the dataset.
3.  **Dataset Splitting**: Splitting the dataset into training and testing sets.
4.  **Model Definition**: Defining a CNN architecture (`CatDogClassifier`) using PyTorch's `nn.Module`. The model includes convolutional layers, max pooling, dropout, and fully connected layers.
5.  **Training**: Training the CNN model on the training dataset using the Adam optimizer and Cross-Entropy Loss. Training loss and accuracy are tracked and printed per epoch.
6.  **Evaluation**: Evaluating the trained model's performance on the testing dataset. Test accuracy is calculated and printed per epoch.
7.  **Visualization**: Plotting the training loss and test accuracy over epochs.
8.  **Model Saving**: Saving the trained model's state dictionary.
9.  **Prediction on a Single Image**: Providing a function to load a single image, preprocess it, and use the trained model to predict its class (Cat or Dog).

## Setup and Usage

1.  **Google Colab**: This code is designed to run in Google Colab. Open a new Colab notebook.
2.  **Kaggle API**: To download the dataset, you need to have a Kaggle account and generate a Kaggle API token.
    *   Go to your Kaggle account settings.
    *   Under the "API" section, click "Create New API Token". This will download a `kaggle.json` file.
    *   Upload this `kaggle.json` file to your Google Colab environment. The code includes commands to move and secure this file.
3.  **Run the Code**: Copy and paste the code from your notebook into the Colab cells and run them sequentially.

### Code Breakdown

-   **Imports**: Essential libraries like `torch`, `torchvision`, `matplotlib`, `PIL`, `random`, `os`, `zipfile`.
-   **Device Setup**: This line checks if a GPU is available and sets the device accordingly.
  ```bash
    python device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ```
-   **Data Transformations**: This defines the transformations applied to the images, including resizing, random flips and rotations, converting to tensor, and normalization.
 ```bash
    python transform = transforms.Compose([...])
 ```
-   **Kaggle Download and Extraction**: These commands download the dataset and extract it.
 ```bash
    !mkdir -p ~/.kaggle !cp kaggle.json
    ~/.kaggle/ !chmod 600
    ~/.kaggle/kaggle.json
    !kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset
    zipfile.ZipFile('microsoft-catsvsdogs-dataset.zip').extractall('./data')
 ```
-   **Corrupted Image Filtering**: The code includes a loop to iterate through the image directories, attempt to open and load each image, and remove those that are corrupted using `PIL`'s error handling.
-   **Dataset Loading and Splitting**: `ImageFolder` loads the dataset, and `random_split` divides it into training and testing sets.
 ```bash
    python full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
 ```
-   **DataLoader**: DataLoaders are used to iterate through the datasets in batches during training and evaluation.
 ```bash
    python train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
 ```
-   **Model Definition**: The `CatDogClassifier` class defines the neural network architecture. The `_get_flatten_size` method dynamically calculates the size of the input to the first fully connected layer.
-   **Model Initialization and Training**:
-   # Training loop
  -  The model, loss function, and optimizer are initialized, and the training loop iterates through epochs and batches to update the model weights.
  -   **Evaluation**: The evaluation loop calculates the accuracy on the test set.
  -   **Plotting**: Matplotlib is used to visualize the training loss and test accuracy.
  ```bash
    python model = CatDogClassifier().to(device) criterion = nn.CrossEntropyLoss() optimizer = optim.Adam(model.parameters(), lr=0.001)
  ```
  -   **Model Saving**: Saves the trained model's parameters.
  -   **Prediction Function**: The `predict_image_class` function takes an image path, the trained model, class names, device, and transform as input, loads the image, preprocesses it, makes a prediction, and displays the image with the predicted class.
  ```bash
    python torch.save(model.state_dict(), 'cat_dog_model.pth')
  ```

## Requirements

-   Python
-   PyTorch
-   TorchVision
-   Matplotlib
-   PIL (Pillow)
-   Kaggle API token

## How to Predict on a New Image

1.  Ensure you have saved the trained model (`cat_dog_model.pth`).
2.  Upload the image you want to predict on to your Colab environment.
3.  Modify the `img_path` variable in the last cell to point to your image file.
4.  Run the last cell to see the prediction.
```bash
python img_path = "/content/your_image.jpg" # Replace with the path to your image
... (rest of the prediction code)
predict_image_class(img_path, model, class_names, device, transform_img)
```

# This README provides a comprehensive explanation of the code and how to use it.
