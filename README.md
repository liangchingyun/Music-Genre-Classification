# Music Genre Classification

This project includes classifiers for music genres. The dataset is organized into a `dataset` directory, which contains ten subdirectories, each representing a different music genre with 50 WAV files. The project utilizes three different classification models and employs five feature extraction methods.

## Dataset Split

The dataset is split into training, validation, and test sets with the following proportions:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

## Training and Evaluation

I train the model for 45 epochs. At the end of each epoch, the following metrics are recorded and plotted:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These metrics, plotted against the number of epochs, help us observe and judge the training performance of the model.

Finally, I evaluate the model's performance using the Test Loss and Test Accuracy.

## Usage

1. Clone or download this project.
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```
    python main.py
    ```

## Results and Analysis
Training process metrics:

![image](https://github.com/liangchingyun/img-folder/blob/main/CNN-Classifier_result.png)


Final test results:\
Test Loss: 0.6178\
Test Accuracy: 71.78%