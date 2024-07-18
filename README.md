# Music Genre Classification

This project includes classifiers for music genres. The dataset is organized into the `dataset` directory, which contains ten subdirectories with different music genre, each with 50 WAV files. The project utilizes three different classification models and employs five feature extraction methods.

## Audio Features Used

1. Mel-Spectrogram
2. Mel-Frequency Cepstral Coefficients (MFCC)
3. Spectral Contrast
4. Zero Crossing Rate
5. Chroma STFT

## Classification Models

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Random Forest

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
4. Customize the classifier and feature extraction method to use in the 'Start Training' section.

## Results and Analysis

|                    | KNN<div style="width:100px"> | SVM<div style="width:100px"> | Random Forest<div style="width:100px"> |
| :----------------- | :--------------------------: | :--------------------------: | :------------------------------------: |
| Mel-Spectrogram    |            0.296             |            0.337             |                 0.366                  |
| MFCC               |          **0.396**           |          **0.562**           |               **0.534**                |
| Spectral Contrast  |            0.352             |             0.43             |                 0.434                  |
| Zero Crossing Rate |            0.216             |            0.238             |                 0.216                  |
| Chroma STFT        |            0.238             |            0.272             |                 0.234                  |

The results show that the MFCC feature achieves the highest accuracy across all classification models.
