import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for model serialization

def extract_features_from_dataset(dataset_path):
    """
    Extract MFCC features from audio files in the dataset.

    Parameters:
    - dataset_path (str): Path to the dataset directory containing audio files.

    Returns:
    - features (list): List of extracted features for each audio file.
    - labels (list): List of labels corresponding to each extracted feature.
    """
    features = []
    labels = []

    # Iterate through each file in the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    # Load audio file
                    audio, sample_rate = librosa.load(file_path, sr=None)
                    
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                    
                    # Calculate mean MFCCs
                    mfccs_mean = np.mean(mfccs, axis=1)
                    
                    # Append features and label to lists
                    features.append(mfccs_mean)
                    labels.append(os.path.basename(root))  # Assuming folder name is the label
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    return features, labels

# Example usage:
dataset_path = r"C:\Users\user\Downloads\heart_sound\train"  # Path to the training dataset
features_train, labels_train = extract_features_from_dataset(dataset_path)
print("Number of training audio files processed:", len(features_train))

# Path to the validation dataset
validation_data_path = r"C:\Users\user\Downloads\heart_sound\val"
features_val, labels_val = extract_features_from_dataset(validation_data_path)
print("Number of validation audio files processed:", len(features_val))

# Initialize the SVM classifier
classifier = SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(features_train, labels_train)

# Save the trained SVM model
joblib.dump(classifier, "your_model.pkl")

# Predict labels for the validation set
labels_pred_val = classifier.predict(features_val)

# Calculate accuracy on validation set
accuracy_val = accuracy_score(labels_val, labels_pred_val)
print("Validation Accuracy:", accuracy_val)
