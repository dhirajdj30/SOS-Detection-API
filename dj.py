import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import datetime
import joblib

# Define the thresholds
THRESHOLDS = {
    'acceleration': 120,
    'rotation': 15,
    'magnetic_field': 500,
    'light': 500
}

# Load data from CSV
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Create balanced synthetic data
def generate_synthetic_data(n_samples=1000):
    synthetic_data = []
    
    # Generate non-triggering cases (FALSE)
    for _ in range(n_samples // 2):
        # Normal operating conditions (well below thresholds)
        if np.random.random() < 0.7:
            row = [
                np.random.uniform(80, THRESHOLDS['acceleration'] - 10),  # Normal acceleration range
                np.random.uniform(5, THRESHOLDS['rotation'] - 2),        # Normal rotation range
                np.random.uniform(300, THRESHOLDS['magnetic_field'] - 20), # Normal magnetic field range
                np.random.uniform(50, THRESHOLDS['light'] - 50),         # Normal light range
                0
            ]
        # Edge cases (close to but below thresholds)
        else:
            row = [
                np.random.uniform(THRESHOLDS['acceleration'] - 10, THRESHOLDS['acceleration'] - 1),
                np.random.uniform(THRESHOLDS['rotation'] - 2, THRESHOLDS['rotation'] - 0.1),
                np.random.uniform(THRESHOLDS['magnetic_field'] - 20, THRESHOLDS['magnetic_field'] - 1),
                np.random.uniform(THRESHOLDS['light'] - 50, THRESHOLDS['light'] - 1),
                0
            ]
        synthetic_data.append(row)
    
    # Generate triggering cases (TRUE)
    for _ in range(n_samples // 2):
        # Randomly choose which sensor(s) will exceed threshold
        num_exceeded = np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.07, 0.03])
        features_to_exceed = np.random.choice(
            ['acceleration', 'rotation', 'magnetic_field', 'light'],
            size=num_exceeded, replace=False
        )
        
        # Generate values
        acceleration = np.random.uniform(
            THRESHOLDS['acceleration'], THRESHOLDS['acceleration'] * 1.3
        ) if 'acceleration' in features_to_exceed else np.random.uniform(80, THRESHOLDS['acceleration'] - 1)
        
        rotation = np.random.uniform(
            THRESHOLDS['rotation'], THRESHOLDS['rotation'] * 1.3
        ) if 'rotation' in features_to_exceed else np.random.uniform(5, THRESHOLDS['rotation'] - 1)
        
        magnetic_field = np.random.uniform(
            THRESHOLDS['magnetic_field'], THRESHOLDS['magnetic_field'] * 1.3
        ) if 'magnetic_field' in features_to_exceed else np.random.uniform(300, THRESHOLDS['magnetic_field'] - 1)
        
        light = np.random.uniform(
            THRESHOLDS['light'], THRESHOLDS['light'] * 1.3
        ) if 'light' in features_to_exceed else np.random.uniform(50, THRESHOLDS['light'] - 1)
        
        synthetic_data.append([acceleration, rotation, magnetic_field, light, 1])

    df = pd.DataFrame(synthetic_data, columns=['acceleration', 'rotation', 'magnetic_field', 'light', 'SOS_triggered'])
    return df

def train_model(data_path):
    # Load the training data
    data = load_data(data_path)
    if data is None:
        return None

    # Validate the required columns
    required_columns = ['acceleration', 'rotation', 'magnetic_field', 'light', 'SOS_triggered']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None

    # Generate and combine synthetic data
    synthetic_data = generate_synthetic_data(1000)  # Generating 1000 balanced samples
    combined_data = pd.concat([data, synthetic_data], ignore_index=True)

    # Print class distribution
    print("\nClass Distribution:")
    print(combined_data['SOS_triggered'].value_counts(normalize=True))

    # Split features and target
    X = combined_data.drop('SOS_triggered', axis=1)
    y = combined_data['SOS_triggered']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Added max_depth to prevent overfitting
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print model performance
    print("\nModel Performance Report:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model

# Function to predict SOS triggering for new data
def predict_sos(model, acceleration, rotation, magnetic_field, light):
    if model is None:
        print("Error: Model not trained")
        return None
    
    features = np.array([[acceleration, rotation, magnetic_field, light]])
    prediction = model.predict(features)[0]
    return bool(prediction)

# Main execution
if __name__ == "__main__":
    # Specify your CSV file path
    csv_file_path = "sensor_data.csv"
    
    # Train the model
    trained_model = train_model(csv_file_path)
    
    if trained_model is not None:
        # Example predictions with a mix of triggering and non-triggering cases
        print("\nExample Predictions:")
        test_cases = [
            # Non-triggering cases
            [100, 10, 450, 400],     # All below thresholds
            [115, 14, 480, 450],     # Close to but below thresholds
            
            # Single threshold exceeding cases
            [125, 10, 450, 400],     # High acceleration
            [100, 16, 450, 400],     # High rotation
            [100, 10, 520, 400],     # High magnetic field
            [100, 10, 450, 550],     # High light
            
            # Multiple threshold exceeding cases
            [125, 16, 450, 400],     # High acceleration and rotation
            [125, 10, 520, 550],     # High acceleration, magnetic field, and light
        ]

        for case in test_cases:
            result = predict_sos(trained_model, *case)
            print(f"Input: {case}")
            print(f"SOS Triggered: {result}\n")



def train_and_save_model(data_path, model_save_dir='saved_models'):
    """
    Train the model and save it to disk
    
    Parameters:
    data_path (str): Path to the CSV data file
    model_save_dir (str): Directory to save the model
    
    Returns:
    tuple: (trained model, model save path)
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Train the model
    model = train_model(data_path)
    
    if model is not None:
        # Generate timestamp for unique model name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_filename = f"sos_model_{timestamp}.joblib"
        model_path = os.path.join(model_save_dir, model_filename)
        
        # Save the model
        try:
            joblib.dump(model, model_path)
            print(f"\nModel successfully saved to: {model_path}")
            
            # Save the thresholds alongside the model
            threshold_filename = f"thresholds_{timestamp}.joblib"
            threshold_path = os.path.join(model_save_dir, threshold_filename)
            joblib.dump(THRESHOLDS, threshold_path)
            print(f"Thresholds saved to: {threshold_path}")
            
            return model, model_path
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return model, None
    
    return None, None

def load_model(model_path):
    """
    Load a saved model from disk
    
    Parameters:
    model_path (str): Path to the saved model file
    
    Returns:
    object: Loaded model or None if loading fails
    """
    try:
        model = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_latest_model(model_dir='saved_models'):
    """
    Get the path to the most recently saved model
    
    Parameters:
    model_dir (str): Directory containing saved models
    
    Returns:
    str: Path to the latest model or None if no models found
    """
    try:
        # Get all model files
        model_files = [f for f in os.listdir(model_dir) if f.startswith('sos_model_') and f.endswith('.joblib')]
        
        if not model_files:
            print("No saved models found.")
            return None
        
        # Get the most recent model
        latest_model = max(model_files)
        model_path = os.path.join(model_dir, latest_model)
        return model_path
    
    except Exception as e:
        print(f"Error finding latest model: {str(e)}")
        return None

# Modified main execution
if __name__ == "__main__":
    # Specify your CSV file path
    csv_file_path = "sensor_data.csv"
    
    # Train and save the model
    trained_model, model_save_path = train_and_save_model(csv_file_path)
    
    if trained_model is not None:
        # Example of loading and using the saved model
        print("\nTesting model loading and prediction...")
        
        # Load the model
        loaded_model = load_model(model_save_path)
        
        if loaded_model is not None:
            # Example predictions
            test_cases = [
                [100, 10, 450, 400],     # Non-triggering case
                [125, 10, 450, 400],     # Triggering case (high acceleration)
            ]

            print("\nMaking predictions with loaded model:")
            for case in test_cases:
                result = predict_sos(loaded_model, *case)
                print(f"Input: {case}")
                print(f"SOS Triggered: {result}\n")