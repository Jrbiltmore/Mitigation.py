# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('data.csv')

# Prepare the data
X = data.drop('target', axis=1)  # Input features
y = data['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# Function to perform detection
def detect(input_data):
    prediction = classifier.predict([input_data])
    return prediction[0]

# Example usage
input_data = [0.2, 0.5, 0.8, 0.1, 0.3]  # Replace with actual input data
prediction = detect(input_data)
print("Detection result:", prediction)
