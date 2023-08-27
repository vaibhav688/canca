import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.optimizer_v1 as tf
import csv
from PIL import Image



class convert:
    def __init__(self,data,inp_path,output_path):
        self.data=data
        self.inp_path=inp_path
        self.output_path=output_path

    def make_to_csv(self):
        LABELS = {
            (255, 255, 255): 0,  # Example: White color corresponds to label 0
            (0, 0, 0): 1,  # Example: Black color corresponds to label 1
            # Add more color-label mappings as needed
        }
        try:
            # Open the image
            image = Image.open(self.inp_path)

            # Convert the image to RGB mode (in case it's grayscale or other modes)
            image = image.convert("RGB")

            # Get the dimensions of the image
            width, height = image.size

            # Prepare CSV writer
            with open(self.output_path, mode='w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write the header row
                csv_writer.writerow(['x', 'y', 'R', 'G', 'B', 'Target', 'Label'])

                # Iterate through each pixel and write its RGB values, target, and label to the CSV file
                for y in range(height):
                    for x in range(width):
                        r, g, b = image.getpixel((x, y))
                        target = f"{r},{g},{b}"
                        label = LABELS.get((r, g, b), -1)  # Use -1 as a default label for unknown colors
                        csv_writer.writerow([x, y, r, g, b, target, label])

            print("CSV file with targets and labels successfully created.")
        except Exception as e:
            print(f"Error: {e}")

class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=self.state_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_actions, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=tf.Adam(), metrics=['accuracy'])
        return model

    def train(self, states, labels, num_epochs, batch_size):
        # One-hot encode the labels
        encoded_labels = pd.get_dummies(labels).values
        states = states.astype('float32')

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(states, encoded_labels, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)

    def predict(self, states):
        return self.model.predict(states)





""""Load the CSV dataset
data1=pd.read_csv("/content/Cancer.csv")
data1=data1.drop('Patient Id',axis=1)

# Extract the features and labels from the dataset
features = data1.drop('Level', axis=1).values
labels = data1['Level'].values

# Preprocess the labels using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Define the state shape and number of actions
state_shape = (features.shape[1],)
num_actions = len(label_encoder.classes_)

# Create the DQNAgent and train the model
agent = DQNAgent(state_shape, num_actions)
agent.train(features, labels, num_epochs=10, batch_size=32)

# Predict the labels using the trained DQNAgent
predictions = agent.predict(features)"""



