import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(split):
    data = np.load(f'data/{split}.npz')
    return data['images'], data['labels']

# Function to create the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy as the metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Main function to load data, train the model, and save the best model
def main():
    input_shape = (28, 28, 1)
    num_classes = 10
    batch_size = 32
    epochs = 10
    
    # Load training and validation data
    x_train, y_train = load_data('train')
    x_val, y_val = load_data('val')
    
    # Create the CNN model
    model = create_model(input_shape, num_classes)
    
    # Set up checkpoint to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint('../models/handwriting_recognition_model.h5', save_best_only=True)
    
    # Train the model with training data, validation data, and checkpoint callback
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

if __name__ == '__main__':
    main()

