import os
import cv2
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from keras import to_categorical

def load_images(data_dir, img_size):
    images = []  
    labels = []  
    img_filenames = os.listdir(data_dir)  
    print("Found image filenames:", img_filenames)
    
    for img_name in img_filenames:
        img_path = os.path.join(data_dir, img_name)  
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (img_size, img_size)) 
        img = img.astype('float32') / 255.0 
        images.append(img) 
        
        label_match = re.match(r'.*?([0-9]+).*', img_name)  
        if label_match:
            label = int(label_match.group(1))  
        else:
            label = -1  
        labels.append(label)  # Add the label to the list
    
    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Filter out invalid labels (-1)
    valid_indices = labels != -1  
    images = images[valid_indices] 
    labels = labels[valid_indices] 

    # Encode labels into integers from 0 to num_classes-1
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  
    num_classes = len(label_encoder.classes_)
    
    print("Unique labels found:", label_encoder.classes_)
    return images, labels, num_classes

# Function to save processed images and labels to a compressed .npz file
def save_data(images, labels, save_path):
    np.savez_compressed(save_path, images=images, labels=labels)  # Save as compressed .npz file

# Main function to load data, preprocess it, and save it for different splits (train, val, test)
def main():
    img_size = 28  # Define the image size (28x28)
    
    # Define directories for train, validation, and test datasets
    data_dirs = {
        'train': 'data/train',
        'val': 'data/val',
        'test': 'data/test'
    }
    
    # Loop through each data split (train, val, test)
    for split, data_dir in data_dirs.items():
        images, labels, num_classes = load_images(data_dir, img_size)        
        labels = to_categorical(labels, num_classes=num_classes)
        save_data(images, labels, f'data/{split}.npz')

if __name__ == '__main__':
    main()
