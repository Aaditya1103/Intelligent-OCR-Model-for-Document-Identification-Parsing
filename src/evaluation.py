import numpy as np
from tensorflow import load_model
from sklearn.metrics import classification_report, confusion_matrix

def load_data(split):
    data = np.load(f'../data/{split}.npz') 
    return data['images'], data['labels']  

def main():
    model_path = '../models/handwriting_recognition_model.h5'
    model = load_model(model_path)
    
    # Load test data (images and labels)
    x_test, y_test = load_data('test')
    
    # Make predictions on the test set
    y_pred = model.predict(x_test) 
    
    # Convert predicted probabilities to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)  
    y_true = np.argmax(y_test, axis=1)
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred_classes))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred_classes))

if __name__ == '__main__':
    main()
