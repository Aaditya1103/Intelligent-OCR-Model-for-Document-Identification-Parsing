from paddleocr import PaddleOCR
import cv2
import os 
ocr = PaddleOCR(lang='en')
image_path = '123.png'
result = ocr.ocr(image_path)
def print_all_words(result):
    for sublist in result:
        if isinstance(sublist, list):
            print_all_words(sublist)
        elif isinstance(sublist, tuple):
            word = sublist[0]
            print(word)

print_all_words(result)