from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
from PIL import Image
import io
import random
import os
import easyocr
import subprocess
app = Flask(__name__)


def download_image(image_url, val_file):
    """Download an image from a URL."""
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Generate a filename based on the URL to minimize overwrites
    filename = f"{val_file}.jpg"
    cv2.imwrite(filename, cv_image)
    
   

def crop_words(image_path, val_file):
    """Crop each word from the image using provided bounding boxes."""
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    results = reader.readtext(img)
    for i, (bbox, text, prob) in enumerate(results):
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        x_min, y_min = top_left
        x_max, y_max = bottom_right
        cropped_image = img[y_min:y_max, x_min:x_max]
        crop_img_path = f'{top_left}_{bottom_right}.png'
        cv2.imwrite(crop_img_path + '/', cropped_image)

def perform_ocr_on_image(image):
    # Define the command as a list of arguments
    command = [
        "python3", "/home/azureuser/deep-text-recognition-benchmark/download/demo.py",
        "--Transformation", "TPS",
        "--FeatureExtraction", "ResNet",
        "--SequenceModeling", "BiLSTM",
        "--Prediction", "Attn",
        "--image_folder", image,
        "--saved_model", "/home/azureuser/deep-text-recognition-benchmark/download/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth"
        ]

# File where the output will be saved
    output_file = "/tmp/result.txt"

# Open the output file in write mode
    with open(output_file, "w") as file:
    # Run the command and redirect stdout to the file
        subprocess.run(command, stdout=file)
def read_files(result_file):
    data = {}
    with open(result_file, 'r') as file:
                for line in file:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        filename, text = parts
                        data[filename.strip()] = text.strip()
    return data
@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    print("api hit done ->>")
    val_file = "/tmp/" + "tempp"
    if not os.path.exists(val_file):
        os.makedirs(val_file)
        print(f"Directory '{val_file}' was created.")
    result_file = "/tmp/result.txt"

    val_txt_file = val_file +  "/" + "labels.txt"
    data = request.json
    print("value of data ===", data)
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'status': 'error', 'message': 'No image URL provided'}), 400

    try:
        image_file = val_file + "/" + "tmpp2"
        download_image(image_url, image_file)
        print("download image done ==")
        # Dummy bounding boxes, replace with actual data or detection logic
        crop_words(image_file + '.jpg', val_file)
        print("cropping word done = ")
        os.remove(image_file + ".jpg")
        perform_ocr_on_image(val_file)
        print("ocr performance doneee")
        data = read_files(result_file)
        ans = {}
        for key in sorted(data.keys()):
            ans[key]=data.get(key, "")
        

        
        return jsonify({'status': 'success', 'data': ans})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port="5000")
