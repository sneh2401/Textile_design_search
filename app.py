from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
import os

app = Flask(__name__, template_folder='files/templates', static_folder='files/static')


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(ellipMask, cornerMask)
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return features

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None, 255, 0, cv2.NORM_MINMAX).flatten()
        return hist


class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=10):
        results = {}

        with open(self.indexPath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid or empty rows
                try:
                    features = [float(x) for x in row[1:]]
                    d = self.chi2_distance(features, queryFeatures)
                    results[row[0]] = d
                except ValueError:
                    continue  # Skip rows with non-numeric data

        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d


def index_images(dataset, index):
    cd = ColorDescriptor((8, 12, 3))
    with open(index, "a", newline='') as output:
        for imagePath in glob.glob(dataset + "/*.jpg"):
            imageID = os.path.basename(imagePath)
            if imageID not in get_indexed_images(index):
                try:
                    image = cv2.imread(imagePath)
                    if image is not None:
                        features = cd.describe(image)
                        features = [str(f) for f in features]
                        output.write("%s,%s\n" % (imageID, ",".join(features)))
                    else:
                        print(f"Failed to load image: {imagePath}")
                except cv2.error as e:
                    print(f"OpenCV Error: {e}")


def get_indexed_images(index):
    indexed_images = set()
    if os.path.exists(index):
        with open(index, "r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    indexed_images.add(row[0])
    return indexed_images


@app.route('/update_index')
def update_index():
    dataset = 'files/static/dataset/'
    index = 'files/index.csv'

    cd = ColorDescriptor((8, 12, 3))
    indexed_images = get_indexed_images(index)

    for imagePath in glob.glob(dataset + "/*.jpg"):
        imageID = os.path.basename(imagePath)
        if imageID not in indexed_images:
            image = cv2.imread(imagePath)
            if image is not None:
                features = cd.describe(image)
                features = [str(f) for f in features]
                with open(index, "a", newline='') as output:
                    output.write("%s,%s\n" % (imageID, ",".join(features)))

    return 'Index update completed.'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/search', methods=['POST'])
def search():
    uploaded_file = request.files['file']
    limit = int(request.form['limit'])
    index = 'files/index.csv'

    file_path = 'files/static/uploads/' + secure_filename(uploaded_file.filename)
    uploaded_file.save(file_path)

    results = search_images(file_path, index, limit)
    os.remove(file_path)

    return render_template('result.html', results=results)


@app.route('/index', methods=['POST'])
def index_dataset():
    dataset = 'files/static/dataset/'
    index = 'files/index.csv'

    index_images(dataset, index)

    return 'Image indexing completed.'


def search_images(query, index, limit=10):
    cd = ColorDescriptor((8, 12, 3))
    queryImage = cv2.imread(query)
    queryFeatures = cd.describe(queryImage)

    searcher = Searcher(index)
    results = searcher.search(queryFeatures, limit=limit)

    return results


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


#upload the photos in the dataset ant then press index 
# one time the the the index.csv file will get cordinate value in the index.csv 
