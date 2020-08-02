import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\david\Downloads\My Project 89279-337745e1ad77.json"
from google.cloud import vision
from google.cloud.vision import types
import csv

directory = os.fsencode('ReceiptImages')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        print(filename)
        client = client = vision.ImageAnnotatorClient()

        image_to_open = os.path.join(os.path.dirname(__file__),
                                     'ReceiptImages/', filename)

        with open(image_to_open, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        text_respone = client.text_detection(image=image)
        texts = text_respone.text_annotations
        input = str(texts[0].description)
        input = input.replace('\n', ' ')
        input = input.replace(',', '')
        row_content = [str(filename), input]
        print(input)
        with open('texts.csv', 'a+', encoding='utf-8') as fd:
            writer = csv.writer(fd)
            writer.writerow(row_content)



    else:
        continue
