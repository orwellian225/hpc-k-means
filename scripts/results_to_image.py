import sys
import math as m
import csv
from PIL import Image
import numpy as np

def main():
    if len(sys.argv) == 4:
        image_name = str(sys.argv[1])
        infile_path = str(sys.argv[2])
        outdir = str(sys.argv[3])
    else:
        image_name = str(input("Enter a image name: "))
        infile_path = str(input("Enter the image data filepath: "))
        outdir = str(input("Enter the output directory: "))

    points = []
    centroids = []
    classifications = []

    with open(infile_path, 'r+') as f:
        csvr = csv.reader(f)

        # Read points
        for line in csvr:
            if len(line) == 0:
                break

            points.append([float(x) for x in line])

        # Read centroids
        for line in csvr:
            if len(line) == 0:
                break

            centroids.append([float(x) for x in line])

        # Read classifications 
        for line in csvr:
            classifications.append([int(x) for x in line])

    points = np.array(points)
    centroids = np.array(centroids)
    classifications = np.array(classifications)

    width = int(m.sqrt(len(points)))
    height = int(m.sqrt(len(points)))
    data = np.zeros((height * width, 3))


    for classification in classifications:
        data[classification[0], :] = centroids[classification[1], :] 

    num_pixels = len(data)
    num_channels = len(data[0])
    data = np.reshape(data, (int(m.sqrt(num_pixels)), int(m.sqrt(num_pixels)), num_channels))
    data = data * 255

    image = Image.fromarray(data.astype(np.uint8))
    outfile = f"{image_name}_{num_pixels}_{num_channels}D_{len(centroids)}C_image.tiff"
    image.save(outdir + outfile, "TIFF")

if __name__ == "__main__":
    main()