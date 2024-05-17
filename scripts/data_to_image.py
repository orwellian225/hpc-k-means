import sys
import math as m
import numpy as np
from PIL import Image

def main():
    if len(sys.argv) == 4:
        image_name = str(sys.argv[1])
        filepath = str(sys.argv[2])
        outdir = str(sys.argv[3])
    else:
        image_name = str(input("Enter the image name: "))
        filepath = str(input("Enter the image data file: "))
        outdir = str(input("Enter the output directory: "))

    data = np.genfromtxt(filepath, delimiter=",")
    num_points = int(m.sqrt(len(data)))
    data = data.reshape((num_points, num_points, 3))
    data = data * 255

    image = Image.fromarray(data.astype(np.uint8))
    outfile = f"{image_name}.tiff"
    image.save(outdir + outfile, "TIFF")

if __name__ == "__main__":
    main()