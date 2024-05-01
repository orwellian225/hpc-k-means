import sys
from PIL import Image
import numpy as np

def main():
    if len(sys.argv) == 4:
        image_name = str(sys.argv[1])
        image_path = str(sys.argv[2])
        outdir = str(sys.argv[3])
    else:
        image_name = str(input("Enter the image name: "))
        image_path = str(input("Enter the image path: "))
        outdir = str(input("Enter the output directory: "))

    image = Image.open(image_path)
    width = image.width
    height = image.height
    num_channels = len(image.getbands())

    outfile = f"{image_name}_{width * height}_{num_channels}D_image.csv"
    data = np.reshape(np.array(image), (width * height, 3)) / 255
    np.savetxt(outdir + outfile, data, delimiter=",")

if __name__ == "__main__":
    main()