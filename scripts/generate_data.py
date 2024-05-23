import sys
import numpy as np
from perlin_noise import PerlinNoise

def main():
    if len(sys.argv) == 4:
        num_dimensions = int(sys.argv[1])
        num_points = int(sys.argv[2])
        outdir = str(sys.argv[3])
    else:
        num_dimensions = int(input("Enter the number of dimensions: "))
        num_points = int(input("Enter the number of points: "))
        outdir = str(input("Enter the output directory for the data: "))

    outfile = f"{num_points * num_points}_{num_dimensions}D.csv"

    data = np.zeros((num_points, num_points, num_dimensions))

    for d in range(num_dimensions):
        noise = PerlinNoise(octaves=10)
        
        for i in range(num_points):
            for j in range(num_points):
                data[i,j,d] = noise([i / num_points, j / num_points])

    min_data = np.min(data)
    max_data = np.max(data)
    data = (data - min_data) / (max_data - min_data)
    data = data.reshape((num_points * num_points, num_dimensions))
    # print(data)

    np.savetxt(outdir + outfile, data, delimiter=",")


if __name__ == "__main__":
    main()