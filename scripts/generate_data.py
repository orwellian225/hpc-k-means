import sys
import numpy as np

def main():
    if len(sys.argv) == 4:
        num_dimensions = int(sys.argv[1])
        num_points = int(sys.argv[2])
        outdir = str(sys.argv[3])
    else:
        num_dimensions = int(input("Enter the number of dimensions: "))
        num_points = int(input("Enter the number of points: "))
        outdir = str(input("Enter the output directory for the data: "))

    outfile = f"{num_points}_{num_dimensions}D.csv"
    print(f"Generating {num_points} {num_dimensions}D points in space in file {outdir + outfile}")

    data = np.random.random((num_points, num_dimensions)) * num_points - num_points / 2

    np.savetxt(outdir + outfile, data, delimiter=",")

if __name__ == "__main__":
    main()