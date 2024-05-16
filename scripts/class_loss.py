import math as m
import sys

def nvec_distance(v1: list, v2: list):
    sum = 0.
    for d in range(len(v1)):
        sum += (v1[d] - v2[d])**2

    return m.sqrt(sum)

def determine_class_los(num_points: int, num_classes: int, serial_file: str, mpi_file: str, cuda_file: str) -> tuple:
    mpi_loss = []
    cuda_loss = []

    with open(serial_file) as serial, open(mpi_file) as mpi, open(cuda_file) as cuda:
        for i, (serial_str, mpi_str, cuda_str) in enumerate(zip(serial, mpi, cuda), 1):
            serial_str = str(serial_str).rstrip()
            mpi_str = str(mpi_str).rstrip()
            cuda_str = str(cuda_str).rstrip()

            if num_points + 1 >= i:
                continue

            if i >= num_points + 1 + num_classes + 1:
                break

            serial_class_centre = [float(x) for x in serial_str.split(",")]
            mpi_class_centre = [float(x) for x in mpi_str.split(",")]
            cuda_class_centre = [float(x) for x in cuda_str.split(",")]

            mpi_loss.append(nvec_distance(mpi_class_centre, serial_class_centre))
            cuda_loss.append(nvec_distance(cuda_class_centre, serial_class_centre))

    return (mpi_loss, cuda_loss)


def average_loss(num_classes: int, mpi_loss: list, cuda_loss: list) -> tuple:
    mpi_loss_sum = 0.
    cuda_loss_sum = 0.
    for k in range(num_classes):
        # print(f"class {k} | MPI loss = {mpi_loss[k]} | CUDA loss = {cuda_loss[k]}")
        mpi_loss_sum += mpi_loss[k]
        cuda_loss_sum += cuda_loss[k]

    mpi_loss_average = mpi_loss_sum / num_classes
    cuda_loss_average = cuda_loss_sum / num_classes

    return (mpi_loss_average, cuda_loss_average)


def main():
    if len(sys.argv) == 6:
        num_points = int(sys.argv[1])
        num_classes = int(sys.argv[2])
        serial_file = str(sys.argv[3])
        mpi_file = str(sys.argv[4])
        cuda_file = str(sys.argv[5])
    else:
        num_points = int(input("Enter the number of points: "))
        num_classes = int(input("Enter the number of classes: "))
        serial_file = str(input("Enter the serial csv filepath: "))
        mpi_file = str(input("Enter the mpi csv filepath: "))
        cuda_file = str(input("Enter the cuda csv filepath: "))

    mpi_loss, cuda_loss = determine_class_los(num_points, num_classes, serial_file, mpi_file, cuda_file)
    mpi_loss_average, cuda_loss_average = average_loss(num_classes, mpi_loss, cuda_loss)

    print(f"Average loss:")
    print(f"\tMPI: {mpi_loss_average}")
    print(f"\tCUDA: {cuda_loss_average}")

if __name__ == "__main__":
    main()