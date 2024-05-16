import sys

def count_class_errors(num_points: int, num_classes: int, serial_file: str, mpi_file: str, cuda_file: str):
    mpi_errors = [0] * num_classes
    cuda_errors = [0] * num_classes

    with open(serial_file) as serial, open(mpi_file) as mpi, open(cuda_file) as cuda:
        for i, (serial_str, mpi_str, cuda_str) in enumerate(zip(serial, mpi, cuda), 1):
            serial_str = str(serial_str).rstrip()
            mpi_str = str(mpi_str).rstrip()
            cuda_str = str(cuda_str).rstrip()

            if i <= num_points + num_classes + 2:
                continue

            serial_class = int(serial_str.split(",")[1])
            incorrect_mpi = serial_str != mpi_str
            incorrect_cuda = serial_str != cuda_str

            if incorrect_mpi or incorrect_cuda:
                if incorrect_mpi:
                    mpi_errors[serial_class] += 1
                if incorrect_cuda:
                    cuda_errors[serial_class] += 1

    return (mpi_errors, cuda_errors)

def count_net_errors(num_classes: int, mpi_errors: list, cuda_errors: list) -> tuple:
    mpi_sum = 0
    cuda_sum = 0

    for k in range(num_classes):
        mpi_sum += mpi_errors[k]
        cuda_sum += cuda_errors[k]

    return (mpi_sum, cuda_sum)

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

    mpi_errors, cuda_errors = count_class_errors(num_points, num_classes, serial_file, mpi_file, cuda_file)
    net_mpi_errors, net_cuda_errors = count_net_errors(num_classes, mpi_errors, cuda_errors)
    print("Error count:")
    print(f"\tMPI: {net_mpi_errors} => {(num_points - net_mpi_errors) / num_points * 100:.2f}% accuracy")
    print(f"\tCUDA: {net_cuda_errors} => {(num_points - net_cuda_errors) / num_points * 100:.2f}% accuracy")

if __name__ == "__main__":
    main()