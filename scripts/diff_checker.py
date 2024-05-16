import sys

def main():
    if len(sys.argv) == 4:
        serial_file = str(sys.argv[1])
        mpi_file = str(sys.argv[2])
        cuda_file = str(sys.argv[3])
    else:
        serial_file = str(input("Enter the serial csv filepath: "))
        mpi_file = str(input("Enter the mpi csv filepath: "))
        cuda_file = str(input("Enter the cuda csv filepath: "))

    mpi_error_count = 0
    cuda_error_count = 0

    found_error = False
    with open(serial_file) as serial, open(mpi_file) as mpi, open(cuda_file) as cuda:
        for i, (s, m, c) in enumerate(zip(serial, mpi, cuda), 1):
            s = str(s).rstrip()
            m = str(m).rstrip()
            c = str(c).rstrip()

            incorrect_mpi = s != m
            incorrect_cuda = s != c

            if incorrect_mpi or incorrect_cuda:
                found_error = True
                # print(f"line {i}:")
                if incorrect_mpi:
                    mpi_error_count += 1
                    # print(f"\tMPI Incorrect: expected {s}, found {m}")
                if incorrect_cuda:
                    cuda_error_count += 1
                    # print(f"\tCuda Incorrect: expected {s}, found {c}")

    if not found_error:
        print("No errors found")
    else:
        print("Error count:")
        print(f"\tMPI: {mpi_error_count}")
        print(f"\tCUDA: {cuda_error_count}")



if __name__ == "__main__":
    main()