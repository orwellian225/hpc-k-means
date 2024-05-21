import sys
import csv
import math as m

import pprint

import class_error
import class_loss

def main():
    if len(sys.argv) == 3:
        csv_filepath = str(sys.argv[1])
        output_filepath = str(sys.argv[2])
    else: 
        csv_filepath = str(input("Data csv filepath: "))
        output_filepath = str(input("Ouput csv filepath: "))

    features = []
    data = {
        'serial': {},
        'mpi': {},
        'cuda': {}
    }
    with open(csv_filepath, 'r+') as f:
        csvr = csv.reader(f)
        features = next(csvr)

        for line in csvr:
            print(f"line {csvr.line_num}\r", end="")
            if (csvr.line_num % 10 == 0):
                sys.stdout.flush()
            dimension = int(line[0])
            num_points = int(line[1])
            num_classes = int(line[2])

            serial_file = str(line[-3])
            mpi_file = str(line[-2])
            cuda_file = str(line[-1])

            parallel_impl = [ 'serial', 'mpi', 'cuda' ]

            for p in parallel_impl:
                try:
                    _ = data[p][dimension]
                except KeyError:
                    data[p][dimension] = { num_points: { num_classes: {
                        'count': 0,
                        'net_time': 0.,
                        'initialize_time': 0.,
                        'sum_classify_time': 0.,
                        'sum_update_time': 0.,
                        'last_classify_time': 0.,
                        'loss': 0.,
                        'errors': 0.
                    } } }

                try:
                    _ = data[p][dimension][num_points]
                except KeyError:
                    data[p][dimension][num_points] = { num_classes: {
                        'count': 0,
                        'net_time': 0.,
                        'initialize_time': 0.,
                        'sum_classify_time': 0.,
                        'sum_update_time': 0.,
                        'last_classify_time': 0.,
                        'loss': 0.,
                        'errors': 0.
                    } }

                try:
                    _ = data[p][dimension][num_points][num_classes]
                except KeyError:
                    data[p][dimension][num_points][num_classes] = {
                        'count': 0,
                        'net_time': 0.,
                        'initialize_time': 0.,
                        'sum_classify_time': 0.,
                        'sum_update_time': 0.,
                        'last_classify_time': 0.,
                        'loss': 0.,
                        'errors': 0.
                    }


            mpi_loss, cuda_loss = class_loss.determine_class_los(num_points, num_classes, serial_file, mpi_file, cuda_file)
            parallel_impl_loss = class_loss.average_loss(num_classes, mpi_loss, cuda_loss)

            mpi_errors, cuda_errors = class_error.count_class_errors(num_points, num_classes, serial_file, mpi_file, cuda_file)
            parallel_impl_errors = class_error.count_net_errors(num_classes, mpi_errors, cuda_errors)
            
            for i in range(3):
                data[parallel_impl[i]][dimension][num_points][num_classes]['count'] += 1
                data[parallel_impl[i]][dimension][num_points][num_classes]['net_time'] += float(line[5 * i + 4])
                data[parallel_impl[i]][dimension][num_points][num_classes]['initialize_time'] += float(line[5 * i + 5])
                data[parallel_impl[i]][dimension][num_points][num_classes]['sum_classify_time'] += float(line[5 * i + 6])
                data[parallel_impl[i]][dimension][num_points][num_classes]['sum_update_time'] += float(line[5 * i + 7])
                data[parallel_impl[i]][dimension][num_points][num_classes]['last_classify_time'] += float(line[5 * i + 8])

                if i > 0:
                    data[parallel_impl[i]][dimension][num_points][num_classes]['loss'] = parallel_impl_loss[i - 1] if not m.isnan(parallel_impl_loss[i - 1]) or not m.isinf(parallel_impl_loss[i - 1]) else 0
                    data[parallel_impl[i]][dimension][num_points][num_classes]['errors'] = parallel_impl_errors[i - 1] if not m.isnan(parallel_impl_errors[i - 1]) or not m.isinf(parallel_impl_errors[i - 1]) else 0


    for p in data:
        for d in data[p]:
            for n in data[p][d]:
                for k in data[p][d][n]:
                    data[p][d][n][k]['net_time'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['initialize_time'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['sum_classify_time'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['sum_update_time'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['last_classify_time'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['loss'] /= data[p][d][n][k]['count']
                    data[p][d][n][k]['errors'] /= data[p][d][n][k]['count']

    with open(output_filepath, 'w+') as f:
        csvw = csv.writer(f, delimiter=',')
        csvw.writerow(['Dimension', 'Num Points', 'Num Classes', 'Parallel Implementation', 'Net Time (ms)', 'Initialize Time (ms)', 'Sum Classify Time (ms)', 'Sum Update Time (ms)', 'Last Classify Time (ms)', 'Loss', 'Errors'])

        for p in data:
            for d in data[p]:
                for n in data[p][d]:
                    for k in data[p][d][n]:
                        csvw.writerow([ d, n, k, p,
                            data[p][d][n][k]['net_time'],
                            data[p][d][n][k]['initialize_time'],
                            data[p][d][n][k]['sum_classify_time'],
                            data[p][d][n][k]['sum_update_time'],
                            data[p][d][n][k]['last_classify_time'],
                            data[p][d][n][k]['loss'],
                            data[p][d][n][k]['errors'],
                        ])

if __name__ == "__main__":
    main()