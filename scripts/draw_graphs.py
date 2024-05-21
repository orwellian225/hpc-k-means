import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) == 2:
        data_filepath = str(sys.argv[1])
    else:
        data_filepath = str(input("Enter the data filepath: "))

    df = pd.read_csv(data_filepath)

    df["Speedup"] = 0.

    for i in range(35):
        df["Speedup"].iloc[i + 35] = df["Net Time (ms)"].iloc[i] / df["Net Time (ms)"].iloc[i + 35]
        df["Speedup"].iloc[i + 70] = df["Net Time (ms)"].iloc[i] / df["Net Time (ms)"].iloc[i + 70]

    df_filtered = df[(df["Parallel Implementation"] != "serial") & (df["Dimension"] == 10)]
    # print(df_filtered[["Dimension", "Num Points", "Num Classes", "Parallel Implementation", "Speedup"]])

    colours = {
        ('mpi', 2): '#FF9999',
        ('mpi', 5): '#FF6666',
        ('mpi', 10): '#FF3333',
        ('mpi', 100): '#FF0000',
        ('cuda', 2): '#6aA765',
        ('cuda', 5): '#487748',
        ('cuda', 10): '#2B472B',
        ('cuda', 100): '#0E180E',
    }

    plt.figure(figsize=(10,6))

    for key_impl, grp_impl in df_filtered.groupby('Parallel Implementation'):
        for key_num_classes, grp_num_classes in grp_impl.groupby('Num Classes'):
            plt.plot(grp_num_classes['Num Points'],  grp_num_classes["Speedup"], color=colours[(key_impl, key_num_classes)], label=f"{key_impl}: K = {key_num_classes}")

    plt.xscale('log')
    plt.xlabel("Num Points")
    plt.ylabel("Speedup")
    plt.title('Speedup vs Number of Points')
    plt.legend(title='Parallel Implementation', loc="upper left")
    plt.savefig('test.pdf')

if __name__ == "__main__":
    main()