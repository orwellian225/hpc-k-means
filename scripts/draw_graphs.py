import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

    df_filtered_no_serial = df[(df["Parallel Implementation"] != "serial") & (df["Dimension"] == 10)]
    df_filtered = df[(df["Dimension"] == 10)]
    # print(df_filtered[["Dimension", "Num Points", "Num Classes", "Parallel Implementation", "Speedup"]])

    colours = {
        ('mpi', 2): '#cd5c5c',
        ('mpi', 5): '#4e0707',
        ('mpi', 10): '#800000',
        ('mpi', 100): '#ff0000',

        ('cuda', 2): '#aef359',
        ('cuda', 5): '#32612d',
        ('cuda', 10): '#3cb043',
        ('cuda', 100): '#00ff00',

        ('serial', 2): '#0492c2',
        ('serial', 5): '#59788e',
        ('serial', 10): '#3944bc',
        ('serial', 100): '#0000FF',
    }

    plt.figure(figsize=(10,6))

    for key_impl, grp_impl in df_filtered_no_serial.groupby('Parallel Implementation'):
        for key_num_classes, grp_num_classes in grp_impl.groupby('Num Classes'):
            plt.plot(grp_num_classes['Num Points'],  grp_num_classes["Speedup"], color=colours[(key_impl, key_num_classes)], label=f"{key_impl}: K = {key_num_classes}")

    plt.xscale('log')
    plt.xlabel("Num Points")
    plt.ylabel("Speedup")
    plt.title('Speedup vs Number of Points')
    plt.legend(title='Parallel Implementation', loc="upper left")
    plt.tight_layout()
    plt.savefig('graphs/speedup.pdf')

    plt.figure(figsize=(10,6))
    for key_impl, grp_impl in df_filtered_no_serial.groupby('Parallel Implementation'):
        for key_num_classes, grp_num_classes in grp_impl.groupby('Num Classes'):
            plt.plot(grp_num_classes['Num Points'],  grp_num_classes["Loss"], color=colours[(key_impl, key_num_classes)], label=f"{key_impl}: K = {key_num_classes}")

    plt.xscale('log')
    plt.xlabel("Num Points")
    plt.legend(title='Parallel Implementation', loc="upper left")
    plt.title('Loss vs Number of Points')
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig('graphs/loss.pdf')

    plt.figure(figsize=(10,6))
    for key_impl, grp_impl in df_filtered_no_serial.groupby('Parallel Implementation'):
        for key_num_classes, grp_num_classes in grp_impl.groupby('Num Classes'):
            plt.plot(grp_num_classes['Num Points'],  (1 - grp_num_classes["Errors"] / grp_num_classes["Num Points"]) * 100 , color=colours[(key_impl, key_num_classes)], label=f"{key_impl}: K = {key_num_classes}")
    plt.xscale('log')
    plt.xlabel("Num Points")
    plt.legend(title='Parallel Implementation', loc="upper left")
    plt.title('Accuracy vs Number of Points')
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig('graphs/accuracy.pdf')

    plt.figure(figsize=(10,6))

    df_percentages = df.rename(columns={
        "Sum Classify Time (ms)": "Sum Classify Time (%)",
        "Sum Update Time (ms)": "Sum Update Time (%)",
        "Last Classify Time (ms)": "Last Classify Time (%)",
    })[(df["Dimension"] == 10) & ((df["Num Points"] == 1_000_000) | (df["Num Points"] == 100_489)) & ((df["Num Classes"] == 100) | (df["Num Classes"] == 10))]
    df_percentages[["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)"]] = df_percentages[["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)"]].div((df_percentages["Net Time (ms)"] - df_percentages["Initialize Time (ms)"]).values, axis=0)

    # Group by 'Num Points', 'Num Classes', 'Parallel Implementation'
    grouped = df_percentages.groupby(['Num Classes', 'Num Points', 'Parallel Implementation'])

    # Prepare data for plotting
    plot_data = {}
    for (num_classes, num_points, parallel_impl), group in grouped:
        key = (num_points, num_classes, parallel_impl)
        plot_data[key] = group[["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)"]].mean()

    plot_df = pd.DataFrame(plot_data).T
    plot_df.columns = ["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)"]

    # Calculate the overhead
    plot_df['Overhead (%)'] = 1 - plot_df[["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)"]].sum(axis=1)

    # Add the custom label for each row
    plot_df['label'] = [f"n={int(num_points):.0e} | k={num_classes} | {parallel_impl}" for num_points, num_classes, parallel_impl in plot_df.index]

    ax = plot_df[["Sum Classify Time (%)", "Sum Update Time (%)", "Last Classify Time (%)", "Overhead (%)"]].plot(kind='barh', stacked=True, figsize=(10, 8))

    # Customize the plot
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'])

    plt.xlabel("Percentage of Net Time")
    plt.title('Percentage of time in each algorithm step')
    plt.tight_layout()
    plt.savefig('graphs/time-breakdown.pdf')

    plt.figure(figsize=(10,6))
    for key_impl, grp_impl in df_filtered.groupby('Parallel Implementation'):
        for key_num_classes, grp_num_classes in grp_impl.groupby('Num Classes'):
            plt.plot(grp_num_classes['Num Points'],  grp_num_classes["Sum Classify Time (ms)"] / 1000 , color=colours[(key_impl, key_num_classes)], label=f"{key_impl}: K = {key_num_classes}")
    plt.xscale('log')
    plt.xlabel("Num Points")
    plt.legend(title='Parallel Implementation', loc="upper left")
    plt.title('Classification time vs Number of Points')
    plt.ylabel("Cumulative classification time (s)")
    plt.tight_layout()
    plt.savefig('graphs/sum_class_time.pdf')

if __name__ == "__main__":
    main()