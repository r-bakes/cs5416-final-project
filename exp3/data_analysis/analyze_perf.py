"""
How to run:
python3 analyze_perf.py --csv_path experiment3.csv --output_dir ./graphs
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.ffill()

    df["reported_client_final_time"] = (
        df["reported_client_final_time"]
        .astype(str)
        .str.replace("s", "", regex=False)
        .astype(float)
    )

    return df

def plot_runtime_vs_batch(df, output_dir):
    plt.figure(figsize=(8,6))
    for w in sorted(df.orchestrator_num_workers.unique()):
        sub = df[df.orchestrator_num_workers == w]
        plt.plot(sub.max_batch_size, sub.reported_client_final_time, marker='o', label=f'workers={w}')
    plt.xlabel('Max Batch Size')
    plt.ylabel('Final Time (s)')
    plt.title('Runtime vs Batch Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_batch.png'))
    plt.close()

def plot_runtime_vs_workers(df, output_dir):
    plt.figure(figsize=(8,6))
    for b in sorted(df.max_batch_size.unique()):
        sub = df[df.max_batch_size == b]
        plt.plot(sub.orchestrator_num_workers, sub.reported_client_final_time, marker='o', label=f'batch={b}')
    plt.xlabel('Number of Workers')
    plt.ylabel('Final Time (s)')
    plt.title('Runtime vs Workers')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_workers.png'))
    plt.close()

def plot_runtime_heatmap(df, output_dir):
    pivot = df.pivot_table(
        index='orchestrator_num_workers',
        columns='max_batch_size',
        values='reported_client_final_time'
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap='viridis')
    plt.title('Runtime Heatmap')
    plt.ylabel('Workers')
    plt.xlabel('Batch Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_heatmap.png'))
    plt.close()

def plot_memory_heatmap(df, output_dir):
    df['avg_mem'] = df[['Max_Ram_Used_Node_0','Max_Ram_Used_Node_1','Max_Ram_Used_Node_2']].mean(axis=1)
    pivot = df.pivot_table(
        index='orchestrator_num_workers',
        columns='max_batch_size',
        values='avg_mem'
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap='magma')
    plt.title('Avg Memory Used Heatmap')
    plt.ylabel('Workers')
    plt.xlabel('Batch Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_heatmap.png'))
    plt.close()


def plot_memory_vs_batch(df, output_dir):
    plt.figure(figsize=(8,6))
    for w in sorted(df.orchestrator_num_workers.unique()):
        sub = df[df.orchestrator_num_workers == w]
        plt.plot(sub.max_batch_size, sub.avg_mem, marker='o', label=f'workers={w}')
    plt.xlabel('Max Batch Size')
    plt.ylabel('Avg Memory Used (GB)')
    plt.title('Memory vs Batch Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_vs_batch.png'))
    plt.close()

def plot_memory_vs_workers(df, output_dir):
    plt.figure(figsize=(8,6))
    for b in sorted(df.max_batch_size.unique()):
        sub = df[df.max_batch_size == b]
        plt.plot(sub.orchestrator_num_workers, sub.avg_mem, marker='o', label=f'batch={b}')
    plt.xlabel('Number of Workers')
    plt.ylabel('Avg Memory Used (GB)')
    plt.title('Memory vs Workers')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_vs_workers.png'))
    plt.close()

def plot_latency_vs_memory(df, output_dir):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df, 
        x='avg_mem', 
        y='reported_client_final_time',
        hue='max_batch_size',
        style='orchestrator_num_workers',
        s=120
    )
    plt.xlabel('Avg Memory Used (GB)')
    plt.ylabel('Latency (s)')
    plt.title('Latency vs Memory Usage (Trade-off)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_vs_memory.png'))
    plt.close()

def plot_throughput_vs_memory(df, output_dir):
    df['throughput'] = df['Successful'] / df['reported_client_final_time']

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x='avg_mem',
        y='throughput',
        hue='max_batch_size',
        style='orchestrator_num_workers',
        s=120
    )
    plt.xlabel('Avg Memory Used (GB)')
    plt.ylabel('Throughput (requests/s)')
    plt.title('Throughput vs Memory Usage (Trade-off)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_memory.png'))
    plt.close()
def plot_throughput_vs_batch(df, output_dir):
    df['throughput'] = df['Successful'] / df['reported_client_final_time']

    plt.figure(figsize=(8,6))
    for w in sorted(df.orchestrator_num_workers.unique()):
        sub = df[df.orchestrator_num_workers == w]
        plt.plot(sub.max_batch_size, sub.throughput, marker='o', label=f'workers={w}')
    plt.xlabel('Max Batch Size')
    plt.ylabel('Throughput (requests/s)')
    plt.title('Throughput vs Batch Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_batch.png'))
    plt.close()


def plot_throughput_vs_workers(df, output_dir):
    df['throughput'] = df['Successful'] / df['reported_client_final_time']

    plt.figure(figsize=(8,6))
    for b in sorted(df.max_batch_size.unique()):
        sub = df[df.max_batch_size == b]
        plt.plot(sub.orchestrator_num_workers, sub.throughput, marker='o', label=f'batch={b}')
    plt.xlabel('Number of Workers')
    plt.ylabel('Throughput (requests/s)')
    plt.title('Throughput vs Workers')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_workers.png'))
    plt.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.csv_path)
    df = df[~((df.orchestrator_num_workers == 8) & (df.max_batch_size == 32))]

    df = df.rename(columns={
        'Max_Ram_Used_Node 0':'Max_Ram_Used_Node_0',
        'Max_Ram_Used_Node 1':'Max_Ram_Used_Node_1',
        'Max_Ram_Used_Node 2':'Max_Ram_Used_Node_2'
    })

    df['avg_mem'] = df[['Max_Ram_Used_Node_0','Max_Ram_Used_Node_1','Max_Ram_Used_Node_2']].mean(axis=1)

    plot_runtime_vs_batch(df, args.output_dir)
    plot_runtime_vs_workers(df, args.output_dir)
    plot_runtime_heatmap(df, args.output_dir)
    plot_memory_heatmap(df, args.output_dir)
    plot_memory_vs_batch(df, args.output_dir)
    plot_memory_vs_workers(df, args.output_dir)
    plot_latency_vs_memory(df, args.output_dir)
    plot_throughput_vs_memory(df, args.output_dir)
    plot_throughput_vs_batch(df, args.output_dir)
    plot_throughput_vs_workers(df, args.output_dir)

if __name__ == '__main__':
    main()
