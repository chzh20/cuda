import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Dictionary to store results for each matrix size
    results = {}
    
    # Parse the data using regex
    size_pattern = r'm=(\d+), n=\1, k=\1'
    perf_pattern = r'(\w+) elapsed time: (\d+\.\d+) ms GFLOPS: (\d+\.\d+)'
    
    current_size = None
    for line in content.split('\n'):
        size_match = re.match(size_pattern, line)
        if size_match:
            current_size = int(size_match.group(1))
            results[current_size] = {'kernels': [], 'time': [], 'gflops': []}
        else:
            perf_match = re.match(perf_pattern, line)
            if perf_match and current_size:
                kernel, time, gflops = perf_match.groups()
                results[current_size]['kernels'].append(kernel)
                results[current_size]['time'].append(float(time))
                results[current_size]['gflops'].append(float(gflops))
    with open('performance_data.json', 'w') as f:
        json.dump(results, f)
    return results

def plot_performance(results):
    sizes = list(results.keys())
    kernels = list(dict.fromkeys([kernel for size in sizes for kernel in results[size]['kernels']]))
    print(kernels)
    print(results)
    
    # Set up the plot style
    plt.style.use('classic')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    
    # Use color cycle
    colors = plt.cm.Set3(np.linspace(0, 1, len(kernels)))
    
    # Width of each bar and positions of bar groups
    bar_width = 0.8 / len(kernels)
    r = np.arange(len(sizes))
    
    # Get cublas baseline performance
    cublas_gflops = {}
    cublas_times = {}
    for size in sizes:
        # Get the second cublas result (warm-up ignored)
        cublas_idx = [i for i, k in enumerate(results[size]['kernels']) if k == 'cublas_sgemm'][1]
        cublas_gflops[size] = results[size]['gflops'][cublas_idx]
        cublas_times[size] = results[size]['time'][cublas_idx]
    
    # Plot bars for each kernel
    for i, (kernel, color) in enumerate(zip(kernels, colors)):
        times = []
        gflops = []
        relative_perf = []
        relative_time = []
        for size in sizes:
            try:
                idx = results[size]['kernels'].index(kernel)
                times.append(results[size]['time'][idx])
                gflops.append(results[size]['gflops'][idx])
                relative_perf.append(results[size]['gflops'][idx] / cublas_gflops[size] * 100)
                relative_time.append(results[size]['time'][idx] / cublas_times[size] * 100)
            except ValueError:
                times.append(0)
                gflops.append(0)
                relative_perf.append(0)
                relative_time.append(0)
        
        pos = r + i * bar_width
        ax1.bar(pos, times, bar_width, label=kernel, color=color)
        ax2.bar(pos, gflops, bar_width, label=kernel, color=color)
        if kernel != 'cublas_sgemm':  # Skip cublas in relative plots
            ax3.bar(pos, relative_perf, bar_width, label=kernel, color=color)
            ax4.bar(pos, relative_time, bar_width, label=kernel, color=color)
    
    # Customize plots
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticks(r + bar_width * len(kernels) / 2)
        ax.set_xticklabels([f'{size}x{size}' for size in sizes])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    
    ax2.set_ylabel('GFLOPS')
    ax2.set_title('Performance Comparison')
    
    ax3.set_ylabel('Percentage of cuBLAS Performance (%)')
    ax3.set_title('Relative Performance to cuBLAS')
    
    ax4.set_ylabel('Percentage of cuBLAS Time (%)')
    ax4.set_title('Relative Execution Time to cuBLAS (Lower is Better)')
    
    plt.tight_layout()
    plt.savefig('gemm_performance.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        results = parse_data("performance_data.txt")
        plot_performance(results)
    except FileNotFoundError:
        print("Error: performance_data.txt not found.")
        print("Please create the file with your performance data first.")
        sys.exit(1)