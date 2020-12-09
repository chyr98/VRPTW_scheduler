import argparse
import json

import matplotlib.pyplot as plt


def median_result(result, methods, num_customers, num_perturbations):
    result_list = {m: [] for m in methods}

    for r in result:
        method = r['methods']

        if method not in methods:
            continue

        if r['parameters']['num_customers'] != num_customers:
            continue

        if r['parameters']['num_perturbations'] != num_perturbations:
            continue

        result_list[method].append((r['time_to_feasible'], r['time_vs_cost']))

    method_to_result = {}

    for m in methods:
        arg_median = int(len(result_list[m]) / 2)
        method_to_result[m] = sorted(result_list[m])[arg_median][1]

    return method_to_result


def create_figure(methods, method_to_result, output):
    fig = plt.figure(figsize=(4.5, 2), dpi=300)
    ax = fig.add_subplot(111)

    for m in methods:
        ax.plot(method_to_result[m]['time'], method_to_result[m]['cost'], label=m, marker='o', markersize=1, linewidth=0.5)

    ax.set_xlabel('time (s)', fontsize=9)
    ax.set_ylabel('cost', fontsize=9)
    plt.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--methods', '-m', type=str, nargs='+',
                        default=['CPLEX1', 'Gurobi1', 'CPLEX2', 'Gurobi2', 'LNS'])
    parser.add_argument('--num-customers', '-c', type=int, default=64)
    parser.add_argument('--num-perturbations', '-p', type=int, default=2)
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        result = json.load(f)

    median_result = median_result(result, args.methods, args.num_customers, args.num_perturbations)
    create_figure(args.methods, median_result, args.output)
