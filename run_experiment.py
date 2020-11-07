import argparse
from concurrent import futures
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Tuple, Optional

import VRPTW_util as util


def parse_params(problem: str) -> Dict[str, Any]:
    pattern = re.compile(r'perturbated(?P<p>\d+)_v(?P<v>\d+)_c(?P<c>\d+)_tw(?P<tw>\d+)_xy(?P<xy>\d+)_(?P<id>\d+)\.txt')
    result = pattern.match(os.path.basename(problem))

    if result is None:
        return {'error': 'failed to parse parameters from the problem'}

    params = {'num_vehicles': int(result.group('v')),
              'num_customers': int(result.group('c')),
              'time_window_parameter': int(result.group('tw')),
              'xy_limit': int(result.group('xy')),
              'problem_id': int(result.group('id')),
              'num_perturbations': int(result.group('p'))}

    return params


def service_time_difference(original_problem: util.VRPTWInstance,  
                            original_solution: List[List[int]],
                            perturbated_problem: util.VRPTWInstance,
                            perturbated_solution: List[int]) -> float:
    original_time = original_problem.get_time(original_solution)
    perturbated_time = perturbated_problem.get_time(perturbated_solution)
    value = 0.0

    for (v1, t1), (v2, t2) in zip(original_time[1:], perturbated_time[1:]):
        if v1 != v2:
            value += t1 + t2
        else:
            value += abs(t1 - t2)

    return value


def run_process(run_id: int, method: str, cmd: str, problem: str,
                output_dir: str, timeout: int) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    result = {'run_id': run_id, 'methods': method, 'output_dir': output_dir}
    params = parse_params(problem)

    if 'error' in params:
        result['error'] = params['error']
    else:
        result['parameters'] = params
        benchmark_dir = os.path.dirname(problem)
        original_problem_name = 'original_v{num_vehicles}_c{num_customers}_tw{time_window_parameter}_xy{xy_limit}_{problem_id}.txt'.format(**params)
        original_problem = os.path.join(benchmark_dir, original_problem_name)
        original_solution_name = 'solution_v{num_vehicles}_c{num_customers}_tw{time_window_parameter}_xy{xy_limit}_{problem_id}.txt'.format(**params)
        original_solution = os.path.join(benchmark_dir, original_solution_name)

        if not os.path.exists(original_problem):
            result['error'] = 'the original problem does not exist'
        elif not os.path.exists(original_solution):
            result['error'] = 'the original solution does not exist'
        else:
            output_file = os.path.join(output_dir, 'solution.txt')
            cmd_to_run = cmd.format(
                **{'original_problem': original_problem,
                   'original_solution': original_solution,
                   'perturbated_problem': problem,
                   'output': output_file})
            start = time.perf_counter()
            try:
                subprocess.run(cmd_to_run.split(), timeout=timeout)
            except Exception as e:
                result['error'] = e
            end = time.perf_counter()

            if os.path.exists(output_file):
                result['solved'] = 1
                result['time'] = end - start
                problem1 = util.VRPTWInstance.load(original_problem)
                problem2 = util.VRPTWInstance.load(problem)
                routes1 = util.load_solution(original_solution)
                routes2 = util.load_solution(output_file)
                result['cost'] = service_time_difference(
                    problem1, routes1, problem2, routes2)
            else:
                result['solved'] = 0

    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', '-m', type=str, required=True)
    parser.add_argument('--benchmark-dir', '-b', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--time-limit', '-t', type=int, default=300)
    parser.add_argument('--max-workers', '-w', type=int, default=None)
    parser.add_argument('--num-runs', '-n', type=int, default=1)
    args = parser.parse_args()

    with open(args.methods) as f:
        methods = json.load(f)

    time_limit = args.time_limit

    results = []

    with futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for filename in os.listdir(args.benchmark_dir):
            if 'perturbated' not in filename or 'solution' in filename:
                continue

            for method in methods:
                for i in range(args.num_runs):
                    j = len(results)
                    output_dir = os.path.join(args.output_dir, 'runs/run-{}'.format(j))
                    problem = os.path.join(args.benchmark_dir, filename)
                    future = executor.submit(run_process, i, method['name'],
                                             method['cmd'], problem, output_dir,
                                             time_limit)
                    results.append(future)

        output = [r.result() for r in results]

        with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
            json.dump(output, f, indent=4)


