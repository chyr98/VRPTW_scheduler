import argparse
import copy
import itertools
import math
import os
import random
from typing import List

import VRPTW_util as util
import problem_generator


def generate_mpp(original_problem: util.VRPTWInstance,
                 original_routes: List[List[int]], perturbated_name: str,
                 tw_limit: int, num_perturbations: int) -> util.VRPTWInstance:
    """
    Generates an MPP in VRPTW Problem with the given parameters.

    :param original_problem: the original VRPTW instance
    :param original_routes: a solution for the original instance
    :param perturbated_name: the name of the perturbated problem
    :param tw_limit: the half of maximum width of the time windows
    :param num_perturbations: the number of swaps
    :return: a well defined VRPTW Problem.
    """
    num_nodes = len(original_problem.nodes)
    indices = list(range(1, num_nodes))
    combinations = list(itertools.combinations(indices, 2))
    perturbated_routes = []
    is_feasible = True

    while is_feasible:
        # swap customers in the routes
        perturbations = random.sample(combinations, num_perturbations)
        perturbation_dict = {i: i for i in range(1, num_nodes)}

        for p in perturbations:
            tmp = perturbation_dict[p[0]]
            perturbation_dict[p[0]] = perturbation_dict[p[1]]
            perturbation_dict[p[1]] = tmp

        perturbated_problem = copy.deepcopy(original_problem)
        perturbated_problem.name = perturbated_name
        perturbated_routes = []

        for r in original_routes:
            prev = 0
            t = 0.0
            route = []

            for cur in r:
                if perturbation_dict[cur] != cur:
                    cur = perturbation_dict[cur]
                    t += perturbated_problem.distance(prev, cur)
                    a = max(math.floor(t) - random.randint(0, tw_limit), 0)
                    b = math.ceil(t) + random.randint(0, tw_limit)
                    perturbated_problem.nodes[cur].update_time_window(a, b)
                else:
                    distance = perturbated_problem.distance(prev, cur)
                    t = max(perturbated_problem.nodes[cur].a, t + distance)

                    if t > perturbated_problem.nodes[cur].b:
                        a = max(math.floor(t) - random.randint(0, tw_limit), 0)
                        b = math.ceil(t) + random.randint(0, tw_limit)
                        perturbated_problem.nodes[cur].update_time_window(a, b)

                route.append(cur)
                prev = cur

            perturbated_routes.append(route)

        is_feasible = perturbated_problem.get_time(original_routes) is not None

    return perturbated_problem, perturbated_routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='VRPTW instance',
                        help='The name of the problem. This is needed only when generating the original problem.')
    parser.add_argument('--num-vehicles', '-v', type=int, default=2,
                        help='The number of vehicles. This is needed only when generating the original problem.')
    parser.add_argument('--x-limit', '-x', type=int, default=10,
                        help='The maixmum value of x-axis. This is needed only when generating the original problem.')
    parser.add_argument('--y-limit', '-y', type=int, default=10,
                        help='The maixmum value of y-axis. This is needed only when generating the original problem.')
    parser.add_argument('--num-customers', '-c', type=int, default=5,
                        help='The number of customers. This is needed only when generating the original problem.')
    parser.add_argument('--tw-limit', '-t', type=int, default=4,
                        help='The half of the maximum width of time windows.')
    parser.add_argument('--num-perturbations', '-m', type=int, default=3,
                        help='The number of perturbations.')
    parser.add_argument('--original-problem', '-o', type=str, required=True,
                        help='The path of the original problem. It will be generated if it or the original solution does not exist.')
    parser.add_argument('--perturbated-problem', '-p', type=str, required=True,
                        help='The number of swaps.')
    parser.add_argument('--solution', '-s', type=str, required=True,
                        help='The path of the original solution. It will be generated if it or the original problem does not exist.')
    parser.add_argument('--perturbated-solution', '-q', type=str,
                        help='The path to save the solution used for generating the perturbated problem. It is not saved if not specified.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    problem1 = None
    routes1 = None

    if args.original_problem is not None \
            and args.solution is not None \
            and os.path.exists(args.original_problem) \
            and os.path.exists(args.solution):
        print(
            'using an existing problem and solution as the original problem and solution...')
        problem1 = util.VRPTWInstance.load(args.original_problem)
        routes1 = util.load_solution(args.solution)
    else:
        print('generating new problem and solution as the original problem and solution...')
        problem1, routes1 = problem_generator.generate_problem(
            args.name, args.num_vehicles, args.x_limit, args.y_limit,
            args.num_customers, args.tw_limit)

        print('saving the original problem...')
        problem1.dump(args.original_problem)

        if args.solution is not None:
            print('saving a solution for the original problem...')
            util.dump_routes(problem1.name, routes1, args.solution)

    print('generating a perturbated problem...')
    perturbated_name = 'perturbated ' + problem1.name
    problem2, routes2 = generate_mpp(
        problem1, routes1, perturbated_name, args.tw_limit, args.num_perturbations)
    print('saving the perturbated problem...')
    problem2.dump(args.perturbated_problem)

    if args.perturbated_solution is not None:
        print('saving a solution for the perturbated problem...')
        util.dump_routes(problem2.name, routes2, args.perturbated_solution)
