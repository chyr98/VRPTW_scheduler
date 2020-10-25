import argparse
import copy
import itertools
import math
import random

import VRPTW_util as util
import problem_generator


def generate_mpp(name: str, num_vehicles: int, x_limit: int, y_limit: int,
                 num_nodes: int, tw_limit_ratio: float, num_perturbation: int,
                 seed: int = 1234) -> util.VRPTWInstance:
    """
    Generates an MPP in VRPTW Problem with the given parameters.

    :param name: the name of the problem
    :param num_vehicles: number of vehicles that can provide services
    :param x_limit: the max value of the x axis in the euclidian space
    :param y_limit: the max value of the y axis in the euclidian space
    :param num_nodes: number of customers in the problem
    :param tw_limit_ratio: the factor indicates how wide can the time window be in terms of multiple of the length of the diagnol of the euclidian space
    :param num_perturbation: the number of swaps
    :param seed: random seed
    :return: a well defined VRPTW Problem.
    """
    assert num_vehicles > 0 and x_limit >= 0 and y_limit >= 0 and num_nodes > 0
    assert (x_limit + 1) * (y_limit + 1) >= num_nodes

    random.seed(seed)

    original_name = 'original ' + name
    original_problem, original_routes = problem_generator.generate_problem(
        original_name, num_vehicles, x_limit, y_limit, num_nodes, tw_limit_ratio, seed=seed)

    tw_limit = int(math.sqrt(x_limit ** 2 + y_limit ** 2) * tw_limit_ratio)
    indices = list(range(1, num_nodes))
    combinations = list(itertools.combinations(indices, 2))
    perturbated_routes = []
    is_feasible = True

    while is_feasible:
        # swap customers in the routes
        perturbations = random.sample(combinations, num_perturbation)
        perturbation_dict = {i: i for i in range(1, num_nodes)}

        for p in perturbations:
            tmp = perturbation_dict[p[0]]
            perturbation_dict[p[0]] = perturbation_dict[p[1]]
            perturbation_dict[p[1]] = tmp

        perturbated_problem = copy.deepcopy(original_problem)
        perturbated_problem.name = 'perturbated ' + name
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

    return original_problem, original_routes, perturbated_problem, perturbated_routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='VRPTW instance')
    parser.add_argument('--num-vehicles', '-v', type=int, default=2)
    parser.add_argument('--x-limit', '-x', type=int, default=10)
    parser.add_argument('--y-limit', '-y', type=int, default=10)
    parser.add_argument('--num-customers', '-c', type=int, default=5)
    parser.add_argument('--tw-limit-ratio', '-t', type=float, default=0.3)
    parser.add_argument('--num-perturbation', '-m', type=int, default=3)
    parser.add_argument('--original-problem', '-o', type=str, required=True)
    parser.add_argument('--perturbated-problem', '-p', type=str, required=True)
    parser.add_argument('--solution', '-s', type=str)
    parser.add_argument('--perturbated-solution', '-q', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    name = args.name
    num_vehicles = args.num_vehicles
    num_customers = args.num_customers
    x_limit = args.x_limit
    y_limit = args.y_limit
    tw_limit_ratio = args.tw_limit_ratio
    num_perturbation = args.num_perturbation
    seed = args.seed
    problem1, routes1, problem2, routes2 = generate_mpp(
        name, num_vehicles, x_limit, y_limit, num_customers, tw_limit_ratio,
        num_perturbation, seed=seed)
    problem1.dump(args.original_problem)
    problem2.dump(args.perturbated_problem)

    if args.solution is not None:
        util.dump_routes(problem1.name, routes1, args.solution)

    if args.perturbated_solution is not None:
        util.dump_routes(problem2.name, routes2, args.perturbated_solution)
