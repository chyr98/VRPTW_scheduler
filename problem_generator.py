import argparse
import itertools
import math
import random

import VRPTW_util as util


def generate_problem(name: str, num_vehicles: int, x_limit: int, y_limit: int,
                     num_nodes: int, tw_limit_ratio: float,
                     seed: int = 1234) -> util.VRPTWInstance:
    """
    Generates a VRPTW Problem with the given parameters.

    Precondition: lb_distance <= ub_distance <= 2 * lb_distance (to maintain the triangle inequality)

    :param num_nodes: number of customers in the problem
    :param num_vehicles: number of vehicles that can provide services
    :param lb_distance: the lower bound of the distance from any location to another
    :param ub_distance: the upper bound of the distance from any location to another
    :param tw_variance: the factor indicates how wide can the time window be in terms of multiple of ub_distance
    :return: a well defined VRPTW Problem.
    """
    assert num_vehicles > 0 and x_limit >= 0 and y_limit >= 0 and num_nodes > 0
    assert (x_limit + 1) * (y_limit + 1) >= num_nodes

    random.seed(seed)

    all_x_y = list(itertools.product(range(x_limit + 1), range(y_limit + 1)))
    x_y_list = random.sample(all_x_y, num_nodes)
    nodes = [util.Node(i, t[0], t[1]) for i, t in enumerate(x_y_list)]

    # To make sure the problem is feasible, we will generate random routes and set the time windows
    # of tasks to match those routes.
    indices = list(range(1, num_nodes))
    random.shuffle(indices)
    k = num_vehicles - 1
    break_points = sorted(random.choices(list(range(num_nodes - 1)), k=k))
    break_points.append(num_nodes - 1)
    routes = []
    start = 0

    for i in range(num_vehicles):
        end = break_points[i]
        routes.append(indices[start:end])
        start = end

    tw_limit = int(math.sqrt(x_limit ** 2 + y_limit ** 2) * tw_limit_ratio)

    for r in routes:
        prev = 0
        t = 0.0

        for cur in r:
            t += nodes[prev].distance(nodes[cur])
            a = max(math.floor(t) - random.randint(0, tw_limit), 0)
            b = math.ceil(t) + random.randint(0, tw_limit)
            nodes[cur].update_time_window(a, b)
            prev = cur

    return util.VRPTWInstance(name, num_vehicles, nodes), routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='VRPTW instance')
    parser.add_argument('--num-vehicles', '-v', type=int, default=2)
    parser.add_argument('--x-limit', '-x', type=int, default=10)
    parser.add_argument('--y-limit', '-y', type=int, default=10)
    parser.add_argument('--num-customers', '-c', type=int, default=5)
    parser.add_argument('--tw-limit-ratio', '-t', type=float, default=0.3)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--solution', '-s', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    name = args.name
    num_vehicles = args.num_vehicles
    num_customers = args.num_customers
    x_limit = args.x_limit
    y_limit = args.y_limit
    tw_limit_ratio = args.tw_limit_ratio
    seed = args.seed
    problem, routes = generate_problem(
        name, num_vehicles, x_limit, y_limit, num_customers, tw_limit_ratio,
        seed=seed)
    problem.dump(args.output)

    if args.solution is not None:
        util.dump_routes(problem.name, routes, args.solution)
