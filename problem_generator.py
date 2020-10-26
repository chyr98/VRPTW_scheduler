import argparse
import itertools
import math
import random

import VRPTW_util as util


def generate_problem(name: str, num_vehicles: int, x_limit: int, y_limit: int,
                     num_nodes: int,
                     tw_limit: int) -> util.VRPTWInstance:
    """
    Generates a VRPTW Problem with the given parameters.

    :param name: the name of the problem
    :param num_vehicles: number of vehicles that can provide services
    :param x_limit: the maximum value of the x axis in the euclidian space
    :param y_limit: the maximum value of the y axis in the euclidian space
    :param num_nodes: number of customers in the problem
    :param tw_limit: the half of maximum width of the time windows
    :param seed: random seed
    :return: a well defined VRPTW Problem.
    """
    assert num_vehicles > 0 and x_limit >= 0 and y_limit >= 0 and num_nodes > 0
    assert (x_limit + 1) * (y_limit + 1) >= num_nodes

    all_x_y = list(itertools.product(range(x_limit + 1), range(y_limit + 1)))
    x_y_list = random.sample(all_x_y, num_nodes)
    nodes = [util.Node(i, t[0], t[1]) for i, t in enumerate(x_y_list)]

    # To make sure the problem is feasible, we will generate random routes and set the time windows
    # of tasks to match those routes.
    indices = list(range(1, num_nodes))
    random.shuffle(indices)
    # randomly divide [0, #customers - 1] to #vehicles adjacent ranges
    k = num_vehicles - 1
    break_points = sorted(random.choices(list(range(num_nodes - 1)), k=k))
    break_points.append(num_nodes - 1)
    routes = []
    start = 0

    for i in range(num_vehicles):
        end = break_points[i]
        routes.append(indices[start:end])
        start = end

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
    parser.add_argument('--name', '-n', type=str, default='VRPTW instance',
                        help='The name of the problem.')
    parser.add_argument('--num-vehicles', '-v', type=int, default=2,
                        help='The number of vehicles.')
    parser.add_argument('--x-limit', '-x', type=int, default=10,
                        help='The maixmum value of x-axis.')
    parser.add_argument('--y-limit', '-y', type=int, default=10,
                        help='The maixmum value of y-axis.')
    parser.add_argument('--num-customers', '-c', type=int, default=5,
                        help='The number of customers including the depot.')
    parser.add_argument('--tw-limit', '-t', type=int, default=4,
                        help='The half of the maximum width of time windows.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='The path to save the generated problem.')
    parser.add_argument('--solution', '-s', type=str,
                        help='The path to save the solution used for generating the problem.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    name = args.name
    num_vehicles = args.num_vehicles
    num_customers = args.num_customers
    x_limit = args.x_limit
    y_limit = args.y_limit
    tw_limit = args.tw_limit

    print('generating a problem...')
    problem, routes = generate_problem(
        name, num_vehicles, x_limit, y_limit, num_customers, tw_limit)
    print('saving the problem...')
    problem.dump(args.output)

    if args.solution is not None:
        print('saving a solution for the problem...')
        util.dump_routes(problem.name, routes, args.solution)
