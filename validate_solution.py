import argparse
from typing import List

import VRPTW_util as util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p', type=str, required=True)
    parser.add_argument('--solution', '-s', type=str, required=True)
    args = parser.parse_args()

    problem = util.VRPTWInstance.load(args.problem)
    solution = util.load_routes(args.solution)
    routes = solution['routes']
    times = solution['time']
    total_cost = 0.0
    feasible = True
    eps = 1e-7

    for i, (r, ts) in enumerate(zip(routes, times)):
        print('vehicle {}'.format(i))
        text = '\t0: 0.0'
        last = 0
        last_time = 0.0

        for c, t in zip(r, ts):
            a = problem.nodes[c].a
            b = problem.nodes[c].b
            arrive_time = problem.distance(last, c) + last_time

            if arrive_time - t > eps:
                print('vehicle {} cannot serve {} at {:.2f} < {:.2f} (cannot arrive)'.format(i, c, t, arrive_time))
                feasible = False
                break
            elif a - t > eps:
                print('vehicle {} cannot serve {} at {:.2f} < {:.2f} (violates a time window)'.format(i, c, t, a))
                feasible = False
                break
            elif t - b > eps:
                print('vehicle {} cannot serve {} at {:.2f} > {:.2f} (violate a time window)'.format(i, c, t, b))
                feasible = False
                break

            text += ', {}: {:.2f}'.format(c, t)
            last = c
            last_time = t

        if not feasible:
            break

        if last != 0:
            total_time = last_time + problem.distance(last, 0)
            text += ', 0: {:.2f}'.format(total_time)
            total_cost += total_time

        print(text)

    if feasible:
        print('total cost: {:.2f}'.format(total_time))
