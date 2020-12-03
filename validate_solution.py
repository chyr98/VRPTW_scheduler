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
    result = problem.get_time(routes)
    total_cost = 0.0

    if result is None:
        print('invalid solution')
    else:
        for i, r in enumerate(routes):
            print('vehicle {}'.format(i))
            text = '\t0: 0.0'
            last = -1
            last_time = 0.0

            for c in r:
                t = result[c][1]
                text += ', {}: {:.2f}'.format(c, t)
                last = c
                last_time = t

            if last != -1:
                total_time = last_time + problem.distance(last, 0)
                text += ', 0: {:.2f}'.format(total_time)
                total_cost += total_time

            print(text)

        print('total cost: {:.2f}'.format(total_time))
