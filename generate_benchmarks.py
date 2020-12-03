import os
import random

import mpp_generator
import problem_generator
import VRPTW_util as util


NUM_VEHICLES = [16]
NUM_CUSTOMERS = [64, 128, 256]
TIME_WINDOW_LIMIT = [16]
AXIS_LIMIT = [32]
NUM_PERTURBATIONS = [2, 4, 8, 16]
NUM_INSTANCES = 1

random.seed(1234)
os.makedirs('benchmarks', exist_ok=True)

for v in NUM_VEHICLES:
    for c in NUM_CUSTOMERS:
        if v >= c:
            continue

        for tw_limit in TIME_WINDOW_LIMIT:
            for x_y_limit in AXIS_LIMIT:
                for i in range(NUM_INSTANCES):
                    name = 'problem {} of {} vehicle(s) {} customer(s), b - a < 2 * {}, x, y <= {}'.format(
                        i, v, c, tw_limit, x_y_limit)
                    problem1, routes1 = problem_generator.generate_problem(
                        name, v, x_y_limit, x_y_limit, c, tw_limit)
                    problem_filename = 'original_v{}_c{}_tw{}_xy{}_{}.txt'.format(
                        v, c, tw_limit, x_y_limit, i)
                    solution_filename = 'solution_v{}_c{}_tw{}_xy{}_{}.txt'.format(
                        v, c, tw_limit, x_y_limit, i)
                    problem1.dump(os.path.join('benchmarks', problem_filename))
                    problem1.dump_routes_with_time(routes1, os.path.join('benchmarks', solution_filename))

                    for j in NUM_PERTURBATIONS:
                        if (c - 1) * (c - 2) / 2 < j:
                            continue

                        perturbated_name = '{} perturbated {}'.format(j, name)
                        problem2, routes2 = mpp_generator.generate_mpp(
                            problem1, routes1, perturbated_name, tw_limit, j)
                        perturbated_filename = 'perturbated{}_v{}_c{}_tw{}_xy{}_{}.txt'.format(
                            j, v, c, tw_limit, x_y_limit, i)
                        problem2.dump(
                            os.path.join('benchmarks', perturbated_filename))
