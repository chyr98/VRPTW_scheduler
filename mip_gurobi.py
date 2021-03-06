from typing import List
import numpy as np
import VRPTW_util
import copy
import time
import sys, getopt
import json

import gurobipy as gp
from gurobipy import GRB

DEPOT_INDEX = 0
LARGE = 1e7


def model_initialization(problem: VRPTW_util.VRPTWInstance, equality=False):
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    # MIP model setup
    model = gp.Model("MainBS")
    model.__problem = problem

    # Variables

    # shape of (k, i, j) i,j=nodes, indicates vehicle k travel from i to j
    service_indicators = []
    # shape of (k, i), stands for the time when vehicle k leaves customer i, unbounded if not visited
    service_time = []

    for k in range(vehicles):

        level1 = []
        vehicle_service_time = []

        for i in range(nodes):

            level2 = []
            for j in range(nodes):

                level2.append(model.addVar(vtype=GRB.BINARY, name="I^%g_%g,%g" % (k, i, j)))

            level1.append(level2)
            vehicle_service_time.append(model.addVar(vtype=GRB.CONTINUOUS, name="s^%g_%g" % (k, i)))

        service_indicators.append(level1)
        service_time.append(vehicle_service_time)

    service_indicators = np.array(service_indicators)
    service_time = np.array(service_time)

    distance_map = [[] for _ in range(nodes)]
    time_window = [[], []]

    for i in range(nodes):
        time_window[0].append(problem.nodes[i].a)
        time_window[1].append(problem.nodes[i].b)

        for j in range(nodes):
            distance_map[i].append(problem.distance(i, j))

    # time window for the depot index
    time_window[0][0] = min(time_window[0])
    time_window[1][0] = max(time_window[1])+max(distance_map[0])

    distance_map = np.array(distance_map)  # shape of (i, j)
    time_window = np.array(time_window)  # shape of (2, i)

    # Constraints
    customer_indices = list(range(nodes))
    customer_indices.remove(DEPOT_INDEX)
    for k in range(vehicles):

        # Each vehicle needs a start and an end
        model.addConstr(gp.quicksum(service_indicators[k, DEPOT_INDEX, :].flatten()) == 1, "StartC^%g" % k)
        model.addConstr(gp.quicksum(service_indicators[k, :, DEPOT_INDEX].flatten()) == 1, "EndC^%g" % k)

        # Make sure Vehicles don't travel back to the location, except the depot index
        model.addConstr(gp.quicksum((service_indicators[k].diagonal())[customer_indices]) == 0, "backLoopC^%g" % k)

        for i in range(nodes):
            # Timewindow Constraints
            if i != DEPOT_INDEX:
                model.addConstr(
                    gp.quicksum(service_indicators[k, i, :].flatten()) * time_window[0][i] <= service_time[k, i],
                    "lb_twC^%g_%g" % (k, i))
                model.addConstr(
                    gp.quicksum(service_indicators[k, i, :].flatten()) * time_window[1][i] >= service_time[k, i],
                    "ub_twC^%g_%g" % (k, i))

            # No loop in each vehicle's route
            if not equality:
                model.addConstr(gp.quicksum(service_indicators[k, i, :].flatten()) <= 1, "travel_limit")
                model.addConstr(gp.quicksum(service_indicators[k, :, i].flatten()) <= 1, "travel_limit")

            # Travel time constraints, start time for each vehicle is 0
            model.addConstr(distance_map[DEPOT_INDEX, i]- service_time[k, i]
                            <= (1-service_indicators[k, DEPOT_INDEX, i]) * LARGE, "timeC^%g_%g,%g" % (k, 0, i))

            for j in customer_indices:
                # Travel time constraints
                model.addConstr(service_time[k, j] - service_time[k, i] + distance_map[j, i] <=
                                (1-service_indicators[k, j, i]) * LARGE, "timeC^%g_%g,%g" % (k, j, i))

            # Each vehicle must travel continuously
            model.addConstr(gp.quicksum(service_indicators[k, i, :].flatten()) ==
                            gp.quicksum(service_indicators[k, :, i].flatten()), "continuous_route^%g_%g" % (k, i))

    # Service completion constraints, node 0 is depot
    for j in customer_indices:
        if equality:
            model.addConstr(gp.quicksum(service_indicators[:, :, j].flatten()) == 1, "completion_%g" % j)
            model.addConstr(gp.quicksum(service_indicators[:, j, :].flatten()) == 1, "completion2_%g" % j)
        else:
            model.addConstr(gp.quicksum(service_indicators[:, :, j].flatten()) >= 1, "completion_%g" % j)

    model.__service_indicators = service_indicators
    model.__service_time = service_time

    return model, service_indicators, service_time, distance_map, time_window


def get_solution(vehicles, nodes, service_indicators_v, service_time):
    routes = [[] for _ in range(vehicles)]
    times = [[] for _ in range(vehicles)]
    for k in range(vehicles):
        curr_pos = np.argmax(service_indicators_v[k, DEPOT_INDEX, :])
        while curr_pos != DEPOT_INDEX:
            routes[k].append(int(curr_pos))
            times[k].append(service_time[k][curr_pos])
            curr_pos = np.argmax(service_indicators_v[k, curr_pos, :])

    return routes, times


def print_solution(solution_output_file, problem, cost_output_file, routes, times, performance):
    with open(solution_output_file, 'w') as f:
        output = {"name": problem.name, "routes": routes, "time": times}
        json.dump(output, f, indent=4)

    with open(cost_output_file, 'w') as f:
        json.dump(performance, f, indent=4)


def callback_on_solution(model, where):
    global mpp_start

    if where == GRB.Callback.MIPSOL:
        model.__cost_list.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        model.__time_list.append(time.perf_counter() - mpp_start)
        performance = {'cost': model.__cost_list, 'time': model.__time_list, 'optimal': False}
        vehicles = model.__problem.num_vehicles
        nodes = len(model.__problem.nodes)
        service_indicators = model.__service_indicators.flatten()
        service_indicators_v = np.array(model.cbGetSolution(service_indicators)).reshape(vehicles, nodes, nodes)
        service_time = model.__service_time.flatten()
        service_time_v = np.array(model.cbGetSolution(service_time)).reshape((vehicles, nodes))
        routes, times = get_solution(vehicles, nodes, service_indicators_v, service_time_v)
        print_solution(model.__solution_file, model.__problem, model.__cost_file, routes, times, performance)


def main(argv, mpp_start):
    file_name = ""
    solution_file = ""
    perturbated_file_name = ""
    perturbated_output_file = ""

    cost_output_file = ""
    threads = 1
    equality = False
    help_message = 'mip_gurobi.py -i <problem file> -I <perturbed problem file> -s <original solution file> ' \
                   '-O <perturbed solution outputfile> -c <cost outputfile> -t <time limit> -g <any>'
    try:
        opts, args = getopt.getopt(argv, "hi:I:s:O:c:t:g", ["ifile=", "Ifile=", "sfile=", "Ofile=", "cost=", "time=", "equality="])
        
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file_name = arg
        elif opt in ("-s", "--sfile"):
            solution_file = arg
        elif opt in ("-I", "--Ifile"):
            perturbated_file_name = arg
        elif opt in ("-O", "--Ofile"):
            perturbated_output_file = arg
        elif opt in ("-c", "--cost"):
            cost_output_file = arg
        elif opt in ("-t", "--threads"):
            threads = int(arg)
        elif opt in ("-g","--equality"):
            equality = True

    problem = VRPTW_util.VRPTWInstance.load(file_name)
    perturbated_problem = VRPTW_util.VRPTWInstance.load(perturbated_file_name)
    solution = VRPTW_util.load_routes(solution_file)
    solution_route = solution['routes']
    solution_time = solution['time']
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    model, service_indicators, service_time, distance_map, _ = model_initialization(perturbated_problem, equality)

    # timewindow for the original problem
    original_time_window = [[], []]
    for i in range(nodes):
        original_time_window[0].append(problem.nodes[i].a)
        original_time_window[1].append(problem.nodes[i].b)

    original_time_window[0][0] = min(original_time_window[0])
    original_time_window[1][0] = max(original_time_window[1]) + max(distance_map[0])

    # Service time of original solution
    service_time_v = np.zeros((vehicles, nodes))
    for k in range(vehicles):
        curr_pos = DEPOT_INDEX

        for i, pos in enumerate(solution_route[k]):
            service_time_v[k, pos] = solution_time[k][i]
            curr_pos = pos

        service_time_v[k, DEPOT_INDEX] = service_time_v[k, curr_pos] + distance_map[curr_pos, DEPOT_INDEX]
        # Add the DEPOT_INDEX at the end of routes for each vehicle, since every vehicle will travel back to depot
        solution_route[k].append(DEPOT_INDEX)

    # MPP solution
    difference_in_travels = []
    for k in range(vehicles):
        for i in range(1, nodes):
            route_change_indicator = model.addVar(vtype=GRB.CONTINUOUS, name="change_ind^%g_%g" % (k, i))
            model.addConstr(service_time[k, i] - service_time_v[k, i] <= route_change_indicator,
                            "route_check^%g_%g" % (k, i))
            model.addConstr(service_time_v[k, i] - service_time[k, i] <= route_change_indicator,
                            "route_check^%g_%g" % (k, i))
            difference_in_travels.append(route_change_indicator)

    model.setParam('Threads', threads)
    model.setObjective(gp.quicksum(difference_in_travels), GRB.MINIMIZE)
    model.__cost_list = []
    model.__time_list = []
    model.__cost_file = cost_output_file
    model.__solution_file = perturbated_output_file
    model.update()

    model.optimize(callback_on_solution)
    mpp_end = time.perf_counter()
    mpp_run_time = mpp_end - mpp_start

    if model.Status == gp.GRB.OPTIMAL:
        model.__cost_list.append(model.ObjVal)
        model.__time_list.append(mpp_run_time)
        performance = {"cost": model.__cost_list, "time": model.__time_list, "optimal": True}
        service_indicators_v = np.array([v.x for v in service_indicators.flatten()]).reshape((vehicles, nodes, nodes))
        service_time_v = np.array([v.x for v in service_time.flatten()]).reshape((vehicles, nodes))
        mpp_opt_routes, mpp_opt_times = get_solution(vehicles, nodes, service_indicators_v, service_time_v)
        print_solution(perturbated_output_file, perturbated_problem, cost_output_file, mpp_opt_routes, mpp_opt_times, performance)


if __name__ == '__main__':
    mpp_start = time.perf_counter()

    #message = "mip_gurobi.py -i benchmarks/original_v4_c64_tw16_xy16_0.txt -I benchmarks/perturbated4_v4_c64_tw16_xy16_0.txt " \
    #          "-s benchmarks/solution_v4_c64_tw16_xy16_0.txt -O perturbed_opt_solution.txt -c opt_cost.txt -t 180"
    
    #main(message.split()[1:])
    main(sys.argv[1:], mpp_start)
