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


class VRPTWNode:
    def __init__(self, decision, parent, children):
        self.decision = decision
        self.constraint = None
        self.parent = parent
        self.children = children


class VRPTWTree:
    def __init__(self, model, service_indicators):
        self.model = model
        self.service_indicators = service_indicators
        self.root = VRPTWNode([], None, [])
        self.curr_node = self.root

    def backtrack(self, best_sol):
        if self.curr_node.parent is not None:
            self.model.remove(self.curr_node.constraint)
            self.curr_node = self.curr_node.parent
            self.model.update()


def relaxed_solution(model: gp.Model):
    relaxed_model = model.relax()
    relaxed_model.optimize()
    if relaxed_model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        return LARGE
    else:
        return relaxed_model.objVal


def model_initialization(problem: VRPTW_util.VRPTWInstance):
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    # MIP model setup
    model = gp.Model("MainBS")

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
        model.addConstr(gp.quicksum(service_indicators[:, :, j].flatten()) >= 1, "completion_%g" % j)

    return model, service_indicators, service_time, distance_map, time_window


def print_solution(route_output_file, problem, cost_output_file, route, performance):
    vehicles, customers = problem.num_vehicles, len(problem.nodes)
    f = open(route_output_file, "w")

    f.write("# " + problem.name + "\n")
    f.write(str(vehicles) + "\n")

    for k in range(vehicles):
        for i in route[k]:
            if i != DEPOT_INDEX:
                f.write(str(i) + " ")
        f.write("\n")

    f.close()

    with open(cost_output_file, 'w') as f:
        json.dump(performance, f)


def main(argv, hint=False):
    file_name = ""
    solution_file = ""
    perturbated_file_name = ""
    perturbated_output_file = ""
    cost_output_file = ""
    max_run_time = 30
    help_message = 'backtrack_search.py -i <problem file> -I <perturbed problem file> -s <original solution file> ' \
                   '-O <perturbed solution outputfile> -c <cost outputfile> -t <time limit>'
    try:
        opts, args = getopt.getopt(argv, "hi:I:s:O:c:t:", ["ifile=", "Ifile=", "sfile=", "Ofile=", "cost=", "time="])
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
        elif opt in ("-t", "--time"):
            max_run_time = int(arg)

    problem = VRPTW_util.VRPTWInstance.load(file_name)
    perturbated_problem = VRPTW_util.VRPTWInstance.load(perturbated_file_name)
    solution_route = VRPTW_util.load_solution(solution_file)
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    model, service_indicators, service_time, distance_map, time_window = model_initialization(perturbated_problem)

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
        if hint:
            hint_pri = len(solution_route[k])
        for pos in solution_route[k]:
            service_time_v[k, pos] = max(service_time_v[k, curr_pos] + distance_map[curr_pos, pos], original_time_window[0][pos])

            # Use the original route as a hint for the mpp route
            if hint_pri > 0 and hint:
                service_indicators[k, curr_pos, pos].setAttr(GRB.Attr.VarHintVal, 1.0)
                service_indicators[k, curr_pos, pos].setAttr(GRB.Attr.VarHintPri, hint_pri)

                hint_pri = hint_pri - 1
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

    model.setObjective(gp.quicksum(difference_in_travels), GRB.MINIMIZE)
    model.update()

    time_limit = 0
    mpp_run_time = 0
    performance = {"cost": [], "time": []}
    while mpp_run_time >= time_limit and time_limit <= max_run_time:
        time_limit = time_limit + 10

        mpp_start = time.perf_counter()
        model.reset()
        model.setParam("TimeLimit", time_limit)
        model.optimize()

        if model.MIPGap != GRB.INFINITY:
            mpp_opt_sol = model.objVal
            performance["cost"].append(mpp_opt_sol)
            print("The optimal cost for the MPP is {}".format(mpp_opt_sol))
            print("The MPP optimizer took {} seconds".format(mpp_run_time))
        else:
            performance["cost"].append("--")
        performance["time"].append(time_limit)

        mpp_end = time.perf_counter()
        mpp_run_time = mpp_end - mpp_start

    service_indicators_v = np.array([v.x for v in service_indicators.flatten()]).reshape((vehicles, nodes, nodes))

    mpp_opt_routes = [[] for _ in range(vehicles)]
    for k in range(vehicles):
        curr_pos = np.argmax(service_indicators_v[k, DEPOT_INDEX, :])
        while curr_pos != DEPOT_INDEX:
            mpp_opt_routes[k].append(curr_pos)
            curr_pos = np.argmax(service_indicators_v[k, curr_pos, :])

    print_solution(perturbated_output_file, perturbated_problem, cost_output_file, mpp_opt_routes, performance)


if __name__ == '__main__':
    #message = "backtrack_search.py -i benchmarks/original_v4_c64_tw16_xy16_0.txt -I benchmarks/perturbated4_v4_c64_tw16_xy16_0.txt " \
    #          "-s benchmarks/solution_v4_c64_tw16_xy16_0.txt -O perturbed_opt_solution.txt -c opt_cost.txt -t 180"

    #main(message.split()[1:], hint=True)
    main(sys.argv[1:], hint=True)

