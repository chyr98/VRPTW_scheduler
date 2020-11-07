from typing import List
import numpy as np
import VRPTW_util
import copy
import time
import sys, getopt

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

    # Minimize the total travel time, notice service time on depot stands for the time when vehicle get back
    model.setObjective(gp.quicksum(service_time[:, DEPOT_INDEX]), GRB.MINIMIZE)

    return model, service_indicators, service_time, distance_map, time_window


def print_solution(file_name, problem, route):
    vehicles, customers = problem.num_vehicles, len(problem.nodes)
    f = open(file_name, "w")

    f.write("# " + problem.name + "\n")
    f.write(str(vehicles) + "\n")

    for k in range(vehicles):
        for i in route[k]:
            if i != DEPOT_INDEX:
                f.write(str(i) + " ")
        f.write("\n")

    f.close()


def recursive_step(tree, guide, ordered_vehicles, routes, customers_waiting, old_available_decisions, upper_bound, depth):
    model = tree.model
    service_indicators = tree.service_indicators
    curr_node = tree.curr_node
    available_decisions = copy.deepcopy(old_available_decisions)

    # when all customers are served, optimizes the model since it is a leaf
    if not customers_waiting:
        model.optimize()
        if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
            best_sol = LARGE
        else:
            best_sol = model.objVal

        tree.backtrack(best_sol)
        return best_sol, routes

    lower_bound = relaxed_solution(model)
    if lower_bound >= upper_bound:
        # already found a better solution
        tree.backtrack(LARGE)
        return LARGE, routes

    best_routes = routes
    best_sol = upper_bound
    for k in ordered_vehicles:
        # check the similarity between guide route and current route, first search on the route close to guide route
        target_node = -1
        if routes[k]:
            i = routes[k][-1]
            if len(guide[k]) > len(routes[k]) and guide[k][len(routes[k])-1] == i:
                target_node = guide[k][len(routes[k])]
        else:
            i = 0
            target_node = guide[k][0]

        choices = list(set(available_decisions[i]).intersection(customers_waiting))
        if target_node in choices:
            choices.remove(target_node)
            choices.insert(0, target_node)

        for j in choices:
            child = VRPTWNode((k, i, j), curr_node, [])
            curr_node.children.append(child)

            # dive to this node
            child.decision = (k, i, j)
            child.constraint = model.addConstr(service_indicators[k, i, j] == 1, "decision^%g_%g,%g" % (k, i, j))
            model.update()

            new_routes = copy.deepcopy(routes[:])
            new_customers_line = copy.deepcopy(customers_waiting[:])

            new_routes[k].append(j)
            new_customers_line.remove(j)

            tree.curr_node = child
            new_sol, new_routes = recursive_step(tree, guide, ordered_vehicles[1:] + ordered_vehicles[0:1],
                                                 new_routes, new_customers_line, available_decisions, best_sol, depth+1)
            #print("depth %g :" % depth + str(routes) + " " + str(customers_waiting) + " with solution : " + str(new_sol))

            if new_sol < best_sol:
                best_sol, best_routes = new_sol, new_routes
            if best_sol <= lower_bound:
                # backtrack
                tree.backtrack(best_sol)
                return best_sol, best_routes
    # backtrack
    tree.backtrack(best_sol)
    return best_sol, best_routes


def backtrack_search(model, original_route, service_indicators, service_time, distance_map, time_window):
    vehicles = service_indicators.shape[0]
    nodes = service_indicators.shape[1]
    tree = VRPTWTree(model, service_indicators)
    curr_routes = [[] for _ in range(vehicles)]
    available_decisions = [[] for _ in range(nodes)]
    customers_waiting = list(range(1, nodes))

    for i in range(nodes):
        for j in range(nodes):
            if i != j and time_window[0][i] + distance_map[i][j] <= time_window[1][j]:
                available_decisions[i].append(j)

    solution, opt_routes = recursive_step(tree, original_route, list(range(vehicles)), curr_routes, customers_waiting, available_decisions, LARGE, 0)

    return solution, opt_routes


def main(argv):
    file_name = ""
    output_file = ""
    perturbated_file_name = ""
    perturbated_output_file = ""
    help_message = 'backtrack_search.py -i <problem file> -I <perturbed problem file> -s <original solution file> ' \
                   '-O <perturbed solution outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:I:s:O:", ["ifile=", "Ifile=", "sfile=", "Ofile"])
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

    problem = VRPTW_util.VRPTWInstance.load(file_name)
    perturbated_problem = VRPTW_util.VRPTWInstance.load(perturbated_file_name)
    solution_route = VRPTW_util.load_solution(solution_file)
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    model, service_indicators, service_time, distance_map, time_window = model_initialization(problem)
    '''
    original_p_start = time.perf_counter()
    model.optimize()
    original_p_end = time.perf_counter()
    ori_run_time = original_p_end - original_p_start

    indicators_v = np.array([v.x for v in service_indicators.flatten()]).reshape(vehicles, nodes, nodes)
    service_time_v = np.array([v.x for v in service_time.flatten()]).reshape(vehicles, nodes)
    ori_opt_sol = model.objVal

    original_route = [[] for _ in range(vehicles)]
    for k in range(vehicles):
        curr_customer = np.argmax(indicators_v[k, DEPOT_INDEX])

        for _ in range(round(float(np.sum(indicators_v[k])))):
            original_route[k].append(curr_customer)
            curr_customer = np.argmax(indicators_v[k, curr_customer])

    print_solution(output_file, problem, original_route)
    print("The optimal cost for the original VRPTW problem is {}".format(ori_opt_sol))
    print("The VRPTW optimizer took {} seconds".format(ori_run_time))
    '''
    # Service time of original solution
    service_time_v = np.zeros((vehicles, nodes))
    for k in range(vehicles):
        curr_pos = DEPOT_INDEX
        for pos in solution_route[k]:
            service_time_v[k, pos] = service_time_v[k, curr_pos] + distance_map[curr_pos, pos]
            curr_pos = pos
        service_time_v[k, DEPOT_INDEX] = service_time_v[k, curr_pos] + distance_map[curr_pos, DEPOT_INDEX]

    # timewindow for perturbed problem
    new_time_window = [[], []]
    for i in range(nodes):
        new_time_window[0].append(perturbated_problem.nodes[i].a)
        new_time_window[1].append(perturbated_problem.nodes[i].b)

    new_time_window[0][0] = min(new_time_window[0])
    new_time_window[1][0] = max(new_time_window[1])+max(distance_map[0])

    # Reset model
    model.reset()
    # MPP solution
    difference_in_travels = []
    for k in range(vehicles):
        for i in range(nodes):
            route_change_indicator = model.addVar(vtype=GRB.CONTINUOUS, name="change_ind^%g_%g" % (k, i))
            model.addConstr(service_time[k, i] - service_time_v[k, i] <= route_change_indicator,
                            "route_check^%g_%g" % (k, i))
            model.addConstr(service_time_v[k, i] - service_time[k, i] <= route_change_indicator,
                            "route_check^%g_%g" % (k, i))
            difference_in_travels.append(route_change_indicator)

            # remove the original timewindow constraints
            model.remove(model.getConstrByName("lb_twC^%g_%g" % (k, i)))
            model.remove(model.getConstrByName("ub_twC^%g_%g" % (k, i)))
            model.update()

            # new timewindow constraints
            model.addConstr(
                gp.quicksum(service_indicators[k, i, :].flatten()) * new_time_window[0][i] <= service_time[k, i],
                "lb_twC^%g_%g" % (k, i))
            model.addConstr(
                gp.quicksum(service_indicators[k, i, :].flatten()) * new_time_window[1][i] >= service_time[k, i],
                "ub_twC^%g_%g" % (k, i))

    model.setObjective(gp.quicksum(difference_in_travels), GRB.MINIMIZE)

    mpp_start = time.perf_counter()
    mpp_opt_sol, mpp_opt_routes = backtrack_search(model, solution_route, service_indicators, service_time,
                                                   distance_map, new_time_window)
    mpp_end = time.perf_counter()
    mpp_run_time = mpp_end - mpp_start

    print_solution(perturbated_output_file, perturbated_problem, mpp_opt_routes)
    print("The optimal cost for the MPP is {}".format(mpp_opt_sol))
    print("The MPP optimizer took {} seconds".format(mpp_run_time))


if __name__ == '__main__':
    message = "backtrack_search.py -i benchmarks/original_v2_c4_tw4_xy16_0.txt -I benchmarks/perturbated1_v2_c4_tw4_xy16_0.txt " \
              "-s benchmarks/solution_v2_c4_tw4_xy16_0.txt -O perturbed_opt_solution.txt"

    #main(message.split()[1:])
    main(sys.argv[1:])

