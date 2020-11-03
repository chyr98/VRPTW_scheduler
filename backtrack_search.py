from typing import List
import numpy as np
import VRPTW_util

import gurobipy as gp
from gurobipy import GRB


def relaxed_solution(model: gp.Model):
    relaxed_model = model.relax()
    relaxed_model.optimize()
    return relaxed_model.objVal


class VRPTWNode:
    def __init__(self, expressions: List[gp.Constr], parent, children):
        self.constr = expressions
        self.parent = parent
        self.children = children
        self.optimal = 0


class VRPTWTree:
    def __init__(self, problem: VRPTW_util.VRPTWInstance):
        self.model, self.service_indicators, _, self.distance_map, self.timewindow = model_initialization(problem)
        self.root = VRPTWNode([], None, [])
        self.current_loc = self.root

    def dive(self, index):
        self.model.addConstrs()


def model_initialization(problem: VRPTW_util.VRPTWInstance):
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    # MIP model setup
    model = gp.Model("MainBS")

    # Variables

    # shape of (k, p, i, j) p=nodes-1, i,j=nodes+1, indicates vehicle k travel from i to j in pth travel
    # take in account of an additional variable to stands for idling after travel
    service_indicators = []
    # shape of (k, p), stands for the time when vehicle k leaves pth customer
    service_time = []

    for k in range(vehicles):

        level1 = []
        vehicle_service_time = []

        for p in range(nodes - 1):

            level2 = []
            for i in range(nodes + 1):

                level3 = []
                for j in range(nodes + 1):
                    level3.append(model.addVar(vtype=GRB.BINARY, name="I^%g_%g,%g,%g" % (k, i, j, p)))

                level2.append(level3)

            level1.append(level2)
            vehicle_service_time.append(model.addVar(vtype=GRB.CONTINUOUS, name="s^%g_%g" % (k, p)))

        service_indicators.append(level1)
        service_time.append(vehicle_service_time)

    service_indicators = np.array(service_indicators)
    service_time = np.array(service_time)

    distance_map = [[] for _ in range(nodes + 1)]
    time_window = [[], []]

    for i in range(nodes):
        time_window[0].append(problem.nodes[i].a)
        time_window[1].append(problem.nodes[i].b)

        for j in range(nodes):
            distance_map[i].append(problem.distance(i, j))
        # distance for idle index
        distance_map[i].append(0)

    distance_map[-1] = [0] * (nodes + 1)
    # time window for the idle index
    time_window[0].append(min(time_window[0]))
    time_window[1].append(max(time_window[1]))

    distance_map = np.array(distance_map)  # shape of (i, j)
    time_window = np.array(time_window)  # shape of (2, j)

    # Constraints

    # Vehicles have to start from node 0
    model.addConstr(gp.quicksum(service_indicators[:, 0, list(range(1, nodes + 1)), :].flatten()) == 0,
                    "Start_positionC")
    # idle index cannot travel to other location
    model.addConstr(gp.quicksum(service_indicators[:, :, -1, :-1].flatten()) == 0, "Stay_idle_con")

    for k in range(vehicles):

        for p in range(nodes - 1):
            # Constraints that make sure each vehicle travels on one arc at each time
            model.addConstr(gp.quicksum(service_indicators[k, p, :, :].flatten()) <= 1, "binaryC^%g_%g" % (k, p))

            # Make sure Vehicles don't travel back to the location, except the idle index
            model.addConstr(gp.quicksum((service_indicators[k, p].diagonal())[:-1]) == 0, "backLoopC^%g_%g" % (k, p))

            # Timewindow Constraints
            model.addConstr(
                gp.quicksum((service_indicators[k, p, :, :] * time_window[0]).flatten()) <= service_time[k, p],
                "lb_twC^%g_%g" % (k, p))
            model.addConstr(
                gp.quicksum((service_indicators[k, p, :, :] * time_window[1]).flatten()) >= service_time[k, p],
                "ub_twC^%g_%g" % (k, p))

            if p == 0:
                # Travel time constraints, start time for each vehicle is 0
                model.addConstr(gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                                <= service_time[k, p], "timeC^%g_%g" % (k, p))
            else:
                # Travel time constraints
                model.addConstr(
                    service_time[k, p - 1] + gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                    <= service_time[k, p], "timeC^%g_%g" % (k, p))

                # Each vehicle must travel continuously
                model.addConstr(gp.quicksum(service_indicators[k, p, :, :].flatten())
                                <= gp.quicksum(service_indicators[k, p - 1, :, :].flatten()),
                                "continuous_route^%g_%g" % (k, p))

                for i in range(nodes + 1):
                    for j in range(nodes + 1):
                        indices = list(range(nodes + 1))
                        indices.remove(j)

                        for var in service_indicators[k, p, indices, :].flatten():
                            # Constraints that make sure vehicles start travel from where they at in next time period.
                            model.addConstr(service_indicators[k, p - 1, i, j] + var <= 1, "connection^%g_%g" % (k, p))
    # Service completion constraints, node 0 is depot
    for j in range(1, nodes):
        model.addConstr(gp.quicksum(service_indicators[:, :, :, j].flatten()) >= 1, "completion_%g" % j)

    model.setObjective(gp.quicksum(service_time[:, -1]), GRB.MINIMIZE)

    print(time_window)
    print(distance_map)
    return model, service_indicators, service_time, distance_map, time_window


def print_solution(file_name, problem, service_indicators):
    vehicles, customers = problem.num_vehicles, len(problem.nodes)
    f = open(file_name, "w")

    f.write("# " + problem.name + "\n")
    f.write(str(vehicles) + "\n")

    for k in range(vehicles):
        for p in range(customers-1):
            index = np.argmax(service_indicators[k][p])
            if index % (nodes + 1) != customers:
                f.write(str(index % (nodes + 1)) + " ")
            else:
                break
        f.write("\n")
    f.close()


def backtrack_search(problem: VRPTW_util.VRPTWInstance, perturbed_problem: VRPTW_util.VRPTWInstance):

    model, service_indicators, service_time, _, _ = model_initialization(problem)
    model.optimize()

    indicator_v = np.array([v.x for v in service_indicators.flatten()]).reshape(vehicles, nodes-1, nodes+1, nodes+1)
    original_opt_sol = model.objVal

    print_solution("opt_original_solution.txt", problem, indicator_v)

    # timewindow for perturbed problem
    new_time_window = [[], []]
    for i in range(nodes):
        new_time_window[0].append(perturbed_problem.nodes[i].a)
        new_time_window[1].append(perturbed_problem.nodes[i].b)

    new_time_window[0].append(min(new_time_window[0]))
    new_time_window[1].append(max(new_time_window[1]))

    # Reset model
    model.reset()
    # MPP solution
    difference_in_travels = []
    for k in range(vehicles):
        for p in range(nodes-1):
            route_change_indicator = model.addVar(vtype=GRB.CONTINUOUS, name="change_ind^%g_%g" % (k, p))
            for i in range(nodes):
                for j in range(nodes):
                    model.addConstr(service_indicators[k, p, i, j] - indicator_v[k, p, i, j] <= route_change_indicator,
                                    "route_check^%g_%g,%g,%g" % (k, p, i, j))
                    model.addConstr(indicator_v[k, p, i, j] - service_indicators[k, p, i, j] <= route_change_indicator,
                                    "route_check^%g_%g,%g,%g" % (k, p, i, j))
            difference_in_travels.append(route_change_indicator)

            # remove the original timewindow constraints
            model.remove(model.getConstrByName("lb_twC^%g_%g" % (k, p)))
            model.remove(model.getConstrByName("ub_twC^%g_%g" % (k, p)))
            model.update()

            # new timewindow constraints
            model.addConstr(
                gp.quicksum((service_indicators[k, p, :, :] * new_time_window[0]).flatten()) <= service_time[k, p],
                "lb_twC^%g_%g" % (k, p))
            model.addConstr(
                gp.quicksum((service_indicators[k, p, :, :] * new_time_window[1]).flatten()) >= service_time[k, p],
                "ub_twC^%g_%g" % (k, p))

    model.setObjective(gp.quicksum(difference_in_travels), GRB.MINIMIZE)
    model.optimize()

    indicator_v = np.array([v.x for v in service_indicators.flatten()]).reshape(vehicles, nodes - 1, nodes + 1,
                                                                                nodes + 1)
    service_time_v = np.array([v.x for v in service_time.flatten()]).reshape(vehicles, nodes - 1)
    mpp_opt_sol = model.objVal

    print_solution("perturbed_opt_solution.txt", perturbed_problem, indicator_v)

    return mpp_opt_sol, indicator_v, service_time_v


if __name__ == '__main__':
    file_name = "problem1.txt"
    perturbated_file_name = "perturbated_problem1.txt"
    problem = VRPTW_util.VRPTWInstance.load(file_name)
    perturbated_problem = VRPTW_util.VRPTWInstance.load(perturbated_file_name)
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    opt_sol, indicators, times = backtrack_search(problem, perturbated_problem)
    print(opt_sol)
    for k in range(vehicles):
        print("Vehicle %g:" % k)
        for p in range(nodes-1):
            index = np.argmax(indicators[k][p])
            print(index // (nodes+1), index % (nodes+1))
    print(times)
