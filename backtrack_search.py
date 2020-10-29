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
    def __init__(self, constraints: List[gp.Constr], parent, children):
        self.constr = constraints
        self.parent = parent
        self.children = children
        self.optimal = 0


class VRPTWTree:
    def __init__(self, problem: VRPTW_util.VRPTWInstance):
        self.problem = problem
        self.root = VRPTWNode([], None, [])


def backtrack_search(problem: VRPTW_util.VRPTWInstance, perturbated_problem: VRPTW_util.VRPTWInstance):
    vehicles, nodes = problem.num_vehicles, len(problem.nodes)

    # MIP model setup
    model = gp.Model("MainBS")

    # Variables
    service_indicators = []
    service_time = []
    for k in range(vehicles):

        level1 = []
        vehicle_service_time = []

        for p in range(nodes-1):

            level2 = []
            for i in range(nodes):

                level3 = []
                for j in range(nodes):
                    level3.append(model.addVar(vtype=GRB.BINARY, name="I^%g_%g,%g,%g" % (k, i, j, p)))

                level2.append(level3)

            level1.append(level2)
            vehicle_service_time.append(model.addVar(vtype=GRB.CONTINUOUS, name="s^%g_%g" % (k, p)))

        service_indicators.append(level1)
        service_time.append(vehicle_service_time)

    service_indicators = np.array(service_indicators)   # shape of (k, p, i, j)
    service_time = np.array(service_time)   # shape of (k, p)

    distance_map = [[] for _ in range(nodes)]
    time_window = [[], []]
    for i in range(nodes):
        time_window[0].append(problem.nodes[i].a)
        time_window[1].append(problem.nodes[i].b)

        for j in range(nodes):
            distance_map[i].append(problem.distance(i, j))
    distance_map = np.array(distance_map)  # shape of (i, j)
    time_window = np.array(time_window)  # shape of (2, j)

    # Constraints

    # Vehicles have to start from node 0
    model.addConstr(gp.quicksum(service_indicators[:, 0, list(range(1, nodes)), :].flatten()) == 0, "Start_positionC")
    for k in range(vehicles):
        model.addConstr(service_time[k, 0] == 0, "starting_time_%g" % k)

        for p in range(nodes-1):
            # Constraints that make sure each vehicle travels on one arc at each time
            model.addConstr(gp.quicksum(service_indicators[k, p, :, :].flatten()) <= 1, "binaryC^%g_%g" % (k, p))

            # Timewindow Constraints
            model.addConstr(gp.quicksum((service_indicators[k, p, :, :] * time_window[0]).flatten()) <= service_time[k, p],
                            "lb_twC^%g_%g" % (k, p))
            model.addConstr(gp.quicksum((service_indicators[k, p, :, :] * time_window[1]).flatten()) >= service_time[k, p],
                            "ub_twC^%g_%g" % (k, p))

            if p == 0:
                # Travel time constraints
                model.addConstr(gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                                <= service_time[k, p], "timeC^%g_%g" % (k, p))
            else:
                # Travel time constraints
                model.addConstr(service_time[k, p-1] + gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                                <= service_time[k, p], "timeC^%g_%g" % (k, p))

                # Each vehicle must travel continuously
                model.addConstr(gp.quicksum(service_indicators[k, p, :, :].flatten())
                                <= gp.quicksum(service_indicators[k, p-1, :, :].flatten()), "continuous_route^%g_%g" % (k, p))

                for i in range(nodes):
                    for j in range(nodes):
                        indices = list(range(nodes))
                        indices.remove(j)

                        for var in service_indicators[k, p, indices, :].flatten():
                            # Constraints that make sure vehicles start travel from where they at in next time period.
                            model.addConstr(service_indicators[k, p - 1, i, j] + var <= 1, "connection^%g_%g" % (k, p))
    # Service completion constraints, node 0 is depot
    for j in range(1, nodes):
        model.addConstr(gp.quicksum(service_indicators[:, :, :, j].flatten()) >= 1, "completion_%g" % j)

    model.setObjective(gp.quicksum(service_time[:, -1]), GRB.MINIMIZE)
    model.optimize()

    indicator_v = np.array([v.x for v in service_indicators.flatten()]).reshape(vehicles, nodes-1, nodes, nodes)
    service_time_v = np.array([v.x for v in service_time.flatten()]).reshape(vehicles, nodes-1)

    print(time_window)
    print(distance_map)
    return model.objVal, indicator_v, service_time_v


if __name__ == '__main__':
    file_name = "problem1.txt"
    problem = VRPTW_util.VRPTWInstance.load(file_name)

    opt_sol, indicators, times = backtrack_search(problem, problem)
    print(opt_sol)
    print(indicators)
    print(times)
