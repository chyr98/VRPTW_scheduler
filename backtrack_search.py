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
    def __init__(self, constraints: List[gp.Constr], parent: VRPTWNode, children: List[VRPTWNode]):
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

    model = gp.Model("MainBS")
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
    for i in range(nodes):
        for j in range(nodes):
            distance_map[i].append(problem.distance(i, j))
    distance_map = np.array(distance_map)

    for k in range(vehicles):
        model.addConstr(service_time[k, 0] == 0, "starting_time_%g" % k)

        for p in range(nodes-1):
            # Constraints that make sure each vehicle travels on one arc at each time
            model.addConstr(gp.quicksum(service_indicators[k, p, :, :].flatten()) <= 1, "binaryC^%g_%g" % (k, p))

            if p == 0:
                model.addConstr(gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                                <= service_time[k, p], "timeC^%g_%g" % (k, p))
            else:
                model.addConstr(service_time[k, p-1] + gp.quicksum((service_indicators[k, p, :, :] * distance_map).flatten())
                                <= service_time[k, p], "timeC^%g_%g" % (k, p))
            # Todo: Timewindow constraints
            for i in range(nodes):
                for j in range(nodes):
                    indices = list(range(nodes))
                    indices.remove(j)
                    for var in service_indicators[k, p+1, indices, :]:
                        # Constraints that make sure vehicles start travel from where they at in next time period.
                        model.addConstr(service_indicators[k, p, i, j] + var <= 1, "connection^%g_%g" % (k, p))




