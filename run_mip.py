import argparse
import json
import os
from os import listdir
from os.path import isfile, join
import re
import time

import random
import numpy as np
from docplex.mp.model import Context, Model


import VRPTW_util as util

K = 99999

from docplex.mp.progress import SolutionRecorder


def extract_solution(problem, mdl):
    # # Get values for decision variables used in the model
    num_vehicles = problem.num_vehicles
    num_nodes = len(problem.nodes)
    vehicle_to_node_time = [sorted([(mdl.get_value("st_{}_{}".format(i,j)), i) for i in range(num_nodes)])
                            for j in range(num_vehicles)]

    routes = []
    times = []

    for ts in vehicle_to_node_time:
        route = []
        time = []

        for t in ts:
            if t[0] > 0.0:
                time.append(t[0])
                route.append(t[1])

        routes.append(route)
        times.append(time)

    return {'name': problem.name, 'routes': routes, 'time': times}


class MyProgressListener(SolutionRecorder):
    def __init__(self, problem, solution_file, cost_file, st):
        SolutionRecorder.__init__(self)
        self.costs = []
        self.times = []
        self.current_objective = 999999
        self.problem = problem
        self.solution_file = solution_file
        self.cost_file = cost_file
        self.st = st

    def notify_start(self):
        super(MyProgressListener, self).notify_start()
        self.last_obj = None

    def is_improving(self, new_obj, eps=1e-8):
        last_obj = self.last_obj
        return last_obj is None or (abs(new_obj- last_obj) >= eps)

    def notify_solution(self, s):
        SolutionRecorder.notify_solution(self, s)
        if s.has_objective() and self.is_improving(s.get_objective_value()):
            self.last_obj = s.get_objective_value()
            print('----> #new objective={0}'.format(self.last_obj))
            self.costs.append(s.get_objective_value())
            self.times.append(time.time()-self.st)
            data = {"name": self.problem.name, "cost":self.costs, "time":self.times, "optimal": False}

            with open(self.cost_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            solution = extract_solution(self.problem, s)

            with open(self.solution_file, 'w') as f:
                json.dump(solution, f, ensure_ascii=False, indent=4)


def get_time_for_sln(prob, sln, time):

    num_vehicles = prob.num_vehicles
    num_nodes = len(prob.nodes)

    st = np.zeros([num_vehicles, num_nodes])

    for i, r in enumerate(sln):
        #print('vehicle {}'.format(i))
        text = '\t0: 0.0'
        last = -1
        last_time = 0.0

        for j, c in enumerate(r):
            t = time[i][j]
            text += ', {}: {:.2f}'.format(c, t)
            last = c
            last_time = t
            st[i][c] = t

        if last != -1:
            total_time = last_time + prob.distance(last, 0)
            text += ', 0: {:.2f}'.format(total_time)

    return st



def check_feasibility(pert_st, lb, ub, tol=1e-9):
    failed = []
    for v, st in enumerate(pert_st):
        for i, t in enumerate(st):
            if t and abs(t) > tol: # Skip if 0 or there exists some minor floating point error
                if  tol < lb[i]-t or t -ub[i] > tol:
                    failed.append(v, i, t, lb[i], ub[i])
                    print(i,t, lb[i], ub[i])

    return failed


def build_MIP(orig_prob, solution, pert_prob, use_stronger_constraint, threads):
    num_vehicles = orig_prob.num_vehicles
    num_nodes = len(orig_prob.nodes)
    orig_sln = solution['routes']
    orig_time = solution['time']

    orig_st = get_time_for_sln(orig_prob, orig_sln, orig_time)
    context = Context.make_default_context()
    context.cplex_parameters.threads = threads

    m = Model("VRPTW", context=context)

    num_vehicles = pert_prob.num_vehicles
    num_nodes = len(pert_prob.nodes)

    x = m.binary_var_cube(num_nodes, num_nodes, num_vehicles, name="x")

    lb = [pert_prob.nodes[i].a for i in range(num_nodes)] #+ [0]
    ub = [pert_prob.nodes[i].b for i in range(num_nodes)] #+ [K]


    s = m.continuous_var_matrix(num_nodes, num_vehicles, lb=0, name="st")

    source_id = 0
    sink_id = 0

    for k in range(num_vehicles):
        m.add_constraint(
            m.sum(x[(source_id,j,k)] for j in range(1, num_nodes)) == 1
        )

    for k in range(num_vehicles):
        m.add_constraint(
            m.sum(x[(i,sink_id,k)] for i in range(1, num_nodes)) == 1
        )

    for k in range(num_vehicles):
        for i in range(1, num_nodes):
            m.add_constraint(
                m.sum(x[(i,j,k)] for j in range(num_nodes)) == m.sum(x[(j,i,k)] for j in range(num_nodes))
            )

    for k in range(num_vehicles):
        for i in range(num_nodes):
            m.add_constraint(x[(i,i,k)] == 0)

    # From Jasper's model
    if use_stronger_constraint:
        for k in range(num_vehicles):
            for j in range(num_nodes):
                m.add_constraint(
                    m.sum(x[(i,j,k)] for i in range(num_nodes)) <= 1 )

        for k in range(num_vehicles):
            for i in range(num_nodes):
                m.add_constraint(
                    m.sum(x[(i,j,k)] for j in range(num_nodes)) <= 1 )

        for j in range(num_nodes):
            m.add_constraint(
                m.sum(x[(i,j,k)] for i in range(num_nodes) for k in range(num_vehicles)) >= 1 )

    # From Louis's model
    else:
        for j in range(1, num_nodes):
            m.add_constraint(
                m.sum(x[(i,j,k)] for i in range(num_nodes) for k in range(num_vehicles)) == 1
            )

        for i in range(1, num_nodes):
            m.add_constraint(
                m.sum(x[(i,j,k)] for j in range(num_nodes) for k in range(num_vehicles)) == 1
            )

    for k in range(num_vehicles):
        for i in range(0, num_nodes):
            for j in range(1, num_nodes):

                if i != j:
                    t_ij = abs(pert_prob.nodes[i].distance(pert_prob.nodes[j]))
                    m.add_constraint(
                        s[(i,k)] - s[(j,k)] + t_ij <= (1-x[(i,j,k)]) * K
                    )

    for k in range(num_vehicles):
        for i in range(num_nodes):
            m.add_constraint(s[(i,k)] >= lb[i] * m.sum(x[i,j,k] for j in range(num_nodes)))
            m.add_constraint(s[(i,k)] <= ub[i] * m.sum(x[i,j,k] for j in range(num_nodes)))


    m.minimize(m.sum(m.abs(s[(i,k)] - orig_st[k][i]) for i in range(num_nodes) for k in range(num_vehicles)))


    return m, lb, ub


def traverse_path(mat, li, i = 0, left=False):
    """
        Traverses (recursively) through path defined in mat

    Parameters
    ----------
    mat: list of list
        Path matrix where (i,j) indices imply the edge i -> j
    li: list
        list for holding historical nodes traveled
    i: int
        index of the current node
    Returns
    -------
    li: list
        list of all historical nodes traveled
    """

    i = np.where(mat[i])[0][0]
    if left and i == 0:
        return li
    li.append(i)
    return traverse_path(mat, li, i, left=True)


def solve_one_problem(original_problem, original_solution, perturbated_problem, output, cost_file, threads, use_stronger_constraint, st):
    orig_prob = util.VRPTWInstance.load(original_problem)
    solution = util.load_routes(original_solution)
    pert_prob = util.VRPTWInstance.load(perturbated_problem)
    mdl, _, _ = build_MIP(orig_prob, solution, pert_prob, use_stronger_constraint, threads)

    listener = MyProgressListener(pert_prob, output, cost_file, st)
    mdl.add_progress_listener(listener)
    msol = mdl.solve(log_output=True)

    print(mdl.objective_value)

    listener.costs.append(mdl.objective_value)
    listener.times.append(time.time()-listener.st)
    result = {'cost': listener.costs, 'time': listener.times, 'optimal': True}
    solution = extract_solution(pert_prob, mdl.solution)

    with open(output, 'w') as f:
        json.dump(solution, f, ensure_ascii=False, indent=4)

    with open(cost_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    st = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-problem', type=str, required=True)
    parser.add_argument('--original-solution', type=str, required=True)
    parser.add_argument('--perturbated-problem', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--stronger-constraint', type=int, default=1)
    args = parser.parse_args()

    solve_one_problem(args.original_problem, args.original_solution, args.perturbated_problem, args.output, args.cost, args.threads, args.stronger_constraint, st)
