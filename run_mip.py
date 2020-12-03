import argparse
import json
import os
from os import listdir
from os.path import isfile, join
import re
import time

import random
import numpy as np
from docplex.mp.model import Model


import VRPTW_util as util

SAVEFOLDERPATH = "./mip_solutions/"
BENCHPATH = "./benchmarks/"

K = 99999

from docplex.mp.progress import SolutionRecorder

class MyProgressListener(SolutionRecorder):
    def __init__(self, model):
        SolutionRecorder.__init__(self)
        self.costs = []
        self.times = []
        self.current_objective = 999999

    def notify_start(self):
        super(SolutionRecorder, self).notify_start()
        self.last_obj = None
        self.st = time.time()

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

    def write_to_file(self, filename):
        data = {"cost":self.costs, "time":self.times}
        # Create folder to store all outputs
        if not os.path.exists(SAVEFOLDERPATH):
            os.makedirs(SAVEFOLDERPATH)
        with open(SAVEFOLDERPATH+filename+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def get_time_for_sln(prob, sln):

    num_vehicles = prob.num_vehicles
    num_nodes = len(prob.nodes)

    time = prob.get_time(sln)
    st = np.zeros([num_vehicles, num_nodes])

    if time is None:
        print('invalid solution')
    else:
        for i, r in enumerate(sln):
            #print('vehicle {}'.format(i))
            text = '\t0: 0.0'
            last = -1
            last_time = 0.0

            for c in r:
                t = time[c][1]
                text += ', {}: {:.2f}'.format(c, t)
                last = c
                last_time = t
                st[i][c] = t

            if last != -1:
                total_time = last_time + prob.distance(last, 0)
                text += ', 0: {:.2f}'.format(total_time)
    print(time)

            #print(text)
        #print('total cost: {:.2f}'.format(time[0]))

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


def build_MIP(original_problem, original_solution, perturbated_problem, param):
    prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size = param
    orig_prob = util.VRPTWInstance.load(original_problem)
    orig_sln = util.load_solution(original_solution)
    pert_prob = util.VRPTWInstance.load(perturbated_problem)

    orig_st = get_time_for_sln(orig_prob, orig_sln)

    m = Model("VRPTW")

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

           # m.add_constraint(x[(sink_id,i,k)] == 0)
            #m.add_constraint(x[(i, source_id, k)] == 0)


    for j in range(1, num_nodes):
        m.add_constraint(
            m.sum(x[(i,j,k)] for i in range(num_nodes) for k in range(num_vehicles)) == 1
        )

    for i in range(1, num_nodes):
        m.add_constraint(
            m.sum(x[(i,j,k)] for j in range(num_nodes) for k in range(num_vehicles)) == 1
        )

    for k in range(num_vehicles):
        for i in range(1, num_nodes):
            for j in range(1, num_nodes):

                if i != j:
                    t_ij = abs(pert_prob.nodes[i].distance(pert_prob.nodes[j]))
                    m.add_constraint(
                        s[(i,k)] - s[(j,k)] + t_ij <= (1-x[(i,j,k)]) * K
                    )

    for k in range(num_vehicles):
        for i in range(num_nodes):
            #m.add_constraint(s[(i,k)] <= b[i] * m.sum(x[i,j,k] for j in range(num_nodes)) )
            m.add_constraint(s[(i,k)] >= lb[i] * m.sum(x[i,j,k] for j in range(num_nodes)) )

            #if i < sink_id:
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


def run_MIP(params):
    """
        Runs the MIP model based on the input parameter tuples.

    Parameters
    ----------
    params: list of tuples
        The list of parameter tuples that the MIP model will use to optimize. The tuples are in order of (prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size)

    Returns
    -------
    results: dict
        Dictionary containing training information such as runtime and objective value (L1 Norm of starting times)
    """

    results = {"prob_num":[], "instance_num":[], "num_vehicles":[], "num_nodes":[], "time_window":[], "map_size":[], "failed":[], "obj_value":[], "total_time":[], "optimization_time":[], "params":[]}

    for param in params:
        prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size = param
        prob_name = "{}{}" + "_v{}_c{}_tw{}_xy{}_".format(num_vehicles, num_nodes, time_window, map_size) + "{}"

        st = time.time()
        m, lb, ub = build_MIP(param)
        st2 = time.time()
        msol = m.solve(log_output=True)
        et = time.time()

        # Get values for decision variables used in the model
        pert_x = [[[m.get_var_by_name("x_{}_{}_{}".format(i,j,k)).solution_value for j in range(num_nodes)] for i in range(num_nodes)] for k in range(num_vehicles)]
        pert_st = [[m.get_var_by_name("st_{}_{}".format(i,k)).solution_value for i in range(num_nodes)] for k in range(num_vehicles)]

        # Generate a list of variables that failed feasiblity test
        failed = check_feasibility(pert_st, lb=lb, ub=ub)

        # Record metrics and parameters
        results["prob_num"].append(prob_num)
        results["instance_num"].append(instance_num)
        results["num_vehicles"].append(num_vehicles)
        results["num_nodes"].append(num_nodes)
        results["time_window"].append(time_window)
        results["map_size"].append(map_size)
        results["params"].append(param)
        results["failed"].append(failed)
        results["obj_value"].append(m.objective_value)
        results["total_time"].append(et-st)
        results["optimization_time"].append(et-st2)

        # Create folder to store all outputs
        if not os.path.exists(SAVEFOLDERPATH):
            os.makedirs(SAVEFOLDERPATH)

        save_path = SAVEFOLDERPATH + prob_name.format("solution_mip", prob_num, instance_num) + ".txt"

        # Dump the results object as JSON
        with open(SAVEFOLDERPATH + '/mip_result.json', 'w+') as fp:
            json.dump(results, fp)

        # Save solutions as per format defined in https://gist.github.com/Kurorororo/21ccc9ecbea2191a52f62e4bed2224db
        with open(save_path, "w+") as f:
            f.writelines(prob_name.format("solution_mip", prob_num, instance_num) + "\n")
            f.writelines(str(num_vehicles) + "\n" + str(num_nodes) +"\n")
            for i, x in enumerate(pert_x):
                paths = list(map(str, traverse_path(x, [])))
                line = " ".join(paths)
                f.writelines(str(line) + "\n")

        print(pert_st)

    return results


def solve_one_problem(original_problem, original_solution, perturbated_problem, output, cost):

    pattern = re.compile(r'perturbated(?P<p>\d+)_v(?P<v>\d+)_c(?P<c>\d+)_tw(?P<tw>\d+)_xy(?P<xy>\d+)_(?P<id>\d+)\.txt')
    m = pattern.match(os.path.basename(perturbated_problem))
    prob_num = int(m.group('p'))
    instance_num = int(m.group('id'))
    num_vehicles = int(m.group('v'))
    num_nodes = int(m.group('c'))
    time_window = int(m.group('tw'))
    map_size = int(m.group('xy'))
    params = prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size


    mdl, _, _ = build_MIP(original_problem, original_solution, perturbated_problem, params)

    listener = MyProgressListener(mdl)
    mdl.add_progress_listener(listener)
    mdl.solve(log_output=True)

    listener.costs.append(mdl.objective_value)
    listener.times.append(time.time()-listener.st)


    # # Get values for decision variables used in the model
    pert_x = [[[mdl.get_var_by_name("x_{}_{}_{}".format(i,j,k)).solution_value for j in range(num_nodes)] for i in range(num_nodes)] for k in range(num_vehicles)]


    prob_name = "{}{}" + "_v{}_c{}_tw{}_xy{}_".format(num_vehicles, num_nodes, time_window, map_size) + "{}"

    listener.write_to_file(prob_name.format("mip_results", prob_num, instance_num))

    with open(output, "w+") as f:
        f.writelines(prob_name.format("solution_mip", prob_num, instance_num) + "\n")
        f.writelines(str(num_vehicles) + "\n")
        for i, x in enumerate(pert_x):
            paths = list(map(str, traverse_path(x, [])))
            line = " ".join(paths)
            f.writelines(str(line) + "\n")

    with open(cost, "w") as f:
        f.write(str(mdl.objective_value) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-problem', type=str, required=True)
    parser.add_argument('--original-solution', type=str, required=True)
    parser.add_argument('--perturbated-problem', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cost', type=str, required=True)
    args = parser.parse_args()

    solve_one_problem(args.original_problem, args.original_solution, args.perturbated_problem, args.output, args.cost)

    # prob_nums = [8]
    # veh_nums = [4]
    # inst_nums = [0,1,2]
    # for num_vehicles in veh_nums:
    #     for prob_num in prob_nums:
    #         for instance_num in inst_nums:
    #             num_nodes, time_window, map_size = 16, 4, 16
    #
    #             original_problem = "C:/Users/Louis/Documents/1stYearMasters/Fall Semester/MIE562/VRPTW_scheduler/benchmarks/original_v{}_c{}_tw{}_xy{}_{}.txt".format(num_vehicles, num_nodes, time_window, map_size, instance_num)
    #             original_solution = "C:/Users/Louis/Documents/1stYearMasters/Fall Semester/MIE562/VRPTW_scheduler/benchmarks/solution_v{}_c{}_tw{}_xy{}_{}.txt".format(num_vehicles, num_nodes, time_window, map_size, instance_num)
    #             perturbated_problem = "C:/Users/Louis/Documents/1stYearMasters/Fall Semester/MIE562/VRPTW_scheduler/benchmarks/perturbated{}_v{}_c{}_tw{}_xy{}_{}.txt".format(prob_num, num_vehicles, num_nodes, time_window, map_size, instance_num)
    #
    #             param = prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size
    #
    #             solve_one_problem(original_problem, original_solution, perturbated_problem, "output", "cost")
    #
    #             break
    #         break
    #     break
