import json
import os
from os import listdir
from os.path import isfile, join
import time


os.chdir("C:/Users/Louis/Documents/1stYearMasters/Fall Semester/MIE562/VRPTW_scheduler/")


import random
import numpy as np
from docplex.mp.model import Model


import VRPTW_util as util

SAVEFOLDERPATH = "./mip_solutions/"
BENCHPATH = "./benchmarks/"

K = 99999

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
                t = time[1][c][1]
                text += ', {}: {:.2f}'.format(c, t)
                last = c
                last_time = t
                st[i][c] = t

            if last != -1:
                total_time = last_time + prob.distance(last, 0)
                text += ', 0: {:.2f}'.format(total_time)

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


def run_MIP(param):

    prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size = param

    prob_name = "{}{}" + "_v{}_c{}_tw{}_xy{}_".format(num_vehicles, num_nodes, time_window, map_size) + "{}"

    prob_path = BENCHPATH + prob_name + '.txt'

    orig_prob = util.VRPTWInstance.load(prob_path.format("original", "", instance_num))
    orig_sln = util.load_solution(prob_path.format("solution", "", instance_num))
    orig_time = orig_prob.get_time(orig_sln)

    pert_prob = util.VRPTWInstance.load(prob_path.format("perturbated", prob_num, instance_num))



    orig_st = get_time_for_sln(orig_prob, orig_sln)

    m = Model("VRPTW")

    num_vehicles = pert_prob.num_vehicles
    num_nodes = len(pert_prob.nodes) + 1

    x = m.binary_var_cube(num_nodes, num_nodes, num_vehicles, name="x")

    lb = [pert_prob.nodes[i].a for i in range(num_nodes-1)] + [0]
    ub = [pert_prob.nodes[i].b for i in range(num_nodes-1)] + [K]


    s = m.continuous_var_matrix(num_nodes, num_vehicles, lb=0, name="st")


    source_id = 0
    sink_id = num_nodes-1

    for k in range(num_vehicles):
        m.add_constraint(
            m.sum(x[(source_id,j,k)] for j in range(num_nodes)) == 1
        )

    for k in range(num_vehicles):
        m.add_constraint(
            m.sum(x[(i,sink_id,k)] for i in range(num_nodes)) == 1
        )

    for k in range(num_vehicles):
        for i in range(1, num_nodes-1):
            m.add_constraint(
                m.sum(x[(i,j,k)] for j in range(num_nodes)) == m.sum(x[(j,i,k)] for j in range(num_nodes))
            )

    for k in range(num_vehicles):
        for i in range(num_nodes):
            m.add_constraint(x[(i,i,k)] == 0)

            m.add_constraint(x[(sink_id,i,k)] == 0)
            m.add_constraint(x[(i, source_id, k)] == 0)


    for i in range(1, num_nodes-1):
        m.add_constraint(
            m.sum(x[(i,j,k)] for j in range(num_nodes) for k in range(num_vehicles)) == 1
        )


    for k in range(num_vehicles):
        for i in range(num_nodes):
            for j in range(num_nodes):
                i_, j_ = i % (num_nodes - 1), j % (num_nodes - 1)
                t_ij = abs(pert_prob.nodes[i_].distance(pert_prob.nodes[j_]))
                m.add_constraint(
                    s[(i,k)] - s[(j,k)] + t_ij <= (1-x[(i,j,k)]) * K
                )

    for k in range(num_vehicles):
        for i in range(num_nodes):
            #m.add_constraint(s[(i,k)] <= b[i] * m.sum(x[i,j,k] for j in range(num_nodes)) )
            m.add_constraint(s[(i,k)] >= lb[i] * m.sum(x[i,j,k] for j in range(num_nodes)) )

            if i < sink_id:
                m.add_constraint(s[(i,k)] <= ub[i] * m.sum(x[i,j,k] for j in range(num_nodes)))


    m.minimize(m.sum(m.abs(s[(i,k)] - orig_st[k][i]) for i in range(num_nodes-1) for j in range(num_vehicles)))


    return m, lb, ub



def traverse_path(mat, li, i = 0):
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

    if not len(np.where(mat[i])[0]):
        return li
    i = np.where(mat[i])[0][0]
    li.append(i)
    return traverse_path(mat, li, i)


def main(params):
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
        m, lb, ub = run_MIP(param)
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

    return results

if __name__ == "__main__":
    # Should match file combinations in benchmark folder
    params = {"num_vehicles":[1,2,4,], "num_nodes":[4,16,64,], "time_window":[4,8,16], "map_size":[16], "instances":[0,1,2,3,4], "problems":[1,2,4,8]}
    # prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size
    params = [(prob, ins, v, n, tw, xy) for v in params['num_vehicles'] for n in params['num_nodes'] for tw in params['time_window'] for xy in params['map_size'] for ins in params['instances'] for prob in params['problems'] if v < n and prob < n]

    results = main(params)






## Items may fail due to numerical issues in CPLEX
# params = []
# for i, f in enumerate(results['failed']):
#     if len(f):
#         param = (results['prob_num'][i], results['instance_num'][i], results['num_vehicles'][i], results["num_nodes"][i], results["time_window"][i], results["map_size"][i])
#         params.append(param)
#
# for param in params:
#     m, lb, ub = run_MIP(param)
#     m.solve()
#     prob_num, instance_num, num_vehicles, num_nodes, time_window, map_size = param
#
#     pert_x = [[[m.get_var_by_name("x_{}_{}_{}".format(i,j,k)).solution_value for j in range(num_nodes)] for i in range(num_nodes)] for k in range(num_vehicles)]
#
#     pert_st = [[m.get_var_by_name("st_{}_{}".format(i,k)).solution_value for i in range(num_nodes)] for k in range(num_vehicles)]
#
#     print(check_feasibility(pert_st, lb, ub, tol=1e-4))