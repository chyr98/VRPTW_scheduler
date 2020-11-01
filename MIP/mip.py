import os
os.chdir("C:/Users/Louis/Documents/1stYearMasters/Fall Semester/MIE562/VRPTW_scheduler/")


import random
import numpy as np
from docplex.mp.model import Model


import VRPTW_util as util

#original_v8_c64_tw4_xy16_3
prob = "./benchmarks/{}{}_v8_c16_tw8_xy16_{}.txt"
K = 999

i = 2
p = 4
orig_prob = util.VRPTWInstance.load(prob.format("original", "", p))
orig_sln = util.load_solution(prob.format("solution", "", p))
orig_time = orig_prob.get_time(orig_sln)

pert_prob = util.VRPTWInstance.load(prob.format("perturbated", i, p))



def get_time_for_sln(prob, sln):

    num_vehicles = prob.num_vehicles
    num_nodes = len(prob.nodes)

    time = prob.get_time(sln)
    st = np.zeros([num_vehicles, num_nodes])

    if time is None:
        print('invalid solution')
    else:
        for i, r in enumerate(sln):
            print('vehicle {}'.format(i))
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

            print(text)
        print('total cost: {:.2f}'.format(orig_time[0]))

    return st

orig_st = get_time_for_sln(orig_prob, orig_sln)




m = Model("VRPTW")


num_vehicles = pert_prob.num_vehicles
num_nodes = len(pert_prob.nodes) + 1



x = m.binary_var_cube(num_nodes, num_nodes, num_vehicles, name="x")

a = [pert_prob.nodes[i].a for i in range(num_nodes-1)] + [0]
b = [pert_prob.nodes[i].b for i in range(num_nodes-1)] + [K]


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
        m.add_constraint(s[(i,k)] >= a[i] * m.sum(x[i,j,k] for j in range(num_nodes)) )

        if i < sink_id:
            m.add_constraint(s[(i,k)] <= b[i] * m.sum(x[i,j,k] for j in range(num_nodes)))


m.minimize(m.sum(m.abs(s[(i,k)] - orig_st[k][i]) for i in range(num_nodes-1) for j in range(num_vehicles)))

msol = m.solve(log_output=True)

print(m.solve_details)

pert_x = [[[m.get_var_by_name("x_{}_{}_{}".format(i,j,k)).solution_value for j in range(num_nodes)] for i in range(num_nodes)] for k in range(num_vehicles)]
pert_st = [[m.get_var_by_name("st_{}_{}".format(i,k)).solution_value for i in range(num_nodes)] for k in range(num_vehicles)]


print(pert_x)
print(pert_st)
print(orig_st)