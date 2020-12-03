import copy as cp
import argparse
import json
import os
import random
import math
import time
from pathlib import Path

import VRPTW_util as util


def readPath(filepath,nCus):
    #reads a solution file and returns a list of sublists for each vehicle route

    with open(filepath) as f:
        solution = json.load(f)

    nVeh = len(solution['routes'])
    times = [[0 for _ in range(nCus)] for _ in range(nVeh)]

    for v, r in enumerate(solution['routes']):
        for i, cus in enumerate(r):
            times[v][cus-1] = solution['time'][v][i]

    return solution['routes'], times

def makeVisitData(path, cData):
    #if the path is feasible, return a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]
    #if the path is infeasible return None

    nVeh = len(path)
    
    all_visits = []

    for i in range(nVeh):
        #for each vehicle, add the depot as the starting node. No need to worry about returning to the depot - this has no impact on the MPP objective
        path[i] = [0] + path[i]
        t = 0

        for j in range((len(path[i])-1)):
            node_curr = path[i][j]
            node_next = path[i][j+1]


            #distances
            coords_curr = (cData[node_curr][0], cData[node_curr][1])
            coords_next = (cData[node_next][0], cData[node_next][1])

            d = distBtwnPts(coords_curr, coords_next)

            #get service time
            t = t + d

            #service time is delayed if need to wait for time windows
            tw_start = cData[node_next][2]
            tw_end = cData[node_next][3]

            if tw_start > t:
                t = tw_start
            elif tw_end < t:
                return None
                #Infeasible
                

            info_next = [node_next, i, t]
            all_visits.append(info_next)
            

    return all_visits


def makeVisitDataInfeasible(path,cData):
        #The infeasible version of the makeVisitData function
        
    nVeh = len(path)
    
    all_visits = []

    for i in range(nVeh):
        #for each vehicle, add the depot as the starting node. No need to worry about returning to the depot - this has no impact on the MPP objective
        path[i] = [0] + path[i]
        t = 0

        for j in range((len(path[i])-1)):
            node_curr = path[i][j]
            node_next = path[i][j+1]


            #distances
            coords_curr = (cData[node_curr][0], cData[node_curr][1])
            coords_next = (cData[node_next][0], cData[node_next][1])

            d = distBtwnPts(coords_curr, coords_next)

            #get service time
            t = t + d

            #service time is delayed if need to wait for time windows
            tw_start = cData[node_next][2]
            tw_end = cData[node_next][3]

            if tw_start > t:
                t = tw_start

            info_next = [node_next, i, t]
            all_visits.append(info_next)
            

    return all_visits



def distBtwnPts(coords1,coords2):
    
    a = ((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2)
    return (math.sqrt(a))
            
def removeVisits(path, all_visits, cData, frac):
    #find visits to remove
    
    #number of visits to remove
    nRem = math.ceil(frac*len(all_visits))
   
    rem_vs = []
    red_path = []


    #randomly choose first removal
    rand_index = random.randrange(len(all_visits))
    rem_vs.append(all_visits[rand_index])
    del all_visits[rand_index]

    while len(rem_vs) < nRem:


        #chose a random visit from the already removed set
        rand_index = random.randrange(len(rem_vs))
        v_compare = rem_vs[rand_index]
        

        removeVisit(v_compare, all_visits, cData, rem_vs)


        
    #delete all removed visits from the path
    for i in range(len(rem_vs)):
        #customer
        
        j = rem_vs[i][0]
        
        #vehicle
        veh = rem_vs[i][1]
        path[veh].remove(j)

        #don't iterate over path while modifiying

    #delete the relatendess from the rem_vs
    for i in range(len(rem_vs)):
        if len(rem_vs[i]) == 4:
            del rem_vs[i][3]


    return rem_vs, path


def removeVisit(v_comp, all_visits, cData,rem_vs):
    #ranks all_visits based on similarity to v_comp, selects one visit to remove, removes from all_visits, appends to rem_vs

    #hyperparameter d, used from paper, controls determinism
    d = 5

    #compute relatedness measure
    for i in range(len(all_visits)):
        r = relatedness(v_comp, all_visits[i], cData)


        #if first time running function
        if len(all_visits[i]) == 3:
            all_visits[i].append(r)
        #if not first time
        else:
            all_visits[i][3] = r

    #sort list by relatedness
    all_visits.sort(key=lambda x: x[3], reverse = True)


    #choose visit: formula is floor(|remaining_visits| * rand^D) as per paper
    rand_num = random.random()
    ind = math.floor(len(all_visits)*(rand_num**d))


    rem_vs.append(all_visits[ind])
    del all_visits[ind]



def relatedness(v1,v2,cData):
    #find the relatedness of two visits

    coords1 = (cData[v1[0]][0] , cData[v1[0]][1])
    coords2 = (cData[v2[0]][0] , cData[v2[0]][1])

    d = distBtwnPts(coords1,coords2)
    
    #T is 1 if vehicle is the same
    T = 0
    if v1[1] == v2[1]:
        T = 1
    
    #difference in service time
    s_diff = abs(v1[2] - v2[2])


    #hyperparameter values
    a = 0.75
    b = 0.1
    g = 1


    r = 1/(a*d + b*s_diff + g*T)
    

    return r

def GetSTimes(nCus,nVeh,v):
    #generate the service times for each vehicle
    #returns [[s_0,1, s_0,2, ..., s_0,(n-1)],...] where s_ij is the time that vehicle i serves customer j 

    res = [[0 for i in range(nCus-1)] for j in range(nVeh)]
    
    for i in range(len(v)):
        cust = v[i][0]
        veh = v[i][1]
        s = v[i][2]

        res[veh][cust-1] = s

    return res

def Objective(s_MPP, s_orig):
    #calculate the MPP objective between the original and the service times
    obj = 0

    if len(s_MPP) != len(s_orig):
        print("warning: service time lists for original and new solutions have different lengths")

    for i in range(len(s_MPP)):
        for j in range(len(s_MPP[i])):

            obj += abs(s_MPP[i][j] - s_orig[i][j])
            

    return obj


def ChooseVisit(r_path, rem_vs, orig_s_times,cData,d):
    #given a set of removed visits and a reduced path, select the variable (i.e return customer ID) to branch on and return the top d points (only these will be searched), 
    #return (customer , index in rem_vs, [(insertion index, vehicle, cost), ... ]) 

    #want the variable with the maximum minimum cost
    max_min_cost_tot = 0
    v = None
    P = None
    v_ind = None

    #A second max min cost in case of infeasibility
    inf_max_tot = 0

    #Big M constant:
    M = (len(cData)**2)*16

    global feasible_found


    for i in range(len(rem_vs)):
        f_pts = []


        for j in range(len(r_path)):
            #print('---------------------------considering vehicle ' , j)
            for k in range(len(r_path[j]) + 1):
                #print('considering insertion index ' , k)
                
                visit = rem_vs[i][0]
                r_path[j].insert(k,visit)

                cost = PathQuality(r_path,orig_s_times,cData)
                

                if cost != None:
                    f_pts.append((k, j , cost))
              
                else:
                    #infeasible

                    if not feasible_found:
                        infeas_cost = PathQualityInfeasible(r_path, orig_s_times, cData)
                        f_pts.append((k,j,infeas_cost))

                #else:
                #    print('infeasible, (point, veh, cost):' , (k, j , cost))

                del r_path[j][k]


    
        if f_pts:
            f_pts.sort(key = lambda x:x[2])


            max_min_cost = f_pts[0][2] 
            if max_min_cost > max_min_cost_tot and max_min_cost < M:

                v = rem_vs[i][0]
                v_ind = i
                P = f_pts[0:d]
                max_min_cost_tot = max_min_cost

            if not feasible_found:

                if max_min_cost > inf_max_tot:
                    v_inf = rem_vs[i][0]
                    v_ind_inf = i
                    P_inf = f_pts[0:d]


                    inf_max_tot = max_min_cost

    #if no feasible solutions found yet, chose an arbitrary variable:
    
    #If any routes giving feasibility exist - return those. Otherwise, branch on an infeasible route.

    if feasible_found or v != None:
        return (v, v_ind, P)
    else:
        return(v_inf, v_ind_inf, P_inf)

    
    
    

def PathQuality(path,orig_s_times,cData):
    #if a path is feasible, return the objective value for the MPP
    #if infeasible, return None
    
    nVeh = len(orig_s_times)
    nCus = len(orig_s_times[0])+1

    visits = makeVisitData(cp.deepcopy(path),cData)

    if visits == None:
        return None
    else:
        s_times = GetSTimes(nCus,nVeh,visits)
        obj = Objective(s_times, orig_s_times)

        return obj


def PathQualityInfeasible(path,orig_s_times,cData):
    
    

    nVeh = len(orig_s_times)
    nCus = len(orig_s_times[0])+1

    visits = makeVisitDataInfeasible(cp.deepcopy(path),cData)

    if visits == None:
        return None
    #print('visits ', visits)

    s_times = GetSTimes(nCus,nVeh,visits)
    obj = InfeasObj(s_times, cData)

    return obj


def InfeasObj(s_times, c_Data):
   
    nVeh = len(s_times)
    
    #nCus includes depot
    nCus = len(c_Data)
    
    #build time window data array
    tw_data = []

    for i in range(nCus-1):
        tw_data.append(c_Data[i+1][3])

    #print('tw data is ' , tw_data)


    #find non-zero service time
    nz_s_times = [0 for i in range(nCus-1)]

    for i in range(nVeh):
        for j in range(nCus-1):
            if s_times[i][j] != 0:
                nz_s_times[j] = s_times[i][j]
                #break

    #print('non-zero start times are', nz_s_times)


    #big M coef:
    M = nCus*nCus*16
    obj = M

    #sum tw_end violations:

    for i in range(nCus - 1):
        if nz_s_times[i] > tw_data[i]:
            obj += nz_s_times[i] - tw_data[i]

    return obj






def countCustomers(path):
    cnt = 0 
    for i in range(len(path)):
        for j in range(len(path[i])):
            cnt += 1

    return cnt


def Reinsert(red_path, rem_vs, d, optimal, og, orig_s_times, cData, t_lim, t_lim_iter, time_list, cost_list):
    #takes a reduced path, visits to reinsert (rem_vs), and a lower bound on the best solution,
    #Explores the search tree of reinserting all of  the visits, following a limited discrepancy search - starting with an initial value for d
    #the ordering heuristic is to choose the worst variable (i.e. the one with the maximum insertion cost to the objective) then select its best value
    #if a better solution than the current best_path is found -> best_path is updated
    
   global best_obj
   global best_path
   global t_start
   global t_iter_start
   global optim_status
   global feasible_found

   t= time.time()

   if (t < (t_iter_start + t_lim_iter)) and (t < (t_start + t_lim)):
    if len(rem_vs) == 0:

   
        res = PathQuality(red_path,orig_s_times,cData)
        if res == None:
            res = PathQualityInfeasible(red_path,orig_s_times,cData)
            #print('infeas res is , ', res)

        if res < best_obj:
            best_obj = res
            best_path = red_path

            #check if this is the first time a feasible solution is found
            #big M constant:
            M = len(cData)*len(cData)*16
            if best_obj < M:
                feasible_found = True
                #print('feasible found at ', str(t-t_start))

            
            if feasible_found == True:
                time_list.append(t - t_start)
                cost_list.append(best_obj)

            #print('------------------NEW BEST OBJ FOUND:' , best_obj)
            #print('------------------NEW BEST Path FOUND: \n' , best_path)


            #Check if within optimality gap
            d_optimal = abs(optimal-res)
            if d_optimal <= og*optimal:
                #print("Arrived at the optimal solution after ", round((t - t_start),1), ' s')
                optim_status = True
                
    else:
      #chose which visit to branch on and which values to explore
      v, v_ind, P = ChooseVisit(red_path,rem_vs,orig_s_times,cData,d)


      if P != None:
            # Implement LDS (see paper)
            i = 0
            p = 0

            while i <= d and p < len(P):

                veh = P[p][1]
                veh_path = red_path[veh]
                veh_path.insert(P[p][0],v)

                #save removed visit
                v_saved = rem_vs[v_ind]
                #delete reinserted visit from rem_vs
                del rem_vs[v_ind]


                Reinsert(cp.deepcopy(red_path),cp.deepcopy(rem_vs),(d-i),optimal, og, orig_s_times,cData, t_lim, t_lim_iter, time_list, cost_list)
                
                #Put back the added visit to removed visit list
                rem_vs.insert(v_ind,v_saved)


                #delete removed visit from path
                del veh_path[P[p][0]]

                
                i += 1
                p += 1





def run_LNS(frac, original_file, original_soln_file, MPP_file, output_file, cost_file, d, optimal, og, t_lim, t_lim_iter):
    #run the LNS algorithm
    #parameters: frac -  The fraction of visits that will be removed and reinserted for each iteration of local search.


    #Read in MPP problem file
    mpp_problem = util.VRPTWInstance.load(MPP_file)

    #get number of customers and vehicles
    nCus = len(mpp_problem.nodes)

    #read customer data, each sublist corresponds to one customer, with index matching customer index, and has data [xCoord, yCoord, twStart, twEnd]
    cData = []

    for c in mpp_problem.nodes:
        int_params = [c.x, c.y, c.a, c.b]
        cData.append(int_params)
        

    #read customer data from original problem
    original_problem = util.VRPTWInstance.load(original_file)
    
    orig_cData = []

    for c in original_problem.nodes:
        int_params = [c.x, c.y, c.a, c.b]
        orig_cData.append(int_params)


    #get original paths/routes, same data form as the candidate routes
    original_path, orig_s_times = readPath(original_soln_file,nCus)

    #to start, set candidate path to original
    path = original_path

    #get a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]
    all_visits = makeVisitDataInfeasible(cp.deepcopy(path),cData)
    #print('all_visits ', all_visits, '\n')

    #initialize global best_path and best_obj variables
    global best_path
    global feasible_found
    global best_obj
    global t_start
    global t_iter_start

    feasible_found = False

    #tracks if optimality has been reached
    global optim_status
    optim_status = False
    

    best_obj = float('inf')
    #print('initial best obj ', best_obj)

    best_path = path

    #print('initial best obj', best_obj)
    
    #iteration count
    i = 0

    time_list = []
    cost_list = []

    while (time.time() < (t_start + t_lim) and optim_status == False):
        #print('iteration ' ,i)
        
        #print(best_path)

        #update all_visits
        all_visits = makeVisitDataInfeasible(cp.deepcopy(best_path),cData)

        rem_vs, red_path = removeVisits(cp.deepcopy(best_path),cp.deepcopy(all_visits),cData,float(frac))

        #update iteration start time
        t_iter_start = time.time()
        Reinsert(red_path,rem_vs,d,optimal, og, orig_s_times,cData, t_lim, t_lim_iter, time_list, cost_list)

        #update iteration count
        i += 1
    
    #print('best obj at end is ', best_obj)

    #only write if there is a feasible solution
    if feasible_found == True:
        #write path:
        mpp_problem.dump_routes_with_time(best_path, output_file)

        result = {'cost': cost_list, 'time': time_list, 'optimal': False}

        with open(cost_file, 'w') as f:
            if feasible_found == True:
                json.dump(result, f, indent=4)
    

if __name__ == '__main__':
    #start time
    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', '-f', type=float, default=0.25,
                        help='The fraction of visits that will be removed and reinserted for each iteration of local search.')
    parser.add_argument('--original-problem', type=str, required=True)
    parser.add_argument('--original-solution', type=str, required=True)
    parser.add_argument('--perturbated-problem', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--optimal', '-o', type=str, default = 0,
                        help='the optimal solution, if available. If not provided, valued at 0. ')
    parser.add_argument('--optimality_gap', '-og', type=str, default = 0.0001, help = 'optimality gap, i.e 0.0001 is 0.01%.')
    parser.add_argument('--d', '-d', type=str, default = 5,
                        help='parameter for limited discrepancy search, as per paper')

    parser.add_argument('--t_lim', '-t', type=str, default = 120,
                        help='total time limit in seconds')
    parser.add_argument('--t_lim_iter', '-ti', type=str, default = 5,
                        help='iteration time limit in seconds')

    args = parser.parse_args()

    if not 0 < args.frac < 1:
        print('frac must be between 0 and 1')

    else:
        print('running LNS...')
        run_LNS(args.frac, args.original_problem, args.original_solution, args.perturbated_problem, args.output, args.cost, int(args.d), float(args.optimal), float(args.optimality_gap), int(args.t_lim), int(args.t_lim_iter))



