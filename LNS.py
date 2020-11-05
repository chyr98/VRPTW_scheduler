import copy as cp
import argparse
import random
import math

def readPath(filepath,nVeh):
    #reads a solution file and returns a list of sublists for each vehicle route

    f = open(filepath,'r')
    f.readline()
    f.readline()

    #record the route of each vehicle in a sublist
    #this is the first candidate solution
    path = []
    for i in range(nVeh):
        params = [int(i) for i in f.readline().split()]
        path.append(params)

    return path

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
    #given a set of removed visits and a reduced path, select the variable (i.e return customer ID) to branch on and return the top d points (only these will be searched)

    #make a list of feasible insertion points for each visit (ind, V, c), inserting the visit at index ind in the route of vehicle v, at cost c

    #find the variable with the maximum minimum cost
    max_min_cost_tot = 0
    #print('\n reduced path is ', r_path)
    #print('\n removed visits are ', rem_vs)


    for i in range(len(rem_vs)):
        f_pts = []

        #print('------------------------------------------considering customer ' , rem_vs[i][0])

        for j in range(len(r_path)):
            #print('---------------------------considering vehicle ' , j)
            for k in range(len(r_path[j]) + 1):
                #print('considering insertion index ' , k)
                
                visit = rem_vs[i][0]
                r_path[j].insert(k,visit)

                cost = PathQuality(r_path,orig_s_times,cData)
                
                if cost != None:
                    f_pts.append((k, j , cost))
                #else:
                #    print('infeasible, (point, veh, cost):' , (k, j , cost))

                del r_path[j][k]


        #print('------------------------------------------------------------------------------------------------')
        #print(' RESULTS FOR CUSTOMER  ' , rem_vs[i][0])
        #print('unsorted feasible points', f_pts)
        f_pts.sort(key = lambda x:x[2])
        #print('sorted', f_pts)

        max_min_cost = f_pts[0][2] 
        if max_min_cost >= max_min_cost_tot:
            #print('new_max_min_cost found: , ' , max_min_cost)
            v = rem_vs[i][0]
            P = f_pts[0:d]
            max_min_cost_tot = max_min_cost

    return (v, P)

    
    

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





def Reinsert(path, rem_vs):
    #takes a reduced path, visits to reinsert (rem_vs), and a lower bound on the best solution,
    #Explores the search tree of reinserting all of  the visits, considering only the top d (i.e. 5) choices for each variable based on the heuristic in the paper
    #the ordering heuristic is to choose the worst variable (i.e. the one with the maximum insertion cost to the objective) then select its best value

    #d is discrepancy limit: set to 5 as per paper
    d = 5
    





def run_LNS(frac, bPath, pInfo, nSwaps,d):
    #run the LNS algorithm
    #parameters: frac -  The fraction of visits that will be removed and reinserted for each iteration of local search.
    #bPath: benchmarks folder path
    #pInfo: details of the problem - will be added to create filepaths such as: "perturbated1_v4_c64_tw4_xy16_1.txt" 
    #nSwaps: number of swaps that the MPP was generated with

 

    #filepaths
    MPP_file = bPath + '\\perturbated' + str(nSwaps) + pInfo + '.txt'
    MPP_soln_file = bPath + '\\perturbated' + str(nSwaps) + "_solution" + pInfo + '.txt'
    original_soln_file = bPath + '\\solution' + str(pInfo) + '.txt'
    original_file = bPath +'\\original' + str(pInfo) + '.txt'

    #Read in MPP problem file
    f1 = open(MPP_file,'r')
    f1.readline()

    #get number of customers and vehicles
    nVeh = int(f1.readline())
    nCus = int(f1.readline())

    #read customer data, each sublist corresponds to one customer, with index matching customer index, and has data [xCoord, yCoord, twStart, twEnd]
    cData = []

    for i in range(nCus):
        params = f1.readline().split()
        int_params = [int(i) for i in params]
        cData.append(int_params)
        

    #read customer data from original problem
    f2 = open(original_file)
    
    #ignore first lines
    f2.readline()
    f2.readline()
    f2.readline()

    orig_cData = []

    for i in range(nCus):
        params = f2.readline().split()
        int_params = [int(i) for i in params]
        orig_cData.append(int_params)


    #get MPP input solution: the route of each vehicle is stored in a sublist in path. This is the input solution to LNS
    path = readPath(MPP_soln_file,nVeh)

    #get original paths/routes, same data form as the candidate routes
    original_path = readPath(original_soln_file,nVeh)

    #get a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]
    all_visits = makeVisitData(cp.deepcopy(path),cData)
    original_visits = makeVisitData(cp.deepcopy(original_path), orig_cData)

    #get initial objective value for MPP
    orig_s_times = GetSTimes(nCus,nVeh, cp.deepcopy(original_visits))
    
    #Get service times from initial solution
    s_times = GetSTimes(nCus, nVeh, cp.deepcopy(all_visits))
    
    best_obj = Objective(cp.deepcopy(s_times),cp.deepcopy(orig_s_times))
    


    print('initial best obj', best_obj)




    #get removed visits and the reduced path
    rem_vs, red_path = removeVisits(cp.deepcopy(path),cp.deepcopy(all_visits),cData,float(frac))

    print('runing choosevisit: \n')
    (v,P) = ChooseVisit(cp.deepcopy(red_path), cp.deepcopy(rem_vs),orig_s_times,cData,d)
    
    print('selected variable and top points: ', v, ' \n ', P)

    

    

    #TO DO: check copy for all list args. CData is OK since not modified.

    soln = []
    return soln

def write_Soln(soln):

    pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', '-f', type=float, default=0.25,
                        help='The fraction of visits that will be removed and reinserted for each iteration of local search.')
    parser.add_argument('--benchmarks-dir', '-b', type=str, required=True,
                        help='The path of the benchmarks folder. Ex. if in same folder, then .\benchmarks')
    parser.add_argument('--prob-info', '-p', type=str, required=True,
                        help='The details of the problem in the format of the benchmark names, i.e. _v1_c4_tw4_xy16_0')
    parser.add_argument('--nSwaps', '-s', type=str, default=2, help='The number of swaps')
    parser.add_argument('--optimal', '-o', type=str, default = 0,
                        help='the optimal solution, if available. If not provided, valued at 0. ')
    parser.add_argument('--d', '-d', type=str, default = 5,
                        help='parameter for limited discrepancy search')

    args = parser.parse_args()

    if not 0 < args.frac < 1:
        print('frac must be between 0 and 1')

    else:
        print('running LNS...')
        soln = run_LNS(args.frac, args.benchmarks_dir, args.prob_info, args.nSwaps,int(args.d))

        print('saving solution...')
        write_Soln(soln)

