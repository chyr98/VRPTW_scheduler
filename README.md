# MPP in VRPTW

## VRPTW Generator

```bash
python3 problem_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5 \
  --tw-limit 3 \
  --output problem1.txt \
  --solution solution_problem1.txt \
  --seed 1234
```

- `--name/-n`: the name of problem (default: 'VRPTW instance')
- `--num-vehicles/-v`: the number of vehicles (default: 2)
- `--x-limit/-x`: the maxumum value of the x axis (default: 10)
- `--y-limit/-y`: the maxumum value of the y axis (defualt: 10)
- `--num-costomers/-c`: the number of customers including the depot (default :5)
- `--tw-limit/-t`:  The half of the maximum width of time windows (defualt: 3)
- `--output/-o`: the path to save the generated problem (required)
- `--solution/-s`: The path to save the solution used for generating the problem (optional)

## MPP Generator
Generate MPP by swapping two customers in the original routes.
If `--original-problem` and `--original-solution` exist, they will be used.
Otherwise, they will be generated accodring to the same options as `problem_generator.py`.
Note that `--tw-limit` should be set appropriately even if an existing problem and solution are used.

```bash
python3 mpp_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5 \
  --tw-limit 3 \
  --num-perturbations 3 \
  --original-output problem1.txt \
  --perturbated-output perturbated_problem1.txt \
  --solution solution_problem1.txt \
  --seed 1234
```

- `--tw-limit/-t`:  The half of the maximum width of time windows (defualt: 3)
- `--num-perturbations/-m`: the number of swaps (default: 3)
- `--original-problem/-o`: the path of the original problem (required)
- `--perturbated-problem/-p`: the path to save the perturbated problem (required)
- `--solution/-s`: The path of the original solution (required)
- `--pertrubated-solution/-q`: the path to save the solution used for generating the perturbated problem (optional)

### Problem Format

```
# problem1
2
5
1 6 0 0
6 6 5 8
9 9 15 19
9 3 5 12
8 9 11 16
```
- the first line is problem name
- the second line is the number of vehicles
- the third line is the number of customers
- the subsequenting lines describe customers
  - each line describes one customer (the first line is the depot)
  - the first column is the x-coordinate
  - the second column is the y-coordinate
  - the third column is the ready time
  - the fourth column is the due time

### Solution Format

```
# prolem1
2
1 2
3 4
```
- the first line is problem name
- the second line is the number of vehicles
- the subsequenting lines describe routes
  - each line describes one route
  - routes are listed in the order of the indices of the vehicles
  - customers are listed in the order of the visits in each line
  - the depot is omitted

## Solution Validator

```bash
python3 validate_solution.py --problem problem1.txt --solution solution_problem1.txt
vehicle 0
        0: 0.0, 3: 8.54, 4: 14.63, 2: 15.63
vehicle 1
        0: 0.0, 1: 5.00
total cost: 34.17
```

- output routes of vehicles and the starting time at each customer



## MIP model

run_MIP.py will search for all data instances in the benchmark folder that matches the "params" variable defined in the run_MIP.py file. Also it will save appropriate outputs to the folder "./mip_solution". If this folder does not exist, a new folder will be created.

- Saves all solutions to "./mip_solution" as per format defined in https://gist.github.com/Kurorororo/21ccc9ecbea2191a52f62e4bed2224db
- Saves all metrics and parameters to "./mip_solution/results.json"

```python3 run_MIP.py```

This depends on numpy, CPLEX, and docplex.

In addition, for each input parameter set, cost improvement over time is tracked and the data points are saved in ./mip_solutions folder as a JSON object. An example output can be found at ./mip_solutions/mip_results8_v4_c16_tw4_xy16_0.json


## Backtrack Search Model

backtrack_search.py takes six command line arguments:

- The path to the file that stores the original VRPTW problem
- The path to the file that stores the perturbed problem
- The path to the solution file of the original problem
- The path of the output file that stores the solution of MPP
- The path of the output file that stores the set of costs at different time steps.
- The time limit for each optimization

Solution route for MPP will be written into the solution file with the given path.\\
Costs vs time steps will be written into the cost file with the given path in the following format.\\
  {"cost": [cost1, cost2, ..., costn], "time": [10, 20, ..., time_limit]}

```python3 backtrack_search.py -i <problem file> -I <perturbed problem file> -s <original solution file> -O <perturbed solution outputfile> -c <cost file> -t <time limit>```

This depends on Gurobi and gurobipy.

## Experiments

```
python3 run_experiment.py \
  -b preliminary_benchmarks \
  -t 180 \
  -o results \
  -w 1 \
  -m methods.json \
  -n 1
```

- `-b`: the directory containing a benchmark set
- `-t`: the time limit
- `-o`: the directory to save the results
  - `results/result.json` is generated if `-o results`
- `-w`: the number of processes to run in parallel
- `-m`: a JSON file describing the methods
- `-n`: the number of runs

### A JSON File for the Mehtod
The file contains a list of mehtods.
"name" specifies the name of the method.
"cmd" specifies the command to run.
`{original_problme}`, `{perturbated_problem}`, `{original_solution}`, and `{output}` are automatically replaced by appropriate file names by `run_experiment.py`.


```json
[
  {
    "name": "backtrack",
    "cmd": "python3.7 backtrack_search.py -i {original_problem} -I {perturbated_problem} -s {original_solution} -O {output} -c {cost}"
  },
  {
    "name": "LNS",
    "cmd": "python3.7 LNS.py --original-problem {original_problem} --perturbated-problem {perturbated_problem} --original-solution {original_solution} --perturbated-solution {perturbated_solution} --output {output} --cost {cost} --t_lim 178"
  },
  {
    "name": "MIP",
    "cmd": "python3.7 run_mip.py --original-problem {original_problem} --perturbated-problem {perturbated_problem} --original-solution {original_solution} --output {output} --cost {cost}"
  }
]
```
