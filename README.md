# MPP in VRPTW

All codes are verified with Python 3.7.9.

## Common Resources

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
A solution is described in a following JSON file.

```json
{
  "name": "problem1",
  "routes": [
    [1, 2],
    [3, 4]
  ],
  "time": [
    [10.0, 20.0],
    [15.0, 30.0]
  ]
}
```
- "routes" describes routes
  - each list describes one route
  - routes are listed in the order of the indices of the vehicles
  - customers are listed in the order of the visits in each line
  - the depot is omitted
- "time describes the service time
  - each list describes the service time for each customer on a route

### VRPTW Generator

```bash
python3 problem_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5 \
  --tw-limit 3 \
  --output problem1.txt \
  --solution solution_problem1.json \
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

### MPP Generator
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
  --solution solution_problem1.json \
  --seed 1234
```

- `--tw-limit/-t`:  The half of the maximum width of time windows (defualt: 3)
- `--num-perturbations/-m`: the number of swaps (default: 3)
- `--original-problem/-o`: the path of the original problem (required)
- `--perturbated-problem/-p`: the path to save the perturbated problem (required)
- `--solution/-s`: The path of the original solution (required)

### Benchmark Generation

We generated a benchmark set using the following program.

```bash
python3 generate_benchmarks.py
```

### Solution Validator

```bash
python3 validate_solution.py --problem problem1.txt --solution solution_problem1.json
vehicle 0
        0: 0.0, 3: 8.54, 4: 14.63, 2: 15.63
vehicle 1
        0: 0.0, 1: 5.00
total cost: 34.17
```

- output routes of vehicles and the service time at each customer

## Methods

### MIP Models Using CPLEX

#### Reuirements
- CPLEX >= 12.9
- docplex >= 2.11.176
- numpy >= 1.19.4

#### Run the Code

```bash
python3 mip_cplex.py \
  --original-problem problem1.txt \
  --original-solution solution1.json \
  --perturbated-problem problem2.txt \
  --output solution2.json \
  --cost cost.json \
  --thread 1 \
  --use-inequality-model 0
```

- `--original-problem`: the path to the file that stores the original VRPTW problem
- `--original-solution`: the path to the file that stores the perturbed problem
- `--perturbated-problem`: the path of the output file that stores the solution of MPP
- `--output`: the path of the output file that stores the solution of MPP
- `--cost`: the path of the output file that stores the set of costs at different time steps.
- `--threads`: the number of threads (default: 1)
- `--use-inequality-model`: use the equality constraints (model1) if 0 and inequality constraints (model2) if 1 (default: 1)

### MIP Models Using Gurobi

#### Requiremetns
- Gurobi >= 9.1.0
- gurobipy >= 9.1.0
- numpy >= 1.19.4

### Run the Code

```bash
python3 mip_gurobi.py \
  -i <problem file> \
  -I <perturbed problem file> \
  -s <original solution file> \
  -O <perturbed solution outputfile> \
  -c <cost file> 
  -t <threads>
  -g <any>
```

- `-i`: the path to the file that stores the original VRPTW problem
- `-I`: the path to the file that stores the perturbed problem
- `-s`: the path of the output file that stores the solution of MPP
- `-O`: the path of the output file that stores the solution of MPP
- `-c`: the path of the output file that stores the set of costs at different time steps.
- `-t`: the number of threads (default: 1)
- `-g`: use the equality constraints (model1) if there is and ineqality constraints (model2) if not

### Large Neighborhood Search (LNS)

#### Run the Code

```bash
python3 LNS.py \
  --original-problem problem1.txt \
  --original-solution solution1.json \
  --perturbated-problem problem2.txt \
  --output solution2.json \
  --cost cost.json \
  --frac 0.25 \
  --d 5 \
  --t_lim 178 \
  --t_lim_iter 5
```

- `--original-problem`: the path to the file that stores the original VRPTW problem
- `--original-solution`: the path to the file that stores the perturbed problem
- `--perturbated-problem`: the path of the output file that stores the solution of MPP
- `--output`: the path of the output file that stores the solution of MPP
- `--cost`: the path of the output file that stores the set of costs at different time steps.
- `--frac/-f`: the fraction of visits that will be removed and reinserted for each iteration of local search (default: 0.25)
- `--d/-d`: parameter for limited discrepancy search (default: 5)
- `--t_lim/-t`: total time limit in seconds (default: 6)
- `--t_lim_iter/-ti`: iteration time limit in seconds (default: 5)

## Experiments

```bash
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
`{original_problme}`, `{perturbated_problem}`, `{original_solution}`, `{output}`, and `{cost}` are automatically replaced by appropriate file names by `run_experiment.py`.


```json
[
  {
    "name": "LNS",
    "cmd": "python3.7 LNS.py --original-problem {original_problem} --perturbated-problem {perturbated_problem} --original-solution {original_solution} --output {output} --cost {cost} --t_lim 178"
  },
  {
    "name": "CPLEX1",
    "cmd": "python3.7 mip_cplex.py --original-problem {original_problem} --perturbated-problem {perturbated_problem} --original-solution {original_solution} --output {output} --cost {cost}"
  },
  {
    "name": "CPLEX2",
    "cmd": "python3.7 mip_cplex.py --original-problem {original_problem} --perturbated-problem {perturbated_problem} --original-solution {original_solution} --output {output} --cost {cost} --use-Jasper-model"
  },
  {
    "name": "Gurobi1",
    "cmd": "python3.7 mip_gurobi.py -i {original_problem} -I {perturbated_problem} -s {original_solution} -O {output} -c {cost} -g"
  },
  {
    "name": "Gurobi2",
    "cmd": "python3.7 mip_gurobi.py -i {original_problem} -I {perturbated_problem} -s {original_solution} -O {output} -c {cost}"
  }
]
```

### Visualization of Experimental Results

`./summarize_results/` containing scripts to create a LaTex table and figures to summarize the results.
