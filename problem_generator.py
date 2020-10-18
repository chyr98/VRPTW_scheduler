import VRPTW_util as util
import random
import sys


def generate_problem(num_nodes: int, num_vehicles: int, lb_distance: float, ub_distance: float, tw_variance=3.0) -> util.VRPTWProblem:
    """
    Generates a VRPTW Problem with the given parameters.

    Precondition: lb_distance <= ub_distance <= 2 * lb_distance (to maintain the triangle inequality)

    :param num_nodes: number of customers in the problem
    :param num_vehicles: number of vehicles that can provide services
    :param lb_distance: the lower bound of the distance from any location to another
    :param ub_distance: the upper bound of the distance from any location to another
    :param tw_variance: the factor indicates how wide can the time window be in terms of multiple of ub_distance
    :return: a well defined VRPTW Problem.
    """
    random.seed(1234)

    nodes = []
    for i in range(num_nodes):
        nodes.append(util.Node(i))

    dt = util.DistanceTable(num_nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dt.update_distance(nodes[i], nodes[j], random.uniform(lb_distance, ub_distance))

    # To make sure the problem is feasible, we will generate a random route and set the time windows
    # of tasks to match that route.
    route = nodes[1:] # the first node is depot
    random.shuffle(route)
    distance = 0
    prev_node = nodes[0]
    for curr_node in route:
        distance += dt.get_distance(prev_node, curr_node)
        curr_node.update_time_window(max(0.0, distance - tw_variance * random.uniform(0,ub_distance)),
                                     distance + tw_variance * random.uniform(0, ub_distance))
        prev_node = curr_node

    return util.VRPTWProblem(num_vehicles, nodes, dt)


def main(argv) -> util.VRPTWProblem:
    """
    Generates a VRPTW problem and possibly print the problem in a file based on user's choice

    Command line usage: problem_generator.py <# of customers> <# of vehicles> -o <output file>
    The problem won't be displayed without -o flag.

    :param argv: command line arguments
    :return: A VRPTW problem that is in same size as user's inputs
    """
    usage_message = "usage: " + argv[0] + " <# of customers> <# of vehicles> -o <output file>"
    args = argv[1:]
    if not (len(args) == 2 or len(args) == 4):
        print(usage_message)
        sys.exit(2)
    try:
        num_customers = int(args[0])
        num_vehicles = int(args[1])
    except ValueError:
        print(usage_message)
        sys.exit(2)

    lb = random.uniform(10, 100)
    up = lb * 2
    problem = generate_problem(num_customers, num_vehicles, lb, up)

    # If user choose to print the problem in some file
    if len(args) == 4:
        if args[2] != "-o":
            print(usage_message)
            sys.exit(2)
        f = open(args[3], "w")
        f.write(str(problem))
        f.close()
    return problem


if __name__ == "__main__":
    the_VRPTW = main(sys.argv)
    print(the_VRPTW)