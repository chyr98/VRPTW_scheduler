from typing import List


class Node:
    """
    A Node stands for a customer in this problem setting, which has an index that
    makes enumeration easier and a time window for possible service time.
    """
    def __init__(self, index: int):
        self.index = index
        self.time_window = (0.0, 0.0)

    def __str__(self):
        return "{" + str(self.index) + ", " + str(self.time_window) + "}"

    def update_time_window(self, lower_bound: float, upper_bound: float) -> None:
        """
        Update the available time windows for the customer

        :param lower_bound: The earliest service time for customer
        :param upper_bound: The latest service time for customer
        :return: None
        """
        self.time_window = (lower_bound, upper_bound)


class Route:
    """
    A Route is a sequence of customers that a vehicle can visit in order.

    *** This class seems not necessary since a list of nodes can do the same thing for now ***
    *** We will see if there is any method that needs to be in this class                  ***
    """
    def __init__(self, nodes: List[Node]):
        self.length = len(nodes)
        self.nodes = nodes

    def __getitem__(self, key: int) -> Node:
        return self.nodes[key]

    def __setitem__(self, key: int, value: Node) -> None:
        self.nodes[key] = value

    def insert(self, key: int, value: Node) -> None:
        self.nodes.insert(key, value)


class DistanceTable:
    """
    A table contains distances from node to node, since travel from i to j is same
    distance as from j to i, the table is upper triangular and half of the table
    is ignored in storage.
    Example: DistanceTable for 4 nodes [0, 1, 2, 3]
           0      1      2      3
    0   d_00   d_01   d_02   d_03
    1          d_11   d_12   d_13
    2                 d_22   d_23
    3                        d_33

    d_ii stands for distance from i to i, which is 0
    """
    def __init__(self, number_of_nodes: int):
        self.number_of_nodes = number_of_nodes
        self.table = []
        for i in range(number_of_nodes):
            self.table.append([0] * (number_of_nodes - i))

    def __str__(self):
        string = ""
        for i in range(self.number_of_nodes):
            string += "\t" + str(i)
        string += "\n"
        for i in range(self.number_of_nodes):
            string += str(i) + "\t" * i
            for j in range(i, self.number_of_nodes):
                string += "\t{0:.2f}".format(self.table[i][j-i])
            string += "\n"
        return string

    def update_distance(self, node_i: Node, node_j: Node, distance: float) -> None:
        """
        Update the distance for traveling from customer i to customer j or vice versa

        :param node_i: Node stands for customer i
        :param node_j: Node stands for customer j
        :param distance: New distance from i to j
        :return: None
        """
        i = node_i.index
        j = node_j.index
        if i >= j:
            self.table[j][i - j] = distance
        else:
            self.table[i][j - i] = distance

    def get_distance(self, node_i: Node, node_j: Node) -> float:
        """
        Get the distance of traveling from customer i to customer j

        :param node_i: Node stands for customer i
        :param node_j: Node stands for customer j
        :return: The distance from i to j
        """
        i, j = node_i.index, node_j.index
        return self.table[i][j-i] if i < j else self.table[j][i-j]

    def get_total_distance(self, route: Route) -> float:
        """
        Get the total distance traveled following the Route route, which
        is the sum of the distances between every consecutive pairs of nodes
        in route.

        :param route: A possible traveling route.
        :return: The total distance of this route.
        """
        distance = 0
        for i in range(route.length - 1):
            distance += self.get_distance(route[i], route[i+1])
        return distance


class VRPTWProblem:
    """
    A VRPTW Problem contains a number of customers that require services, each of them has a
    specific time window restricts the service time.
    There is a fleet of vehicles to provide services and we cannot change the amount of
    vehicles in this setting.
    The map shows the distance between every customers and the depot.

    self.nodes[0] stands for the depot
    """
    def __init__(self, num_vehicles: int, nodes: List[Node], distances: DistanceTable):
        self.num_vehicles = num_vehicles
        # self.routes = []
        self.nodes = nodes
        self.map = distances

    def __str__(self):
        string = "VRPTW Problem with " + str(len(self.nodes)) + " Customers, " + str(self.num_vehicles) + " Vehicles\n"
        string += "Set of Customers: " + str([str(node) for node in self.nodes]) + "\n"
        string += str(self.map)
        return string


