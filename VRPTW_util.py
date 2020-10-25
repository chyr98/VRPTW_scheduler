from __future__ import annotations
import math
from typing import List, Optional, Tuple


class Node:
    """
    A Node stands for a customer in this problem setting, which has an index that
    makes enumeration easier, x and y coordinates, and a time window for possible service time.
    """

    def __init__(self, index: int, x: int, y: int, a: float = 0.0, b: float = 0.0) -> __class__:
        self.index = index
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def update_time_window(self, a: float, b: float) -> None:
        """
        Update the available time windows for the customer

        :param lower_bound: The earliest service time for customer
        :param upper_bound: The latest service time for customer
        :return: None
        """
        self.a = a
        self.b = b

    def distance(self, other: __class__) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def dump_routes(problem_name: str, routes: List[List[int]], filename: str):
    with open(filename, 'w') as f:
        f.write('# {}\n'.format(problem_name))
        f.write(str(len(routes)) + '\n')
        for r in routes:
            f.write(' '.join([str(c) for c in r]) + '\n')


class VRPTWInstance:
    """
    A VRPTW instance contains a number of customers that require services, each of them has a
    specific time window restricts the service time.
    There is a fleet of vehicles to provide services and we cannot change the amount of
    vehicles in this setting.
    The map shows the distance between every customers and the depot.

    self.nodes[0] stands for the depot
    """

    def __init__(self, name: str, num_vehicles: int, nodes: List[Node]) -> __class__:
        self.num_vehicles = num_vehicles
        self.nodes = nodes
        self.name = name

    def distance(self, i: int, j: int) -> float:
        return self.nodes[i].distance(self.nodes[j])

    def get_time(self, routes: List[List[int]]) -> Optional[float, List[Tuple[int, float]]]:
        node_to_vehicle_time = [(-1, -1.0)] * len(self.nodes)
        node_to_vehicle_time[0] = (-1, 0.0)
        cost = 0.0

        for i, r in enumerate(routes):
            t = 0
            prev = 0

            for cur in r:
                distance = self.distance(prev, cur)
                cost += distance
                t = max(t + distance, self.nodes[cur].a)

                # too late
                if t > self.nodes[cur].b:
                    return None

                # already visited
                if node_to_vehicle_time[cur][0] != -1:
                    return None

                node_to_vehicle_time[cur] = (i, t)
                prev = cur

            cost += self.distance(prev, 0)

        # a node is not visited
        if any([t[0] == -1 for t in node_to_vehicle_time[1:]]):
            return None

        return cost, node_to_vehicle_time

    def dump(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write('# {}\n'.format(self.name))
            f.write(str(self.num_vehicles) + '\n')
            f.write(str(len(self.nodes)) + '\n')

            for node in self.nodes:
                f.write('{} {} {} {}\n'.format(node.x, node.y, node.a, node.b))

    @classmethod
    def load(cls, filename: str) -> __class__:
        with open(filename) as f:
            name = f.readline().rstrip()[2:]
            num_vehicles = int(f.readline().rstrip())
            num_nodes = int(f.readline().rstrip())
            nodes = []

            for i in range(num_nodes):
                node_info = f.readline().split()
                x = int(node_info[0])
                y = int(node_info[1])
                a = float(node_info[2])
                b = float(node_info[3])
                nodes.append(Node(i, x, y, a, b))

        return cls(name, num_vehicles, nodes)
