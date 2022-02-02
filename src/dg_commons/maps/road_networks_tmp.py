from __future__ import annotations
import time
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping, Set, Dict, Any, Callable, Union
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.planning.planning_problem import PlanningProblem
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, LineCollection
from dg_commons.sim import SimObservations, SimTime
from dg_commons import PlayerName
from copy import deepcopy
from shapely.geometry import Polygon, LinearRing, Point
from shapely.strtree import STRtree
from .utils import *
from .prediction_structures_tmp import Prediction, PlayerGoalBools, PlayerGoalLikelihoods, PlayerGoalRewards

# factor to make resource ids unique
resource_id_factor = 100


# fixme: temporary workaround to plot with digraph as input. This can be used to avoid deepcopying entire
# dynamic graph but only Digraph at each simulation timestep.
def get_collections_networkx_temp(resource_graph: DiGraph(), ax: Axes = None) -> Tuple[PathCollection, LineCollection,
                                                                                       Mapping[int, plt.Text]]:
    """
    Get collections for plotting a graph on top of a scenario

    :param ax: Axes on which to draw the Artists
    :param resource_graph: graph to draw
    """
    nodes = resource_graph.nodes
    edges = resource_graph.edges
    cents = []
    for node in nodes.data():
        cents.append(node[-1]['polygon'].centroid.coords[0])

    centers = dict(zip(nodes.keys(), cents))

    # set default edge and node colors
    edge_colors = ['k'] * len(resource_graph.edges)
    node_colors = ['#1f78b4'] * len(resource_graph.nodes)

    # set special node and edge colors depending on node and edge attributes
    for node in nodes.data():
        node_idx = list(nodes).index(node[0])
        if node[1]['goal_of_interest']:
            node_colors[node_idx] = 'magenta'
        if node[1]['ego_occupied_resource']:
            node_colors[node_idx] = 'r'
        if node[1]['occupied_by_agent']:
            node_colors[node_idx] = 'cyan'
        if node[1]['goal']:
            node_colors[node_idx] = 'limegreen'
        if node[1]['start']:
            node_colors[node_idx] = 'gold'

    # color edges possibly occupied by ego
    for edge in edges.data():
        edge_idx = list(edges).index((edge[0], edge[1]))
        if edge[2]['ego_occupied_edge']:
            edge_colors[edge_idx] = 'r'

    # return collections and set zorders
    # the functions draw_* already plots on axes
    nodes_plot = draw_networkx_nodes(G=resource_graph, ax=ax, pos=centers, node_size=200, node_color=node_colors)
    nodes_plot.set_zorder(50)

    edges_plot = draw_networkx_edges(G=resource_graph, ax=ax, pos=centers, edge_color=edge_colors)
    for edge in range(len(edges_plot)):
        edges_plot[edge].set_zorder(49)

    labels_plot = draw_networkx_labels(G=resource_graph, ax=ax, pos=centers)
    for label in labels_plot.keys():
        labels_plot[label].set_zorder(51)

    return nodes_plot, edges_plot, labels_plot


def get_weight_from_lanelets(lanelet_network: LaneletNetwork, id_lanelet_1: int, id_lanelet_2: int) -> float:
    """
    Adapted from Commonroad Route Planner.
    Calculate weights for edges on graph.
    For successor: calculate average length of the involved lanes.
    For right/left adjacent lanes: calculate average road width.
    """
    if id_lanelet_2 in lanelet_network.find_lanelet_by_id(id_lanelet_1).successor:
        length_1 = lanelet_network.find_lanelet_by_id(id_lanelet_1).distance[-1]
        length_2 = lanelet_network.find_lanelet_by_id(id_lanelet_2).distance[-1]
        return (length_1 + length_2) / 2.0
    # rough approximation by only calculating width on first point of polyline
    elif id_lanelet_2 == lanelet_network.find_lanelet_by_id(id_lanelet_1).adj_left \
            or id_lanelet_2 == lanelet_network.find_lanelet_by_id(id_lanelet_1).adj_right:
        width_1 = np.linalg.norm(lanelet_network.find_lanelet_by_id(id_lanelet_1).left_vertices[0]
                                 - lanelet_network.find_lanelet_by_id(id_lanelet_1).right_vertices[0])
        width_2 = np.linalg.norm(lanelet_network.find_lanelet_by_id(id_lanelet_2).left_vertices[0]
                                 - lanelet_network.find_lanelet_by_id(id_lanelet_2).right_vertices[0])
        return (width_1 + width_2) / 2.0
    else:
        raise ValueError("You are trying to assign a weight but no edge exists.")


def split_lanelet_into_polygons(lanelet: Lanelet, max_length: Optional[float]) -> List[tuple[Polygon, int]]:
    """
    Split a lanelet in smaller polygons by dividing uniformly along the centerline in curvilinear coordinates.
    Return list of polygons and unique index for each polygon.

    :param lanelet: Commonroad lanelet to divide into smaller polygons
    :param max_length:  Maximum length along the lanelet centerline which is accepted as centerline
                        length for a polygon. The algorithm will try and make polygons with uniform length along the
                        centerline and length closes to max_length as possible.
    """

    lanelet_length = lanelet.distance[-1]

    # if lanelet is too short, return polygon without diving it
    if max_length is None or lanelet_length <= max_length:
        return [(lanelet.polygon.shapely_object, lanelet.lanelet_id * resource_id_factor)]

    n_polygons = lanelet_length // max_length + 1
    polygon_length = lanelet_length / n_polygons

    left_vertices = lanelet.left_vertices
    right_vertices = lanelet.right_vertices

    counter = 0
    current_polygon_base_id = lanelet.lanelet_id * resource_id_factor

    polygons = []

    current_polygon_vertices_l = []
    current_polygon_vertices_r = []

    previous_index_before = 0

    # handle first polygon with n=0
    previous_left_interp = Point(left_vertices[0])
    previous_right_interp = Point(right_vertices[0])

    for n in range(int(n_polygons)):

        current_distance = (n + 1) * polygon_length

        # return index in distance vector after the current distance
        # index after is expected to always be >1
        index_after = next((el_idx for el_idx, el in enumerate(lanelet.distance) if el > current_distance), None)
        # condition true if current_distance == lanelet.distance[-1]
        if index_after is None:
            index_after = len(lanelet.distance) - 1

        index_before = index_after - 1

        # at first iteration with n=0 this means appending the initial points of the lanelet
        current_polygon_vertices_l.append(previous_left_interp)
        current_polygon_vertices_r.append(previous_right_interp)

        # consider control points that may be in-between start and end of polygon
        if previous_index_before != index_before:
            for idx in range(index_before - previous_index_before):
                index_intermediate = previous_index_before + (idx + 1)
                current_polygon_vertices_l.append(left_vertices[index_intermediate])
                current_polygon_vertices_r.append(right_vertices[index_intermediate])

        if n != (int(n_polygons) - 1):
            # interpolate on left and right boundaries
            fraction = (current_distance - lanelet.distance[index_before]) \
                       / (lanelet.distance[index_after] - lanelet.distance[index_before])
            left_points = [left_vertices[index_before], left_vertices[index_after]]
            right_points = [right_vertices[index_before], right_vertices[index_after]]
            left_interp_point = Point(interpolate2d(fraction=fraction, points=left_points))
            right_interp_point = Point(interpolate2d(fraction=fraction, points=right_points))

            # store for next polygon
            previous_left_interp = left_interp_point
            previous_right_interp = right_interp_point

            current_polygon_vertices_l.append(left_interp_point)
            current_polygon_vertices_r.append(right_interp_point)

        # handle last polygon
        if n == (int(n_polygons) - 1):
            current_polygon_vertices_l.append(left_vertices[-1])
            current_polygon_vertices_r.append(right_vertices[-1])

        # create desired structure for Shapely LinearRing
        current_polygon_vertices_r.insert(0, current_polygon_vertices_l[0])
        current_polygon_vertices_r.reverse()

        current_polygon_vertices_l.extend(current_polygon_vertices_r)
        linear_ring = LinearRing(current_polygon_vertices_l)
        current_polygon = Polygon(linear_ring)

        # clear lists for next polygon
        current_polygon_vertices_l.clear()
        current_polygon_vertices_r.clear()

        # append polygon with id
        polygons.append((current_polygon, current_polygon_base_id + counter))

        previous_index_before = index_before
        counter = counter + 1

    return polygons


def reward_1(weight: float):
    return -weight


class ResourceNetwork:
    uncrossable_line_markings = ["solid", "broad_solid"]  # no vehicle can cross this

    def __init__(self, lanelet_network: LaneletNetwork, max_length: Optional[float] = None,
                 excluded_lanelets: Optional[List[int]] = None):
        """
        Create a digraph, here called Resource Network, from a Commonroad lanelet network.
        :param lanelet_network: Commonroad lenelet network
        :param max_length: maximum length of a cell. If None, each lanelet is a cell.
        :param excluded_lanelets: lanelets that should not be added to the graph
        """
        # get lanelet network and create digraph
        # store in STRTree as well (or only in STRTree)
        self.resource_graph: DiGraph = DiGraph()
        self.tree: STRtree
        if excluded_lanelets is None:
            excluded_lanelets = list()
        self.excluded_lanelets = excluded_lanelets
        self._create_rtree(lanelet_network=lanelet_network, max_length=max_length)
        # fixme: merge the graph creation functions
        self._init_resource_graph(lanelet_network=lanelet_network, max_length=max_length)

    def set_default_attributes(self) -> None:
        # set default attributes for nodes
        for current_node in list(self.resource_graph.nodes):
            self.set_node_attribute(attribute='start', value=False, node=current_node)
            self.set_node_attribute(attribute='goal', value=False, node=current_node)
            self.set_node_attribute(attribute='ego_occupied_resource', value=False, node=current_node)
            self.set_node_attribute(attribute='occupied_by_agent', value=False, node=current_node)
            self.set_node_attribute(attribute='goal_of_interest', value=False, node=current_node)

        # set default attributes for edges
        for edge in list(self.resource_graph.edges):
            self.set_edge_attribute(attribute='ego_occupied_edge', value=False, edge=edge)
        return

    def set_node_attribute(self, attribute: str, value: Any, node: int) -> None:
        try:
            self.resource_graph.nodes[node][attribute] = value
        except KeyError:
            print("Specified node does not exist.")

    def set_edge_attribute(self, attribute: str, value: Any, edge: Tuple[int, int]) -> None:
        try:
            self.resource_graph.edges[(edge[0], edge[1])][attribute] = value
        except KeyError:
            print("Specified edge does not exist.")

    def plot_graph(self, file_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(60, 60))
        _, _, _ = self.get_collections_networkx(ax=ax)
        if file_path is not None:
            plt.savefig(file_path)
        plt.close()
        return

    def get_collections_networkx(self, ax: Axes = None) -> Tuple[PathCollection, LineCollection,
                                                                 Mapping[int, plt.Text]]:
        """
        Get collections for plotting a graph on top of a scenario
        :param ax: Axes on which to draw the Artists
        """
        nodes = self.resource_graph.nodes
        edges = self.resource_graph.edges
        cents = []
        for node in nodes.data():
            cents.append(np.asarray(node[-1]['polygon'].centroid))

        centers = dict(zip(nodes.keys(), cents))

        # set default edge and node colors
        edge_colors = ['k'] * len(self.resource_graph.edges)
        node_colors = ['#1f78b4'] * len(self.resource_graph.nodes)

        # set special node and edge colors depending on node and edge attributes
        for node in nodes.data():
            node_idx = list(nodes).index(node[0])
            if node[1]['goal_of_interest']:
                node_colors[node_idx] = 'magenta'
            if node[1]['ego_occupied_resource']:
                node_colors[node_idx] = 'r'
            if node[1]['occupied_by_agent']:
                node_colors[node_idx] = 'cyan'
            if node[1]['goal']:
                node_colors[node_idx] = 'limegreen'
            if node[1]['start']:
                node_colors[node_idx] = 'gold'

        # color edges possibly occupied by ego
        for edge in edges.data():
            edge_idx = list(edges).index((edge[0], edge[1]))
            if edge[2]['ego_occupied_edge']:
                edge_colors[edge_idx] = 'r'

        # return collections and set zorders
        # the functions draw_* already plots on axes
        nodes_plot = draw_networkx_nodes(G=self.resource_graph, ax=ax, pos=centers, node_size=200,
                                         node_color=node_colors)
        nodes_plot.set_zorder(50)

        edges_plot = draw_networkx_edges(G=self.resource_graph, ax=ax, pos=centers, edge_color=edge_colors)
        for edge in range(len(edges_plot)):
            edges_plot[edge].set_zorder(49)

        labels_plot = draw_networkx_labels(G=self.resource_graph, ax=ax, pos=centers)
        for label in labels_plot.keys():
            labels_plot[label].set_zorder(51)

        return nodes_plot, edges_plot, labels_plot

    def get_potential_occupancy(self, start_node: int, end_node: int) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Compute all nodes where an agent could be when transiting from start_node to end_node
        Return both occupied nodes and occupied edges

        :param start_node: departure of agent
        :param end_node: destination of agent
        """
        t1 = time.time()
        paths = all_simple_paths(self.resource_graph, source=start_node, target=end_node)

        occupancy_nodes = set()
        occupancy_edges = set()
        for path in paths:
            # add new nodes only if they are not yet in occupancy_nodes
            occupancy_nodes |= set(path)
            # add new edges only if they are not yet in occupancy_edges
            path_edges = nx.utils.pairwise(path)  # transforms [a,b,c,...] in [(a,b),(b,c),...]
            occupancy_edges |= set(path_edges)
        return occupancy_nodes, occupancy_edges

    def get_occupancy_children(self, occupied_resources: Set[int]) -> List[int]:
        """
        Compute the children of the nodes in the "occupancy zone" as computed by "get_potential_occupancy"
        Children are the first resources on the next lanelet (needs to be validated)
        :param occupied_resources: part of digraph where the ego could be on his journey to the goal
        """
        children = []
        for node in occupied_resources:
            candidates = list(self.resource_graph.successors(node))
            for cand in candidates:
                if cand in occupied_resources:
                    continue
                else:
                    children.append(cand)


        # map each child to the first resource of the next lanelet
        children_shifted = []
        for child in children:
            # case where there are no successors, i.e. the list is empty
            if not list(self.resource_graph.successors(child)):
                children_shifted.append(child)
                continue

            child_base = child//resource_id_factor
            # assuming only one successor
            while list(self.resource_graph.successors(child))[0]//resource_id_factor == child_base:
                child = list(self.resource_graph.successors(child))[0]

            # query successor it once more to get first resource of next lanelet (only if it exists)
            if list(self.resource_graph.successors(child)):
                child = list(self.resource_graph.successors(child))[0]

            children_shifted.append(child)
        return children_shifted

#    def continue_to_next_lanelet(self, nodes: List[int]) -> List[int]:
#        for node in nodes:
#            while self.resource_graph.successors(node)


    def is_upstream(self, node_id: int, nodes: Set[int]) -> bool:
        """
        Determine wether a given node is upstream a set of nodes.
        Returns True even if node_id is in nodes

        :param node_id: node id of node we want to know if is upstream
        :param nodes: set of nodes to check against
        """

        children = nx.traversal.bfs_tree(G=self.resource_graph, source=node_id).nodes
        is_upstream = (set(children) & nodes) != set()

        return is_upstream

    def shortest_path_and_reward(self, start_node: int, end_node: int, reward: Callable) \
            -> Tuple[List[int], float]:  # tbd: what type returned
        """
        Compute all shortest simple paths between two nodes using dijkstra algorithm.
        Weight is considered. If several paths have same (shortest) length, return them all.
        Returns path and reward.

        :param start_node: starting node of path
        :param end_node: ending node of path
        :param reward: function to use to calculate reward of a specific path
        """

        # assumption: is graph is weighted, there will be only one shortest path
        # tbd: check this assumption holds
        paths = nx.all_shortest_paths(G=self.resource_graph, source=start_node, target=end_node,
                                      weight='weight', method='dijkstra')
        rewards = []
        paths_list = list(paths)
        for path in paths_list:
            rewards.append(self.get_path_reward(path=path, reward=reward))
        # fixme: here we enforce above assumption s.t. we can assume asumption holds outside this block
        return paths_list[0], rewards[0]

    def get_path_reward(self, path: List[int], reward: Callable) -> float:
        path_edges = nx.utils.pairwise(path)
        path_reward = 0.0
        for edge in path_edges:
            if self.resource_graph.get_edge_data(edge[0], edge[1]) is None:
                print("None Found. You are probably using a 3D map.")
            path_reward += reward(self.resource_graph.get_edge_data(edge[0], edge[1])['weight'])
        return path_reward

    # fixme: current implementation can not handle 3D structures (e.g. underpass). STRTree has issues!
    def get_resource_by_position(self, position: Union[np.ndarray, Point]) -> int:
        """
        Compute resource (possibly multiple) that contains the queried position by exploiting StrTree.
        :param position: query position. CoM of agent.
        """
        if isinstance(position, np.ndarray):
            position = Point(position)
        resource = self.tree.nearest_item(position)
        # resources = self.tree.query_items(position)
        # resources_geom = self.tree.query(position)
        # test = self.tree.nearest_item(resources_geom[0])
        # candidate_resources = [o for o in self.tree.query(position) if o.intersects(position)]
        # candidate_items = []
        # for res in candidate_resources:
        #    candidate_items.append(self.tree.nearest_item(res))

        if not isinstance(resource, int):
            print("Position: " + str(position) + ". No resource found that contains the position you asked for.")

        return resource

    def _init_resource_graph(self, lanelet_network: LaneletNetwork, max_length: Optional[float]):
        """
        Construct road graph from road network. All lanelets are added, including the lanelet type.
        Lanelets that are in "excluded_lanelets" will be omitted.
        Each lanelet is divided into cells smaller than max_length.
        Edges are constructed between adjacent polygons.
        If a lane between adjacent lanelets is uncrossable, edges are omitted.
        Weight of a polygon is given by its size.
        """

        resources = {}

        # pre-compute and store all resources for all lanelets
        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelets
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            resources[lanelet.lanelet_id] = split_lanelet_into_polygons(
                lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_length)

        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelet
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            # add resources of permissible lanelet to graph
            for current_polygon, polygon_id in resources[lanelet.lanelet_id]:
                self.resource_graph.add_node(polygon_id,
                                             lanelet_type=lanelet.lanelet_type,
                                             polygon=current_polygon)

            # add edges between subsequent resources in a lanelet
            for current_polygon, polygon_idx in resources[lanelet.lanelet_id]:
                # skip last resource of lanelet
                if polygon_idx == resources[lanelet.lanelet_id][-1][1]:
                    continue
                weight = 1.0
                self.resource_graph.add_edge(polygon_idx, polygon_idx + 1, weight=weight)

            # add edge for all succeeding lanelets
            # specifically, connect last resource of a lanelet with first resource of succeeding lanelet
            for id_successor in lanelet.successor:
                # skip excluded lanelet (may be a successor of an allowed lanelet)
                if id_successor in self.excluded_lanelets:
                    continue
                weight = 1.0
                last_resource_lanelet = resources[lanelet.lanelet_id][-1][1]
                first_resource_successor = resources[id_successor][0][1]
                self.resource_graph.add_edge(last_resource_lanelet, first_resource_successor, weight=weight)

            # add edge for adjacent right lanelet (if existing)
            if lanelet.adj_right_same_direction and lanelet.adj_right is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_right in self.excluded_lanelets \
                        or lanelet.line_marking_right_vertices.value \
                        in self.uncrossable_line_markings:
                    continue
                current_resources = resources[lanelet.lanelet_id]
                right_resources = resources[lanelet.adj_right]
                for resource, resource_id in current_resources:
                    for right_resource, right_resource_id in right_resources:
                        if resource.intersects(right_resource):
                            weight = 1.0
                            self.resource_graph.add_edge(resource_id, right_resource_id, weight=weight)

            # add edge for adjacent left lanelet (if existing)
            if lanelet.adj_left_same_direction and lanelet.adj_left is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_left in self.excluded_lanelets \
                        or lanelet.line_marking_left_vertices.value \
                        in self.uncrossable_line_markings:
                    continue
                current_resources = resources[lanelet.lanelet_id]
                left_resources = resources[lanelet.adj_left]
                for resource, resource_id in current_resources:
                    for left_resource, left_resource_id in left_resources:
                        if resource.intersects(left_resource):
                            weight = 1.0
                            self.resource_graph.add_edge(resource_id, left_resource_id, weight=weight)

        self.set_default_attributes()

        file_path = "max_size=1000.jpg"
        self.plot_graph(file_path=file_path)
        return

    def _init_road_graph(self, lanelet_network: LaneletNetwork):
        """
        Construct road graph from road network. All lanelets are added, including the lanelet type.
        Lanelets that are in "excluded_lanelets" will be omitted.
        Edges are constructed between a lanelet and its successor, its right adjacent, and left adjacent.
        If a lane between adjacent lanelets is uncrossable, edge is omitted.
        Length of lane is considered and added as weight.
        """
        # add all nodes
        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelet
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            # add permissible lanelet to graph
            polygon = lanelet_network.find_lanelet_by_id(lanelet.lanelet_id).polygon.shapely_object
            # center = get_lanelet_center_coordinates(lanelet_network, lanelet.lanelet_id)

            self.resource_graph.add_node(lanelet.lanelet_id * resource_id_factor,
                                         lanelet_type=lanelet.lanelet_type,
                                         polygon=polygon)

            # add edge for all succeeding lanelets
            for id_successor in lanelet.successor:
                # skip excluded lanelet (may be a successor of an allowed lanelet)
                if id_successor in self.excluded_lanelets:
                    continue
                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=id_successor)
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             id_successor * resource_id_factor, weight=weight)

            # add edge for adjacent right lanelet (if existing)
            if lanelet.adj_right_same_direction and lanelet.adj_right is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_right in self.excluded_lanelets \
                        or lanelet.line_marking_right_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=lanelet.adj_right)
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             lanelet.adj_right * resource_id_factor, weight=weight)

            # add edge for adjacent left lanelets (if existing)
            if lanelet.adj_left_same_direction and lanelet.adj_left is not None:

                # skip excluded lanelet (may be adj left of an allowed lanelet)
                if lanelet.adj_left in self.excluded_lanelets \
                        or lanelet.line_marking_left_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=lanelet.adj_left)
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             lanelet.adj_left * resource_id_factor, weight=weight)

        self.set_default_attributes()

        return

    def _create_rtree(self, lanelet_network: LaneletNetwork, max_length: Optional[float]):
        resources = []

        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelets
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            if max_length is not None:
                new_resources = split_lanelet_into_polygons(
                    lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_length)
            else:
                new_resources = [(lanelet.polygon.shapely_object, lanelet.lanelet_id * resource_id_factor)]
            resources.extend(new_resources)

        resource_polygons = [resource[0] for resource in resources]
        resource_ids = [resource[1] for resource in resources]
        self.tree = STRtree(geoms=resource_polygons, items=resource_ids)
        return


def zero(a: Any) -> float:
    return 0.0


class DynamicRoadGraph(ResourceNetwork):
    """
        Class to represent dynamic digraphs.
        Extend RoadGraph by adding positions of observed agents and start & end position of the ego
    """

    def __init__(self, lanelet_network: LaneletNetwork,
                 excluded_lanelets: Optional[List[int]] = None, max_length: Optional[float] = None):
        """
        """
        super().__init__(lanelet_network=lanelet_network, excluded_lanelets=excluded_lanelets, max_length=max_length)
        # everything initialized to None because simulation context and planning problem are not known at instantiation
        self.ego_start: Optional[int] = None
        self.ego_goal: Optional[int] = None
        self.planning_problem_updated: bool = False

        self.locations: Dict[PlayerName, List[Tuple[SimTime, int]]] = {}
        self.prediction: Optional[Prediction] = None
        self.graph_storage = []  # just for plotting animations

    def update_predictions(self, sim_obs: SimObservations):
        """ Executed at each time step during simulation. Given the newest observations, the locations of all agents
            are updated, the graph attributes are updated accordingly, the reachability for each agent is computed,
            the rewards for each goal and finally the probabilities for each goal.

            param sim_obs: simulation observations at current time step
        """
        my_name = PlayerName("Ego")

        self.update_locations(sim_obs)
        self.update_dynamic_graph()
        players = list(sim_obs.players.keys())
        players.remove(my_name)
        self.update_reachability(players=players)
        self.compute_rewards()
        self.compute_probabilities()

        return

    def initialize_prediction(self, initial_obs: SimObservations):
        """ Called at start of simulation. Based on initial locations of all agents, a Prediction object is
            instantiated.

            param initial_obs: Simulation observations at t=0
        """
        my_name = PlayerName("Ego")
        players = list(initial_obs.players.keys())
        for player in players:
            self.locations[player] = []
        # update locations and dynamic graph with initial positions of all agents
        self.update_locations(sim_obs=initial_obs)
        self.update_dynamic_graph()

        # Transform Dict to List[List[int]] for Prediction object generation
        # remove myself from reachability calculation
        players.remove(my_name)
        goals = self.get_reachable_goals(players=players)
        goals_list = []
        for player_goals in goals.values():
            goals_list.append(list(player_goals.keys()))

        self.prediction = Prediction(players=players, goals=goals_list)
        return

    def start_and_goal_info(self, problem: PlanningProblem) -> None:
        """
        Get start an goal for ego vehicle from planning problem of scenario.
        Write start and goal in self.ego_start and self.ego_goal, respectively.
        Update of goal and start during simulation is supported

        :param problem: Planning Problem as defined by Commonroad
        """

        # fixme: at first iteration use get_resource_by_position.
        # get_resource_by_position does not work because it needs an initial id. Generalize this.

        start = self.get_resource_by_position(position=problem.initial_state.position)
        self.ego_start = start
        goal = self.get_resource_by_position(position=problem.goal.state_list[0].position.center)
        self.ego_goal = goal
        print("Start of planning problem: ")
        print(self.ego_start)
        print("Goal of planning problem: ")
        print(self.ego_goal)

        # there is a new planning problem
        self.planning_problem_updated = True
        return

    def get_lanelet_by_position_restricted(self, player: PlayerName, position: np.ndarray) -> int:
        """
        Version get_resource_by_position in resource_graph with restricted search.
        Position is used to find lanelet id only by querying on current lanelet
        polygon and the subsequent lanelet polygons.
        :param position: query position
        """
        previous_id = self.locations[player][-1][1]
        # check if position is in previous polygon
        if self.resource_graph.nodes[previous_id]['polygon'].contains_point(position):
            return previous_id

        # check if position is in subsequent polygons
        next_ids = self.resource_graph.neighbors(previous_id)
        for next_id in list(next_ids):
            if self.resource_graph.nodes[next_id]['polygon'].contains_point(position):
                return next_id

        print("Position: " + str(position) + ". No lanelet found that contains"
                                             " the position you asked for by using restricted algorithm.")

    def keep_track(self):
        self.graph_storage.append(deepcopy(self.resource_graph))

    def instantiate_prediction_object(self, players: List[PlayerName]):
        self.prediction = Prediction(self.get_reachable_goals(players=players))
        return

    # warning: this can only handle maps completely in 2D. If there are overlapping lanelets, this algorithm
    # can't distinguish between different heights.
    def update_locations(self, sim_obs: SimObservations):
        t = sim_obs.time
        for player, player_obs in sim_obs.players.items():
            player_pos = np.array([player_obs.state.x, player_obs.state.y])
            resource_id = self.get_resource_by_position(position=player_pos)
            self.locations[player].append((t, resource_id))
        return

    def get_past_path(self, player: PlayerName) -> List[int]:
        """
        Get past history. Skip repeating nodes.

        :param player: query player
        """
        past_path = []
        previous_id = None
        for t, current_id in self.locations[player]:
            if previous_id == current_id:
                continue
            past_path.append(current_id)
            previous_id = current_id
        return past_path

    # tbd: check this works properly
    def check_reachability(self, nodes: List[int], player: PlayerName) -> List[int]:
        """
        Compute which nodes are reachable by player from a given list of nodes.

        :param nodes: query nodes
        :param player: player for which to compute reachable goals
        """
        # latest position of player
        current_node = self.get_player_location(player)
        reachable_nodes = []

        for node in nodes:
            if self.is_upstream(node_id=current_node, nodes={node}):
                reachable_nodes.append(node)

        return reachable_nodes

    # tbd: check this works properly
    def get_reachable_goals(self, players: List[PlayerName]) -> PlayerGoalBools:
        """
        Compute reachable goals for each player and store in a dictionary.
        :param players: list of players for which to compute goals
        """

        # determine goals of interest just outside Ego occupancy
        relevant_goals = []
        for node_id, node_data in self.resource_graph.nodes(data=True):
            if node_data['goal_of_interest']:
                relevant_goals.append(node_id)

        # check which of these goals are reachable
        goals = []
        for player in players:
            reachable_goals = self.check_reachability(nodes=relevant_goals, player=player)
            goals.append(reachable_goals)

        return PlayerGoalBools().from_lists(players=players, goals=goals, init_value=True)

    # tbd: check this works properly
    def update_reachability(self, players: List[PlayerName]):
        reachability = self.get_reachable_goals(players=players)
        if self.prediction.reachability != reachability:
            print("The reachability just changed. Old reachability dictionary: " + str(self.prediction.reachability))
            print(" New reachability dictionary: " + str(reachability))
            self.prediction.reachability = self.prediction.reachability.boolean_intersect(reachability)

    def compute_rewards(self) -> None:
        """
        Compute past rewards from t=0 to now, optimal path from now to goal and associated reward,
        optimal paths from initial position to goal and associated reward. Computation done for all players and all
        reachable goals.
        """

        for player, player_goals in self.prediction.reachability.items():
            # loop over goals for each player
            for goal, goal_reachability in player_goals.items():

                # only consider reachable goals
                if goal_reachability:
                    loc = self.get_player_location(player)
                    # compute optimal path and associated reward from current position to final goal
                    partial_shortest_path, partial_reward = self.shortest_path_and_reward(start_node=loc,
                                                                                          end_node=goal,
                                                                                          reward=reward_1)
                    # compute past path and associated reward (from t=0 to now)
                    past_path = self.get_past_path(player=player)
                    past_reward = self.get_path_reward(path=past_path, reward=reward_1)

                    # fixme: this could be calculated only at beginning one to optimize code
                    # compute optimal path and associated reward from initial position to goal
                    optimal_path, optimal_reward = self.shortest_path_and_reward(self.locations[player][0][1],
                                                                                 end_node=goal, reward=reward_1)

                    self.prediction.suboptimal_reward[player][goal] = partial_reward + past_reward
                    self.prediction.optimal_reward[player][goal] = optimal_reward

                # goal is not reachable anymore. Set reward to approx. -Inf
                else:
                    self.prediction.suboptimal_reward[player][goal] = -99999999999.0

    def compute_probabilities(self) -> None:
        """
        Compute probability of each goal for each agent.
        """

        # prob_dict is initially filled with 0.0
        self.prediction.probabilities = self.prediction.probabilities.valfun(func=zero)

        # compute difference between suboptimal and optimal reward
        self.prediction.probabilities = self.prediction.probabilities + self.prediction.suboptimal_reward
        self.prediction.probabilities = self.prediction.probabilities - self.prediction.optimal_reward

        # scale with rationality factor and raise to exponential
        self.prediction.probabilities = self.prediction.probabilities.scalar_multiplication(self.prediction.params.beta)
        self.prediction.probabilities = self.prediction.probabilities.valfun(func=np.exp)

        # multiply with prior
        self.prediction.probabilities = self.prediction.probabilities * self.prediction.params.priors
        self.prediction.probabilities = self.prediction.probabilities.normalize()

        return

    def get_player_location(self, player: PlayerName):
        return self.locations[player][-1][1]

    def update_dynamic_graph(self) -> None:
        """
        Update node attributes depending on ego_start, ego_goal and locations of all agents
        """

        if self.planning_problem_updated:
            # reset default values of node and edge attributes
            self.set_default_attributes()

            # compute new edge attributes
            self.set_node_attribute(attribute='start', value=True, node=self.ego_start)
            self.set_node_attribute(attribute='goal', value=True, node=self.ego_goal)

            ego_resources_nodes, ego_resources_edges = \
                self.get_potential_occupancy(start_node=self.ego_start, end_node=self.ego_goal)
            goals_of_interest = self.get_occupancy_children(ego_resources_nodes)

            for node in ego_resources_nodes:
                self.set_node_attribute(attribute='ego_occupied_resource', value=True, node=node)
            for edge in ego_resources_edges:
                self.set_edge_attribute(attribute='ego_occupied_edge', value=True, edge=edge)
            for node in goals_of_interest:
                self.set_node_attribute(attribute='goal_of_interest', value=True, node=node)

        for player, locations in self.locations.items():
            if player != 'Ego':
                # skip first loop iteration
                if len(locations) > 1:
                    # remove previous location
                    self.set_node_attribute(attribute='occupied_by_agent', value=False, node=locations[-2][1])
                    # set new location
                self.set_node_attribute(attribute='occupied_by_agent', value=player, node=locations[-1][1])

        # updated plan has been processed
        self.planning_problem_updated = False

        return


if __name__ == '__main__':
    # scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
    # scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
    scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24
    scenario_path4 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Intersection-1_1_T-1_nando.xml"

    scenario, planning_problem_set = CommonRoadFileReader(scenario_path3).open(lanelet_assignment=True)
    net = scenario.lanelet_network

    obj = ResourceNetwork(lanelet_network=net, max_length=1000.0)

    # Plot graph and scenario

    fig, axs = plt.subplots(2, figsize=(50, 50))

    nodes = obj.resource_graph.nodes
    cents = []
    for node in nodes.data():
        # cent_point = node[-1]['polygon']
        cent_point = node[-1]['polygon'].centroid
        cents.append([cent_point.x, cent_point.y])

    cents = dict(zip(nodes.keys(), cents))

    # centers = dict(zip(nodes.keys(), cents))
    nodes_plot = draw_networkx_nodes(G=obj.resource_graph, ax=axs[0], pos=cents, node_size=50)
    edges_plot = draw_networkx_edges(G=obj.resource_graph, ax=axs[0], pos=cents)
    # plt.savefig("graph_debug.png")
    # plt.close()

    # plt.subplots()
    rnd = MPRenderer(ax=axs[1], )
    scenario.draw(rnd)
    # planning_problem_set.draw(rnd)
    rnd.render()
    plt.savefig("scenario_testing_tmpcode_10122021_size1000.png")
    plt.close()

    test_lanelet = 3318
    test_lanelet_adj_right = 3316
    test_lanelet_adj_left = 3320
    new_polygons = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet), max_length=10.0)
    new_polygons_right = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_right),
                                                         max_length=10.0)
    new_polygons_left = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_left), max_length=10.0)

    old_polygon = net.find_lanelet_by_id(test_lanelet).polygon.shapely_object
    old_polygon_right = net.find_lanelet_by_id(test_lanelet_adj_right).polygon.shapely_object
    old_polygon_left = net.find_lanelet_by_id(test_lanelet_adj_left).polygon.shapely_object

    plt.plot()
    x, y = old_polygon.exterior.xy
    plt.plot(x[:], y[:], 'k')
    x, y = old_polygon_right.exterior.xy
    plt.plot(x[:], y[:], 'k')
    x, y = old_polygon_left.exterior.xy
    plt.plot(x[:], y[:], 'k')
    for i, polygon in enumerate(new_polygons_left):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')
    for i, polygon in enumerate(new_polygons):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')
    for i, polygon in enumerate(new_polygons_right):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')

    plt.savefig("debugging_21-12-2021.png")
    plt.close()
