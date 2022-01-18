from abc import ABC
from typing import List

import numpy as np
from scipy.spatial import KDTree, cKDTree

from dg_commons import SE2Transform


class Node(ABC):
    def __init__(self, pose: SE2Transform):
        self.pose = pose
        self.path: List[SE2Transform] = [pose]
        self.parent = None


class StarNode(Node):
    def __init__(self, pose: SE2Transform):
        super().__init__(pose=pose)
        self.cost = 0.0


class AnyNode(StarNode):
    def __init__(self, pose: SE2Transform, id: str):
        super().__init__(pose=pose)
        self.valid = True
        self.id = id
        self.children = []
        self.first_children = []


class Path:
    def __init__(self, path: List[SE2Transform], nodes: List[Node]):
        self.path = path
        self.nodes = nodes


class Tree:
    def __init__(self, root_node: AnyNode):
        self.root_node = root_node
        self.tree = {root_node.id: root_node}
        self.point_list = [root_node.pose.p]
        self.node_index_list = [root_node.id]

    def insert(self, node: AnyNode):
        self.tree[node.id] = node
        try:
            self.point_list[self.node_index_list.index(node.id)] = node.pose.p
        except:
            self.point_list.append(node.pose.p)
            self.node_index_list.append(node.id)

    def invalid_childs(self, parent_node: AnyNode):
        if parent_node.children:
            for child in parent_node.children:
                if child.id in self.tree:
                    self.tree[child.id].valid = False

    def remove(self):
        invalid_nodes = []
        for val in self.tree.values():
            if not val.valid:
                invalid_nodes.append(val)
        self.remove_nodes(invalid_nodes)

    def find_best_path(self, last_node: AnyNode) -> Path:
        path = []
        nodes = []
        node = self.tree[last_node.id]
        while node.parent is not None:
            nodes.append(node)
            for p in reversed(node.path):
                path.append(p)
            node = node.parent
        nodes.append(node)
        path.append(node.pose)
        final_path = Path(path=list(reversed(path)), nodes=list(reversed(nodes)))

        return final_path

    def last_id_node(self) -> int:
        keys = self.tree.keys()
        last = max([int(k) for k in keys])

        return last

    def last_node(self) -> AnyNode:
        return self.tree.get(self.last_id_node())

    def set_child(self, parent: AnyNode, child: AnyNode) -> None:
        self.tree[parent.id].first_children.append(child)
        set_parent = self.tree[parent.id]
        while set_parent is not None:
            try:
                child_idx = self.tree[set_parent.id].children.index(child)
                print("child already set")
            except:
                self.tree[set_parent.id].children.append(child)
            set_parent = set_parent.parent

    def set_invalid_node(self, node) -> None:
        self.tree[node.id].valid = False

    def remove_nodes(self, nodes: List[AnyNode]):
        for n in nodes:
            children = n.children.copy()
            children.append(n)
            if n.parent.id in self.node_index_list:
                first_child_idx = n.parent.first_children.index(n)
                self.tree[n.parent.id].first_children.pop(first_child_idx)
                self.remove_children(parent=n.parent, children=children)
            self.tree.pop(n.id)
            idx = self.node_index_list.index(n.id)
            self.node_index_list.pop(idx)
            self.point_list.pop(idx)

    def get_new_trimmed_path(self, path: List[SE2Transform], pose: SE2Transform) -> List[SE2Transform]:
        d = [(p.p[0] - pose.p[0]) ** 2 + (p.p[1] - pose.p[1]) ** 2 for p in path]
        minind = d.index(min(d))
        new_path = path[minind:]
        new_path[0] = pose

        return new_path

    def set_new_root_node(self, node: AnyNode, pose: SE2Transform, path_idx: int):
        new_root_node = AnyNode(pose=pose, id='0')
        new_path = node.path[path_idx:]
        new_path[0] = pose
        node.path = new_path
        node.parent = new_root_node
        self.tree[node.id] = node
        root_children = node.children.copy()
        new_root_node.children = root_children
        new_root_node.children.append(node)
        new_root_node.first_children = [node]
        self.root_node = new_root_node
        self.tree = {}
        self.point_list = []
        self.node_index_list = []
        self.insert(self.root_node)
        for c in root_children:
            self.insert(c)

    def get_nearest_node_from_tree(self, point: np.array) -> AnyNode:
        tree = KDTree(data=self.point_list, leafsize=10, compact_nodes=True)
        dd, ii = tree.query(point, k=1, workers=4)
        node_id = self.node_index_list[ii]

        return self.tree[node_id]

    def rewire(self, node_id: str, new_node: AnyNode):
        old_cost = self.tree[node_id].cost
        old_children = self.tree[node_id].children.copy()
        old_children.append(self.tree[node_id])
        old_parent = self.tree[node_id].parent
        self.remove_children(old_parent, old_children)
        if self.tree[old_parent.id].first_children:
            child_idx = old_parent.first_children.index(self.tree[node_id])
            self.tree[old_parent.id].first_children.pop(child_idx)
        self.tree[node_id].pose = new_node.pose
        self.tree[node_id].cost = new_node.cost
        self.tree[node_id].path = new_node.path
        self.tree[node_id].parent = new_node.parent
        self.set_child(parent=new_node.parent, child=self.tree[node_id])
        if self.tree[node_id].children:
            self.add_children(parent=new_node.parent, children=self.tree[node_id].children)
        self.propagate_cost_to_leaves(new_node, old_cost=old_cost)

    def remove_children(self, parent: AnyNode, children: List[AnyNode]):
        check_parent = self.tree[parent.id]
        while check_parent is not None:
            for child in children:
                try:
                    # parent_children = self.tree[check_parent.id].children
                    # child_idx = parent_children.index(child)
                    # self.tree[check_parent.id].children.pop(child_idx)
                    child_idx = self.tree[check_parent.id].children.index(child)
                    self.tree[check_parent.id].children.pop(child_idx)
                except:
                    continue
                # child_idx = parent_children.index(child)
                # self.tree[check_parent.id].children.pop(child_idx)
                # child_idx = self.tree[check_parent.id].children.index(child)
                # self.tree[check_parent.id].children.pop(child_idx)
            check_parent = check_parent.parent

    def add_children(self, parent: AnyNode, children: List[AnyNode]):
        check_parent = self.tree[parent.id]
        while check_parent is not None:
            for child in children:
                self.tree[check_parent.id].children.append(child)
            check_parent = check_parent.parent

    def propagate_cost_to_leaves(self, parent_node: AnyNode, old_cost: float) -> None:
        for node in parent_node.children:
            cost = node.cost - old_cost + parent_node.cost
            self.tree[node.id].cost = cost

