from abc import ABC
from typing import List

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

    def insert(self, node: AnyNode):
        self.tree[node.id] = node

    def invalid_childs(self, parent_node: AnyNode):
        if parent_node.children:
            for child in parent_node.children:
                self.tree[child.id].valid = False


    def remove(self):
        for key in self.tree:
            if not self.tree[key].valid:
                self.tree.pop(key)

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
        while parent.parent is not None:
            self.tree[parent.id].children.append(child)
            parent = parent.parent
        self.tree[parent.id].children.append(child)

    def set_invalid_node(self, node) -> None:
        self.tree[node.id].valid = False

    def remove_nodes(self, nodes: List[AnyNode]):
        for n in nodes:
            self.tree.pop(n.id)

    def get_new_trimmed_path(self, path: List[SE2Transform], pose: SE2Transform) -> List[SE2Transform]:
        d = [(p.p[0] - pose.p[0]) ** 2 + (p.p[1] - pose.p[1]) ** 2 for p in path]
        minind = d.index(min(d))
        new_path = path[minind:]
        new_path[0] = pose

        return new_path

    def set_new_root_node(self, node: AnyNode, pose: SE2Transform, path_idx: int):
        # passed_nodes = []
        new_root_node = AnyNode(pose=pose, id='0')
        # parent_node = node.parent
        # while parent_node is not None:
        #     for child in parent_node.children:
        #         if child.id != node.id:
        #             passed_nodes.append(child)
        #     passed_nodes.append(parent_node)
        #     parent_node = parent_node.parent
        # self.remove_nodes(passed_nodes)
        # new_path = self.get_new_trimmed_path(node.path, pose)
        new_path = node.path[path_idx:]
        new_path[0] = pose
        node.path = new_path
        node.parent = new_root_node
        self.tree[node.id] = node
        root_children = node.children
        root_children.append(node)
        new_root_node.children = root_children
        self.root_node = new_root_node
        self.tree = {}
        self.insert(self.root_node)
        for c in root_children:
            self.insert(c)


# @dataclass()
# class Node(ABC):
#     pose: SE2Transform
#     path: List[SE2Transform]
#
#
# @dataclass()
# class RRTNode(Node):
#     parent: Node
