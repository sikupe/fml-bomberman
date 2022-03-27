"""Module for the Genealogy."""

from dataclasses import dataclass
import graphviz
from colour import Color
from typing import List, Optional
import numpy as np


@dataclass
class Parents:
    """Parent datastructure."""
    left: Optional[int] = None
    right: Optional[int] = None


@dataclass
class GenealogyNode:
    """One node in the genealogy."""
    parents: Optional[Parents] = None
    child: Optional[int] = None
    winner: int = 0


class Genealogy:
    """Generate a genealogy of the evolution process."""
    def __init__(self, mu: int):
        self.count: int = mu
        self.data: List[GenealogyNode] = list()
        self.children: List[int] = list()
        self.parents: np.ndarray = np.arange(mu, dtype=int)

        for i in range(mu):
            self.data.append(GenealogyNode(Parents(None, None), i))

    def process_winners(self, winner_indicies: List[int], iteration: int):
        """
        Use a list of winner indicies to set according winner attributes for
        the current iteration.
        """
        np_all = np.append(self.parents, np.array(self.children, dtype=np.int32))
        self.parents = np_all[winner_indicies]
        # Mark winners
        for i in self.parents:
            self.data[i].winner = iteration
        self.children.clear()

    def add_child(self, parent_1_index: int, parent_2_index: int):
        """Add a new child."""
        self.data.append(
            GenealogyNode(
                Parents(self.parents[parent_1_index], self.parents[parent_2_index]),
                self.count,
            )
        )
        self.children.append(self.count)
        self.count += 1


    def generate_dot(
        self, mu: int, lambda_param: int, iterations: int
    ):
        """
        Use the genealogy to generate a graphical represenation via graphviz
        dot.
        """
        dot = graphviz.Digraph(comment="Genealogy")
        dot.attr(ranksep="1.0")
        dot.attr("node", odering="out")

        colors = list(Color("red").range_to(Color("green"), iterations + 1))

        # First get the inital parents
        with dot.subgraph() as s:
            for genealogy_node in self.data[:mu]:
                i = genealogy_node.winner
                s.attr(rank="same")
                s.node(
                    str(genealogy_node.child),
                    **{"style": "filled", "fillcolor": colors[i].get_web()}
                    if i != 0
                    else {},
                )

        for i in range(iterations):
            with dot.subgraph() as s:
                for genealogy_node in self.data[
                                      (mu + i * lambda_param): (mu + (i + 1) * lambda_param)
                                      ]:
                    i = genealogy_node.winner
                    s.attr(rank="same")
                    s.node(
                        str(genealogy_node.child),
                        **{"style": "filled", "fillcolor": colors[i].get_web()}
                        if i != 0
                        else {},
                    )
                    assert genealogy_node.parents
                    left = str(genealogy_node.parents.left)
                    right = str(genealogy_node.parents.right)
                    child = str(genealogy_node.child)
                    dot.edges([(left, child), (right, child)])

        with open("genealogy.svg", "wb+") as dot_file:
            dot_file.write(dot.pipe(format="svg"))
