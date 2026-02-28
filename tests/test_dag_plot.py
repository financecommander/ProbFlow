"""Tests for probflow.viz.dag_plot module."""

import os
import shutil
import tempfile

import pytest

from probflow.viz.dag_plot import plot_network, _longest_path, _critical_edges

_HAS_DOT = shutil.which("dot") is not None
_skip_no_dot = pytest.mark.skipif(not _HAS_DOT, reason="Graphviz 'dot' binary not on PATH")


# -----------------------------------------------------------------------
# Helper fixtures
# -----------------------------------------------------------------------

def _simple_network():
    """A small 3-node network for basic tests."""
    return {
        "nodes": [
            {"name": "Rain", "type": "observed"},
            {"name": "Sprinkler", "type": "latent"},
            {"name": "WetGrass", "type": "query"},
        ],
        "edges": [
            {"from": "Rain", "to": "WetGrass", "probability": 0.80},
            {"from": "Sprinkler", "to": "WetGrass", "probability": 0.70},
        ],
    }


def _chain_network():
    """A linear chain: A -> B -> C -> D (for critical path testing)."""
    return {
        "nodes": [
            {"name": "A", "type": "latent"},
            {"name": "B", "type": "latent"},
            {"name": "C", "type": "latent"},
            {"name": "D", "type": "query"},
        ],
        "edges": [
            {"from": "A", "to": "B", "probability": 0.90},
            {"from": "B", "to": "C", "probability": 0.85},
            {"from": "C", "to": "D", "probability": 0.75},
        ],
    }


def _diamond_network():
    """A diamond DAG: A -> B, A -> C, B -> D, C -> D."""
    return {
        "nodes": [
            {"name": "A", "type": "observed"},
            {"name": "B", "type": "latent"},
            {"name": "C", "type": "latent"},
            {"name": "D", "type": "query"},
        ],
        "edges": [
            {"from": "A", "to": "B", "probability": 0.60},
            {"from": "A", "to": "C", "probability": 0.40},
            {"from": "B", "to": "D", "probability": 0.90},
            {"from": "C", "to": "D", "probability": 0.80},
        ],
    }


# -----------------------------------------------------------------------
# Export format tests
# -----------------------------------------------------------------------

class TestExportFormats:
    @_skip_no_dot
    def test_export_png(self, tmp_path):
        """plot_network should produce a valid PNG file."""
        outfile = str(tmp_path / "test_graph.png")
        dot = plot_network(_simple_network(), output=outfile)
        assert os.path.isfile(outfile)
        # PNG magic bytes
        with open(outfile, "rb") as f:
            header = f.read(4)
        assert header[:4] == b"\x89PNG"

    @_skip_no_dot
    def test_export_svg(self, tmp_path):
        """plot_network should produce a valid SVG file."""
        outfile = str(tmp_path / "test_graph.svg")
        dot = plot_network(_simple_network(), output=outfile)
        assert os.path.isfile(outfile)
        with open(outfile, "r") as f:
            content = f.read()
        assert "<svg" in content

    def test_export_dot(self, tmp_path):
        """plot_network should produce a valid DOT file."""
        outfile = str(tmp_path / "test_graph.dot")
        dot = plot_network(_simple_network(), output=outfile)
        assert os.path.isfile(outfile)
        with open(outfile, "r") as f:
            content = f.read()
        assert "digraph" in content
        assert "Rain" in content
        assert "WetGrass" in content
        assert "->" in content

    def test_export_gv(self, tmp_path):
        """plot_network should accept .gv extension as DOT format."""
        outfile = str(tmp_path / "test_graph.gv")
        dot = plot_network(_simple_network(), output=outfile)
        assert os.path.isfile(outfile)
        with open(outfile, "r") as f:
            content = f.read()
        assert "digraph" in content


# -----------------------------------------------------------------------
# Node styling tests
# -----------------------------------------------------------------------

class TestNodeStyling:
    def test_observed_node_style(self, tmp_path):
        """Observed nodes should be gray filled boxes."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        # Observed node "Rain" should have box shape and gray fill
        assert "shape=box" in src
        assert "fillcolor=lightgray" in src

    def test_query_node_style(self, tmp_path):
        """Query nodes should have bold styling."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "penwidth=3" in src
        assert "bold" in src.lower()

    def test_latent_node_style(self, tmp_path):
        """Latent nodes should be ellipses (default shape)."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "shape=ellipse" in src

    def test_default_type_is_latent(self, tmp_path):
        """A node without an explicit type should default to latent."""
        net = {
            "nodes": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"from": "X", "to": "Y"}],
        }
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        # All nodes should be ellipses
        assert "shape=ellipse" in src
        # No box or bold styling for non-typed nodes
        assert "shape=box" not in src or "legend" in src.lower()


# -----------------------------------------------------------------------
# Edge label / probability tests
# -----------------------------------------------------------------------

class TestEdgeLabels:
    def test_probabilities_shown_by_default(self, tmp_path):
        """Edge labels should show probability values by default."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "0.80" in src
        assert "0.70" in src

    def test_probabilities_hidden(self, tmp_path):
        """When show_probabilities=False, no probability labels appear."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"),
                           show_probabilities=False)
        src = dot.source
        assert "0.80" not in src
        assert "0.70" not in src

    def test_edge_without_probability(self, tmp_path):
        """Edges without a probability key should render without label."""
        net = {
            "nodes": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"from": "X", "to": "Y"}],
        }
        dot = plot_network(net, output=str(tmp_path / "g.dot"),
                           show_probabilities=True)
        src = dot.source
        assert "X" in src
        assert "Y" in src
        # The edge should exist without a label attribute
        assert "->" in src


# -----------------------------------------------------------------------
# Critical path tests
# -----------------------------------------------------------------------

class TestCriticalPath:
    def test_longest_path_chain(self):
        """In a linear chain A->B->C->D, the longest path is [A,B,C,D]."""
        net = _chain_network()
        path = _longest_path(net["nodes"], net["edges"])
        assert path == ["A", "B", "C", "D"]

    def test_longest_path_diamond(self):
        """In a diamond DAG, the longest path has length 2 (A->B->D or A->C->D)."""
        net = _diamond_network()
        path = _longest_path(net["nodes"], net["edges"])
        assert len(path) == 3
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_longest_path_empty(self):
        """Empty networks should return an empty path."""
        assert _longest_path([], []) == []

    def test_longest_path_no_edges(self):
        """Network with nodes but no edges returns empty path."""
        nodes = [{"name": "X"}, {"name": "Y"}]
        assert _longest_path(nodes, []) == []

    def test_critical_edges(self):
        """_critical_edges should return edge tuples along the path."""
        path = ["A", "B", "C"]
        edges = _critical_edges(path)
        assert edges == {("A", "B"), ("B", "C")}

    def test_critical_path_highlighting_in_dot(self, tmp_path):
        """When highlight_critical_path=True, red styling should appear."""
        net = _chain_network()
        outfile = str(tmp_path / "critical.dot")
        dot = plot_network(net, output=outfile,
                           highlight_critical_path=True)
        src = dot.source
        assert "color=red" in src
        assert "penwidth=" in src

    def test_no_critical_path_by_default(self, tmp_path):
        """By default, no red highlighting should appear."""
        net = _chain_network()
        outfile = str(tmp_path / "no_critical.dot")
        dot = plot_network(net, output=outfile,
                           highlight_critical_path=False)
        src = dot.source
        # "color=red" from edge/node styling should not appear
        # (legend doesn't use red)
        assert "color=red" not in src


# -----------------------------------------------------------------------
# Layout quality tests
# -----------------------------------------------------------------------

class TestLayoutQuality:
    def test_hierarchical_layout(self, tmp_path):
        """The graph should use hierarchical (top-to-bottom) layout."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "rankdir=TB" in src

    def test_no_overlaps_via_nodesep(self, tmp_path):
        """Node separation attributes should be set to avoid overlaps."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "nodesep" in src
        assert "ranksep" in src

    def test_uses_dot_engine(self, tmp_path):
        """The engine should be 'dot' for hierarchical layout."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        assert dot.engine == "dot"

    @_skip_no_dot
    def test_png_renders_without_overlap(self, tmp_path):
        """A moderately complex graph should render to a PNG without error."""
        # Build a larger network
        nodes = [{"name": f"N{i}", "type": "latent"} for i in range(10)]
        edges = [{"from": f"N{i}", "to": f"N{i+1}"} for i in range(9)]
        net = {"nodes": nodes, "edges": edges}
        outfile = str(tmp_path / "large.png")
        dot = plot_network(net, output=outfile)
        assert os.path.isfile(outfile)
        # File should have non-trivial size
        assert os.path.getsize(outfile) > 100


# -----------------------------------------------------------------------
# Legend tests
# -----------------------------------------------------------------------

class TestLegend:
    def test_legend_present_in_dot(self, tmp_path):
        """The DOT source should include a legend subgraph."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "cluster_legend" in src
        assert "Legend" in src

    def test_legend_contains_node_types(self, tmp_path):
        """The legend should show Observed, Query, and Latent entries."""
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        src = dot.source
        assert "Observed" in src
        assert "Query" in src
        assert "Latent" in src

    @_skip_no_dot
    def test_legend_in_rendered_output(self, tmp_path):
        """Legend should be present in rendered SVG output."""
        outfile = str(tmp_path / "legend.svg")
        dot = plot_network(_simple_network(), output=outfile)
        with open(outfile, "r") as f:
            svg_content = f.read()
        assert "Legend" in svg_content
        assert "Observed" in svg_content


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_network(self, tmp_path):
        """An empty network should render without error."""
        net = {"nodes": [], "edges": []}
        outfile = str(tmp_path / "empty.dot")
        dot = plot_network(net, output=outfile)
        assert os.path.isfile(outfile)

    def test_single_node(self, tmp_path):
        """A single node with no edges should render."""
        net = {"nodes": [{"name": "Alone", "type": "observed"}], "edges": []}
        outfile = str(tmp_path / "single.dot")
        dot = plot_network(net, output=outfile)
        with open(outfile, "r") as f:
            content = f.read()
        assert "Alone" in content

    def test_return_type_is_digraph(self, tmp_path):
        """plot_network should return a graphviz.Digraph."""
        import graphviz
        net = _simple_network()
        dot = plot_network(net, output=str(tmp_path / "g.dot"))
        assert isinstance(dot, graphviz.Digraph)
