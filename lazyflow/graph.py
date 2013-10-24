"""
This module implements the basic flow graph
of the lazyflow module.

Basic usage example:

---
import numpy
import lazyflow.graph
from lazyflow.operators.operators import  OpArrayPiper


g = lazyflow.graph.Graph()

operator1 = OpArrayPiper(graph=g)
operator2 = OpArrayPiper(graph=g)

operator1.inputs["Input"].setValue(numpy.zeros((10,20,30), dtype=numpy.uint8))

operator2.inputs["Input"].connect(operator1.outputs["Output"])

result = operator2.outputs["Output"][:].wait()
---

"""

#Python
import sys
import copy
import functools
import collections
import itertools
import threading
import logging

#third-party
import psutil
if (int(psutil.__version__.split(".")[0]) < 1 and
    int(psutil.__version__.split(".")[1]) < 3):
    print ("Lazyflow: Please install a psutil python module version"
           " of at least >= 0.3.0")
    sys.exit(1)

#SciPy
import numpy

#lazyflow
from lazyflow import rtype
from lazyflow.request import Request
from lazyflow.stype import ArrayLike
from lazyflow.utility import slicingtools, Tracer, OrderedSignal, Singleton
from lazyflow.slot import InputSlot, OutputSlot, Slot
from lazyflow.operator import Operator, InputDict, OutputDict, OperatorMetaClass
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.metaDict import MetaDict

class Graph(object):
    """
    A Graph instance is shared by all connected operators and contains any 
    bookkeeping or globally accessible state needed by all operators/slots in the graph.
    """
    def __init__(self):
        self._setup_depth = 0
        self._sig_setup_complete = None
        self._lock = threading.Lock()
        
        self.ops_to_config = set()

    def call_when_setup_finished(self, fn):
        # The graph is considered in "setup" mode if any slot is executing a function that affects the state of the graph.
        # See slot.py for details.  Such operations typically invoke a chain reaction of setup operations.
        # The entire setup is "finished" when the initially invoked setup function returns.
        """
        See comment above.
        
        If the graph is not in the middle of a setup operation as described above,
        immediately call the given callback.  Otherwise, save the callback and 
        execute it when the setup operation completes.  The callback is executed 
        only once, and then it is discarded.
        """
        if self._setup_depth == 0:
            # Not setting up.  Call immediately
            fn()
        else:
            # Subscribe to the next completion.
            self._sig_setup_complete.subscribe(fn)
    
    class SetupDepthContext(object):
        """
        A context manager to manage the "depth" of a setup operation.
        When the depth reaches zero, the graph's `_sig_setup_complete` signal is emitted.
        """
        def __init__(self, g, slot):
            self._graph = g
            self._slot = slot
            
        def __enter__(self):
            if self._graph:
                with self._graph._lock:
                    if self._graph._setup_depth == 0:
                        print "Starting new setup"
                        self._graph._sig_setup_complete = OrderedSignal()
                    self._graph._setup_depth += 1

        def __exit__(self, *args):
            if self._graph:
                if self._graph._setup_depth == 1:
                    print "Close to finishing..."
                    # The original setup_func that we triggered is complete.
                    # Now we have to configure all the operators that were affected by it,
                    #  and then we have to keep looping until all operators are configured.
                    start_op = self._slot.getRealOperator()
                    while self._graph.ops_to_config:
                        # Recompute setup-order DAG.
                        # Unfortunately, we have to recompute the dag after every iteration because configuring an operator can cause changes to the graph.
                        dag = self._generate_setup_dag( start_op )
                        sorted_nodes = nx.topological_sort(dag)
                        for index, node in enumerate(sorted_nodes):
                            if node.type == 'output' and node.op in self._graph.ops_to_config:
                                self._graph.ops_to_config.remove( node.op )
                                if node.op.configured():
                                    print "Setting up {}".format( node.op.name )
                                    node.op._setupOutputs()
                                    print "Finished setup"
                                    #start_op = node.op
                                    sorted_nodes = sorted_nodes[index+1:]
                                    break
                
                    print "Finished all setup."
                sig_setup_complete = None
                with self._graph._lock:
                    self._graph._setup_depth -= 1
                    if self._graph._setup_depth == 0:
                        sig_setup_complete = self._graph._sig_setup_complete
                        # Reset.
                        self._graph._sig_setup_complete = None
                if sig_setup_complete:
                    sig_setup_complete()

        def _generate_setup_dag(self, start_op):
            builder = SetupDigraphBuilder()
            builder.add_op( start_op )
            dag = builder.digraph
            #assert nx.is_directed_acyclic_graph(dag)
            return dag

import networkx as nx
class SetupDigraphBuilder(object):
    Node = collections.namedtuple("Node", ['op', 'type'])
    Node.__str__ = lambda n: n.op.name + '({})'.format( n.type )
        
    def __init__(self):
        self._digraph = nx.DiGraph()
        self._visited_ops = set()
    
    @property
    def digraph(self):
        return self._digraph
    
    def add_op(self, op):
        """
        Add an operator to the setup digraph, along with all reachable downstream operators.
        """
        if op in self._visited_ops:
            return
        self._visited_ops.add(op)

        Node = SetupDigraphBuilder.Node
        dg = self._digraph

        # Each operator gets two nodes: input and output
        input_node = Node(op, "input")
        output_node = Node(op, "output")
        dg.add_node(input_node, color='red')
        dg.add_node(output_node, color='white')
        # Implicit edge between them.
        dg.add_edge( input_node, output_node )

        # Add child operators
        for child in op.children:
            self.add_op(child)
            dg.add_edge( Node(op, "input"), Node(child, "input") )
            dg.add_edge( Node(child, "output"), Node(op, "output") )
        
        # Now add all other connected operators
        # Children will be skipped because they were already visited
        for slot in op.inputs.values() + op.outputs.values():
            self._visit_partners(slot)
                
    def _visit_partners(self, slot):
        dg = self._digraph
        Node = SetupDigraphBuilder.Node
        
        if slot.level == 0:
            for partner in slot.partners:
                upstream_op = slot.getRealOperator()
                downstream_op = partner.getRealOperator()
                upstream_node = Node( upstream_op, slot._type )
                downstream_node = Node( downstream_op, partner._type )

                # This edge might already be in the graph (if this is a child connection),
                #  but that's okay: nx.DiGraph ignores duplicate edges
                dg.add_edge( upstream_node, downstream_node )

                # If this is a child op, this operator will be skipped (it's already in the digraph)
                self.add_op(downstream_op)
        else:
            for subslot in slot:
                self._visit_partners(subslot)

# Monkey-patch the operator __str__ conversion (FIXME)
Operator.__str__ = lambda op: op.name

































