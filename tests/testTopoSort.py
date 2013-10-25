import networkx as nx
import matplotlib.pyplot as plt

import collections

from lazyflow.graph import Graph, Operator, InputSlot, OutputSlot, SetupDigraphBuilder
from lazyflow.operators.operators import OpArrayPiper

configuration_order = []

class OpMaxValue(Operator):
    # Just a dumb operator to use for this test.
    InputA = InputSlot()
    InputB = InputSlot()
    
    MaxValue = OutputSlot()
    
    _name = ""
    
    def setupOutputs(self):
        self.MaxValue.meta.shape = (1,)
        self.MaxValue.meta.dtype = self.InputA.meta.dtype
    
    def execute(self, slot, subindex, roi, result):
        maxA = self.InputA[:].wait().max()
        maxB = self.InputB[:].wait().max()
        result[0] = max(maxA, maxB)
        return result
    
    def propagateDirty(self, slot, subindex, roi):
        self.MaxValue.setDirty()

    @property
    def name(self):
        if self.parent:
            return self.parent.name + '.' + self._name
        return self._name

class OpWithNestedOps(Operator):
    Input = InputSlot()

    OutputM1 = OutputSlot()
    OutputM2 = OutputSlot()
    
    _name = ""

    def __init__(self, *args, **kwargs):
        super(OpWithNestedOps, self).__init__(*args, **kwargs)
        opA1 = OpArrayPiper( parent=self )
        opA2 = OpArrayPiper( parent=self )
        
        opA1._name = "A1"
        opA2._name = "A2"
        
        opA1.Input.connect( self.Input )
        opA2.Input.connect( self.Input )

        opM1 = OpMaxValue( parent=self )
        opM1.InputA.connect( opA1.Output )
        opM1.InputB.connect( opA2.Output )
        
        opM2 = OpMaxValue( parent=self )
        opM2.InputA.connect( opA1.Output )
        opM2.InputB.connect( opA2.Output )

        opM1._name = "M1"
        opM2._name = "M2"
        
        self.opA1 = opA1
        self.opA2 = opA2
        self.opM1 = opM1
        self.opM2 = opM2

    def setupOutputs(self):
        pass
    
    def execute(self, slot, subindex, roi, result):
        pass
        
    def propagateDirty(self, slot, subindex, roi):
        pass

    @property
    def name(self):
        if self.parent:
            return self.parent.name + '.' + self._name
        return self._name

class TestTopoSort(object):
    
    def test(self):
        graph = Graph()
        opA1 = OpArrayPiper(graph=graph)
        opA2 = OpArrayPiper(graph=graph)
        
        opA1._name = "A1"
        opA2._name = "A2"
        
        opM1 = OpMaxValue(graph=graph)
        opM1.InputA.connect( opA1.Output )
        opM1.InputB.connect( opA2.Output )

        opM2 = OpMaxValue(graph=graph)
        opM2.InputA.connect( opM1.MaxValue )
        opM2.InputB.connect( opA2.Output )

        opM1._name = "M1"
        opM2._name = "M2"

        opA1.Input.setValue( 1 )
        opA2.Input.setValue( 2 )
        
        assert opM2.MaxValue.value == 2
        
        opNested = OpWithNestedOps( graph=graph )
        opNested.Input.connect( opM2.MaxValue )
        
        opNested._name = "opNested"
        
        opA3 = OpArrayPiper(graph=graph)
        opA3.Input.connect( opNested.OutputM1 )
        opA4 = OpArrayPiper(graph=graph)
        opA4.Input.connect( opNested.OutputM2 )
        
        opA3._name = "A3"
        opA4._name = "A4"

        builder = SetupDigraphBuilder(opA1)
        digraph = builder.digraph
        assert nx.is_directed_acyclic_graph(digraph)

        sorted_nodes = nx.topological_sort(digraph)
        for node in sorted_nodes:
            print node

        node_colors = map( lambda n: digraph.node[n]['color'],
                           digraph.nodes() )        
        nx.draw(digraph, node_color=node_colors)
        plt.show()
        return digraph

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)

