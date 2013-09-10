import logging
logger = logging.getLogger('tests.testOpBlockedLabelArray')

import numpy
import vigra
from lazyflow.graph import Graph
from lazyflow.operators.opBlockedLabelArray import OpBlockedLabelArray
from lazyflow.operators.opBlockedSparseLabelArray import OpBlockedSparseLabelArray

from lazyflow.roi import sliceToRoi
from lazyflow.utility.slicingtools import sl, slicing2shape
from lazyflow.utility import Timer, timeLogged

class TestLabelBenchmarks(object):
    
    def _init_op(self, labelOpClass):
        arrayshape = (1,1000,1000,10,1)
        blockshape = (1,64,64,64,1)
        
        dummy_data = numpy.ndarray( arrayshape, dtype=numpy.uint8 )
        dummy_data = vigra.taggedView( dummy_data, 'txyzc' )

        graph = Graph()
        op = labelOpClass(graph=graph)
        op.Input.setValue( dummy_data )
        op.blockShape.setValue( blockshape )
        op.eraser.setValue( 100 )
        op.deleteLabel.setValue( -1 )

        # The old label operator has a 'shape' input
        if 'shape' in op.inputs:
            op.shape.setValue( arrayshape )
        
        assert op.Output.ready()
        assert op.maxLabel.value == 0
        
        return arrayshape, op
    
    @timeLogged( logger, logging.INFO )
    def _testLotsOfLabels(self, labelOpClass):
        arrayshape, op = self._init_op( labelOpClass )
        
        slicing = numpy.s_[0:1, 0:1000, 0:1000, 5:6, 0:1]
        slicing_shape = slicing2shape(slicing)

        with Timer() as timer:
            op.Input[slicing] = 1*numpy.ones( slicing_shape, dtype=numpy.uint8 )
        logger.debug( "Creating labels took {} seconds".format( timer.seconds() ) )

        with Timer() as timer:
            op.deleteLabel.setValue( 1 )
            op.deleteLabel.setValue( -1 )
        logger.debug( "Deleting labels took {} seconds".format( timer.seconds() ) )

        with Timer() as timer:
            max_label = op.maxLabel.value
        logger.debug( "Determining the max label took {} seconds".format( timer.seconds() ) )

    @timeLogged( logger, logging.INFO )
    def testLotsOfLabels_OpBlockedLabelArray(self):
        self._testLotsOfLabels(OpBlockedLabelArray)
    
    @timeLogged( logger, logging.INFO )
    def testLotsOfLabels_OpBlockedSparseLabelArray(self):
        self._testLotsOfLabels(OpBlockedSparseLabelArray)
        
class TestOpBlockedSparseLabelArray1(object):
    """Basic test case."""

    def test(self):
        fake_input = numpy.zeros( (100,100,100), dtype=numpy.uint8 )
        fake_input = vigra.taggedView( fake_input, 'xyz' )
        g = Graph()
        op = OpBlockedLabelArray( graph=g )
        op.Input.setValue( fake_input )
        
        ERASER = 100
        op.eraser.setValue( ERASER )
        op.deleteLabel.setValue( 255 )
        op.blockShape.setValue( (10,20,10) )
        op.blockShape.setValue( (10,20,50) )
        
        block1 = numpy.ones( (10,10,10), dtype=numpy.uint8 )
        op.Input[0:10, 0:10, 0:10] = block1
    
        max_label = op.maxLabel.value
        assert max_label == 1
    
        block2 = 2*numpy.ones( (10,10,10), dtype=numpy.uint8 )
        op.Input[30:40, 0:10, 0:10] = block2
    
        max_label = op.maxLabel.value
        assert max_label == 2
        
        # Check the values we wrote
        assert ( op.Output[0:10,0:10,0:10].wait() == 1 ).all()
        assert ( op.Output[30:40, 0:10, 0:10].wait() == 2 ).all()
    
        # Check some values we left blank
        assert ( op.Output[20:30, 0:10, 0:10].wait() == 0 ).all()
    
        # Ask which blocks are nonzero
        assert op.nonzeroBlocks.value == [(slice(0, 10, None), slice(0, 20, None), slice(0, 50, None)),
                                          (slice(30, 40, None), slice(0, 20, None), slice(0, 50, None))]
        
        block1[5:10] = ERASER
        op.Input[0:10, 0:10, 0:10] = block1
        block1 = numpy.where( block1 == ERASER, 0, block1 )
        assert ( op.Output[0:10,0:10,0:10].wait() == block1 ).all()
    
        # Now delete label 1 entirely.
        # Label 2 should be shifted down into label 1
        op.deleteLabel.setValue(1)
        assert ( op.Output[0:10,0:10,0:10].wait() == 0 ).all()
        assert ( op.Output[30:40, 0:10, 0:10].wait() == 1 ).all()

class TestOpBlockedSparseLabelArray2(object):
    
    def setup(self):
        graph = Graph()
        op = OpBlockedLabelArray(graph=graph)
        arrayshape = (1,100,100,10,1)
        dummy_input = numpy.zeros( arrayshape, dtype=numpy.uint8 )
        dummy_input = vigra.taggedView(dummy_input, 'txyzc')
        op.Input.setValue( dummy_input )
        blockshape = (1,10,10,10,1)
        op.inputs["blockShape"].setValue( blockshape )
        op.eraser.setValue(100)

        slicing = sl[0:1, 1:15, 2:36, 3:7, 0:1]
        inDataShape = slicing2shape(slicing)
        inputData = ( 3*numpy.random.random(inDataShape) ).astype(numpy.uint8)
        op.Input[slicing] = inputData
        data = numpy.zeros(arrayshape, dtype=numpy.uint8)
        data[slicing] = inputData
        
        self.op = op
        self.slicing = slicing
        self.inData = inputData
        self.data = data

    def testOutput(self):
        """
        Verify that the label array has all of the data it was given.
        """
        op = self.op
        slicing = self.slicing
        inData = self.inData
        data = self.data

        # Output
        outputData = op.Output[...].wait()
        assert numpy.all(outputData[...] == data[...])

        # maxLabel        
        assert op.maxLabel.value == inData.max()

    def testSetupTwice(self):
        """
        If one of the inputs to the label array is changed, the output should not change (including max label value!)
        """
        # Change one of the inputs, causing setupOutputs to be changed.
        self.op.eraser.setValue(255)
        
        # Run the plain output test.
        self.testOutput()
        
    def testDeleteLabel(self):
        """
        Check behavior after deleting an entire label class from the sparse array.
        """
        op = self.op
        slicing = self.slicing
        inData = self.inData
        data = self.data

        op.deleteLabel.setValue(1)
        outputData = op.Output[...].wait()

        # Expected: All 1s removed, all 2s converted to 1s
        expectedOutput = numpy.where(self.data == 1, 0, self.data)
        expectedOutput = numpy.where(expectedOutput == 2, 1, expectedOutput)
        assert (outputData[...] == expectedOutput[...]).all()
        
        assert op.maxLabel.value == expectedOutput.max() == 1

        # delete label input resets automatically
        # assert op.deleteLabel.value == -1 # Apparently not?
    
    def testDeleteLabel2(self):
        """
        Another test to check behavior after deleting an entire label class from the sparse array.
        This one ensures that different blocks have different max label values before the delete occurs.
        """
        op = self.op
        slicing = self.slicing
        data = self.data

        assert op.maxLabel.value == 2
        
        # Choose slicings that do NOT intersect with any of the previous data or with each other
        # The goal is to make sure that the data for each slice ends up in a separate block
        slicing1 = sl[0:1, 60:65, 0:10, 3:7, 0:1]
        slicing2 = sl[0:1, 90:95, 0:90, 3:7, 0:1]

        expectedData = self.data[...]

        labels1 = numpy.ndarray(slicing2shape(slicing1), dtype=numpy.uint8)
        labels1[...] = 1
        op.Input[slicing1] = labels1
        expectedData[slicing1] = labels1

        labels2 = numpy.ndarray(slicing2shape(slicing2), dtype=numpy.uint8)
        labels2[...] = 2
        op.Input[slicing2] = labels2
        expectedData[slicing2] = labels2

        # Sanity check:
        # Does the data contain our new labels?
        assert (op.Output[...].wait() == expectedData).all()
        assert expectedData.max() == 2
        assert op.maxLabel.value == 2

        # Delete label 1
        op.deleteLabel.setValue(1)
        outputData = op.Output[...].wait()

        # Expected: All 1s removed, all 2s converted to 1s
        expectedData = numpy.where(expectedData == 1, 0, expectedData)
        expectedData = numpy.where(expectedData == 2, 1, expectedData)
        assert (outputData[...] == expectedData[...]).all()
        
        assert op.maxLabel.value == expectedData.max() == 1
        
    def testEraser(self):
        """
        Check that some labels can be deleted correctly from the sparse array.
        """
        op = self.op
        slicing = self.slicing
        inData = self.inData
        data = self.data

        assert op.maxLabel.value == 2
        
        erasedSlicing = list(slicing)
        erasedSlicing[1] = slice(1,2)

        outputWithEraser = data
        outputWithEraser[erasedSlicing] = 100
        
        op.Input[erasedSlicing] = outputWithEraser[erasedSlicing]

        expectedOutput = outputWithEraser
        expectedOutput[erasedSlicing] = 0
        
        outputData = op.Output[...].wait()
        assert (outputData[...] == expectedOutput[...]).all()
        
        assert expectedOutput.max() == 2
        assert op.maxLabel.value == 2
    
    def testEraseAll(self):
        """
        Test behavior when all labels of a particular class are erased.
        Note that this is not the same as deleting a label class, but should have the same effect on the output slots.
        """
        op = self.op
        slicing = self.slicing
        data = self.data

        assert op.maxLabel.value == 2
        
        newSlicing = list(slicing)
        newSlicing[1] = slice(1,2)

        # Add some new labels for a class that hasn't been seen yet (3)        
        threeData = numpy.ndarray(slicing2shape(newSlicing), dtype=numpy.uint8)
        threeData[...] = 3
        op.Input[newSlicing] = threeData        
        expectedData = data[...]
        expectedData[newSlicing] = 3
        
        # Sanity check: Are the new labels in the data?
        assert (op.Output[...].wait() == expectedData).all()
        assert expectedData.max() == 3
        assert op.maxLabel.value == 3

        # Now erase all the 3s
        eraserData = numpy.ones(slicing2shape(newSlicing), dtype=numpy.uint8) * 100
        op.Input[newSlicing] = eraserData        
        expectedData = data[...]
        expectedData[newSlicing] = 0
        
        # The data we erased should be zeros
        assert (op.Output[...].wait() == expectedData).all()
        
        # The maximum label should be reduced, because all the 3s were removed.
        assert expectedData.max() == 2
        assert op.maxLabel.value == 2

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler( sys.stdout ) )
    logger.setLevel( logging.DEBUG )
    
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
