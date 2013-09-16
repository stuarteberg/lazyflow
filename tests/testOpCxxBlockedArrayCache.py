import threading
import numpy
import vigra
from lazyflow.graph import Graph
from lazyflow.roi import TinyVector, sliceToRoi, roiToSlice, getIntersectingBlocks, roiFromShape, getBlockBounds
from lazyflow.request import Request
from lazyflow.operators import OpArrayPiper
from lazyflow.operators.opCxxBlockedArrayCache import OpCxxBlockedArrayCache
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache
from lazyflow.operators.opSlicedBlockedArrayCache import OpSlicedBlockedArrayCache

import logging
logger = logging.getLogger("testOpCxxBlockedArrayCache")

import gc
import psutil
def getMemoryUsageMb():
    """
    Get the current memory usage for the whole system (not just python).
    """
    # Collect garbage first
    gc.collect()
    vmem = psutil.virtual_memory()
    mem_usage_mb = (vmem.total - vmem.available) / 1e6
    return mem_usage_mb

class OpArrayPiperWithAccessCount(OpArrayPiper):
    """
    A simple array piper that counts how many times its execute function has been called.
    """
    def __init__(self, *args, **kwargs):
        super(OpArrayPiperWithAccessCount, self).__init__(*args, **kwargs)
        self.accessCount = 0
        self._lock = threading.Lock()
    
    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.accessCount += 1        
        super(OpArrayPiperWithAccessCount, self).execute(slot, subindex, roi, result)
        

class TestOpCxxBlockedArrayCacheMemoryUsage(object):

    def setUp(self):
        self.dataShape = (1,1000,1000,1000,1)
        self._dtype = numpy.float32
        self._dtype_bytes = 4
        self.data = numpy.zeros( self.dataShape, dtype=self._dtype )
        self.data = self.data.view(vigra.VigraArray)
        self.data.axistags = vigra.defaultAxistags('txyzc')

        graph = Graph()
        opProvider = OpArrayPiperWithAccessCount(graph=graph)
        opProvider.Input.setValue(self.data)
        self.opProvider = opProvider

        self._blockshape = TinyVector([1,100,100,100,1])
        opCache = OpCxxBlockedArrayCache(graph=graph)
        opCache.innerBlockShape.setValue( self._blockshape )
        #opCache = OpSlicedBlockedArrayCache(graph=graph)
        #opCache.innerBlockShape.setValue( (self._blockshape,) )
        #opCache.outerBlockShape.setValue( (self._blockshape,) )
        #opCache = OpBlockedArrayCache(graph=graph)
        #opCache.innerBlockShape.setValue( tuple(self._blockshape) )
        #opCache.outerBlockShape.setValue( tuple(self._blockshape) )

        opCache.Input.connect(opProvider.Output)
        opCache.fixAtCurrent.setValue(False)
        self.opCache = opCache

    def testForMemoryLeaks(self):
        opCache = self.opCache
        access_roi = [(0,0,0,0,0),(1,200,400,1000,1)]
        access_roi = map(TinyVector, access_roi)

        access_block_shape = (1,100,100,200,1)

        # Prepare to request a block
        destination_block = numpy.zeros( access_block_shape, dtype=self._dtype )

        starting_memory_usage_mb = getMemoryUsageMb()
        logger.debug("Starting with memory usage: {:.1f} MB".format( starting_memory_usage_mb ))

        print "Extant reqs:", len(filter( lambda o: isinstance(o, Request), gc.get_objects() ))
        print "Extant ndarrays:", len(filter( lambda o: isinstance(o, numpy.ndarray), gc.get_objects() ))

        block_starts = getIntersectingBlocks( access_block_shape, access_roi )
        for _ in range(2):
            print "****************************************"
            opCache.Input.setDirty()
            for start in block_starts:
                start = TinyVector( map(int, start) )
                slicing = roiToSlice( start, start+access_block_shape )
                req = opCache.Output( slicing ).writeInto(destination_block)

                # Access it.  Will cause cache storage.
                data = req.wait()
                data = data.view(vigra.VigraArray)
                data.axistags = opCache.Output.meta.axistags
                assert (data == self.data[slicing]).all()
                del data
                req.clean()
                del req

        print "Extant reqs:", len(filter( lambda o: isinstance(o, Request), gc.get_objects() ))
        print "Extant ndarrays:", len(filter( lambda o: isinstance(o, numpy.ndarray), gc.get_objects() ))

        #assert len(opCache._inprocess_requests) == 0
        cached_mb = self._dtype_bytes * numpy.prod( access_roi[1] ) / 1e6
        
        # Allow a little extra RAM consumption for bookkeeping members.
        overhead_tolerance_mb = 0.1*cached_mb
        
        memory_increase_mb = getMemoryUsageMb() - starting_memory_usage_mb
        logger.debug("Cache should consume {:.1f} MB".format( cached_mb ))
        logger.debug("After accesses, memory increase is: {:.1f} MB".format( memory_increase_mb ))
        assert memory_increase_mb < cached_mb + overhead_tolerance_mb, \
            "Memory leak: Expected only {:.1f} additional MB consumed, but {:.1f} were consumed."\
            " (Leaked {} MB)"\
            "".format( cached_mb, memory_increase_mb, memory_increase_mb - cached_mb )


class KeyMaker():
    def __getitem__(self, *args):
        return list(*args)
make_key = KeyMaker()

class OpArrayPiperWithAccessCount(OpArrayPiper):
    """
    A simple array piper that counts how many times its execute function has been called.
    """
    def __init__(self, *args, **kwargs):
        super(OpArrayPiperWithAccessCount, self).__init__(*args, **kwargs)
        self.accessCount = 0
        self._lock = threading.Lock()
    
    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.accessCount += 1        
        super(OpArrayPiperWithAccessCount, self).execute(slot, subindex, roi, result)
        

class TestOpCxxBlockedArrayCache(object):

    def setUp(self):
        self.dataShape = (1,100,100,10,1)
        self.data = (numpy.random.random(self.dataShape) * 100).astype(numpy.uint32)
        self.data = self.data.view(vigra.VigraArray)
        self.data.axistags = vigra.defaultAxistags('txyzc')

        graph = Graph()
        opProvider = OpArrayPiperWithAccessCount(graph=graph)
        opProvider.Input.setValue(self.data)
        self.opProvider = opProvider
        
        opCache = OpCxxBlockedArrayCache(graph=graph)
        opCache.Input.connect(opProvider.Output)
        opCache.innerBlockShape.setValue( (10,10,10,10,10) )
        opCache.outerBlockShape.setValue( (20,20,20,20,20) )
        opCache.fixAtCurrent.setValue(False)
        self.opCache = opCache

    def testCacheAccess(self):
        opCache = self.opCache
        opProvider = self.opProvider        
        
        expectedAccessCount = 0
        assert opProvider.accessCount == expectedAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, expectedAccessCount)
        
        # Block-aligned request
        slicing = make_key[0:1, 0:10, 10:20, 0:10, 0:1]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1        
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, expectedAccessCount)

        # Same request should come from cache, so access count is unchanged
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == expectedAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, expectedAccessCount)
                
        # Not block-aligned request
        slicing = make_key[0:1, 5:15, 10:20, 0:10, 0:1]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, expectedAccessCount)

        # Same request should come from cache, so access count is unchanged
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == expectedAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, expectedAccessCount)

    def testDirtySource(self):
        opCache = self.opCache
        opProvider = self.opProvider        
        
        oldAccessCount = 0
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)

        # Request
        slicing = make_key[:, 0:50, 15:45, 0:10, :]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()
        
        # Our slice intersects 3*3=9 outerBlocks, and a total of 20 innerBlocks
        # Inner caches are allowed to split up the accesses, so there could be as many as 20
        minAccess = oldAccessCount + 9
        maxAccess = oldAccessCount + 20
        assert opProvider.accessCount >= minAccess
        assert opProvider.accessCount <= maxAccess
        oldAccessCount = opProvider.accessCount

        # Track dirty notifications
        gotDirtyKeys = []
        def handleDirty(slot, roi):
            gotDirtyKeys.append( list(roiToSlice(roi.start, roi.stop)) )
        opCache.Output.notifyDirty(handleDirty)
        
        # Change some of the input data and mark it dirty
        dirtykey = make_key[0:1, 10:20, 20:30, 0:3, 0:1]
        self.data[dirtykey] = 0.12345
        opProvider.Input.setDirty(dirtykey)        
        assert len(gotDirtyKeys) > 0
        
        # Same request, but should need to access the data again due to dirtiness
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()

        # The dirty data intersected 1 outerBlocks and a total of 1 innerBlock.
        minAccess = oldAccessCount + 1
        maxAccess = oldAccessCount + 1
        assert opProvider.accessCount >= minAccess
        assert opProvider.accessCount <= maxAccess
        oldAccessCount = opProvider.accessCount

    def testFixAtCurrent(self):
        opCache = self.opCache
        opProvider = self.opProvider        

        # Track dirty notifications
        gotDirtyKeys = []
        def handleDirty(slot, roi):
            gotDirtyKeys.append( list(roiToSlice(roi.start, roi.stop)) )
        opCache.Output.notifyDirty(handleDirty)

        opCache.fixAtCurrent.setValue(True)

        oldAccessCount = 0
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)

        # Request (no access to provider because fixAtCurrent)
        slicing = make_key[:, 0:50, 15:45, 0:1, :]
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)

        # We haven't accessed this data yet,
        # but fixAtCurrent is True so the cache gives us zeros
        assert (data == 0).all()

        opCache.fixAtCurrent.setValue(False)
        
        # Since we got zeros while the cache was fixed, the requested 
        #  tiles are signaled as dirty when the cache becomes unfixed.
        assert len(gotDirtyKeys) == 1
        #assert gotDirtyKeys[0] == make_key[0:1, 0:60, 0:60, 0:10, 0:1]

        # Request again.  Data should match this time.
        oldAccessCount = opProvider.accessCount
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()

        # Our slice intersects 3*3=9 outerBlocks, and a total of 20 innerBlocks
        # Inner caches are allowed to split up the accesses, so there could be as many as 20
        minAccess = oldAccessCount + 9
        maxAccess = oldAccessCount + 20
        assert opProvider.accessCount >= minAccess
        assert opProvider.accessCount <= maxAccess
        oldAccessCount = opProvider.accessCount

        # Request again.  Data should match WITHOUT requesting from the source.
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)

        # Freeze it again
        opCache.fixAtCurrent.setValue(True)

        # Clear previous
        gotDirtyKeys = []

        # Change some of the input data that ISN'T cached yet and mark it dirty
        dirtykey = make_key[0:1, 90:100, 90:100, 0:1, 0:1]
        self.data[dirtykey] = 0.12345
        opProvider.Input.setDirty(dirtykey)

        # Dirtiness not propagated due to fixAtCurrent
        assert len(gotDirtyKeys) == 0
        
        # Same request.  Data should still match the previous data (not yet refreshed)
        data2 = opCache.Output( slicing ).wait()
        data2 = data2.view(vigra.VigraArray)
        data2.axistags = opCache.Output.meta.axistags
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)
        assert (data2 == data).all()

        # Unfreeze.
        opCache.fixAtCurrent.setValue(False)

        # Dirty blocks are propagated after the cache is unfixed.
        assert len(gotDirtyKeys) > 0

        # Same request.  Data should be updated now that we're unfrozen.
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()

        # Dirty data did not intersect with this request.
        # Data should still be cached (no extra accesses)
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)

        ###########################3
        # Freeze it again
        opCache.fixAtCurrent.setValue(True)

        # Reset tracked notifications
        gotDirtyKeys = []
        
        # Change some of the input data that IS cached and mark it dirty
        dirtykey = make_key[:, 0:25, 20:40, 0:1, :]
        self.data[dirtykey] = 0.54321
        opProvider.Input.setDirty(dirtykey)

        # Dirtiness not propagated due to fixAtCurrent
        assert len(gotDirtyKeys) == 0
        
        # Same request.  Data should still match the previous data (not yet refreshed)
        data2 = opCache.Output( slicing ).wait()
        data2 = data2.view(vigra.VigraArray)
        data2.axistags = opCache.Output.meta.axistags
        assert opProvider.accessCount == oldAccessCount, "Access count={}, expected={}".format(opProvider.accessCount, oldAccessCount)
        assert (data2 == data).all()

        # Unfreeze. Previous dirty notifications should now be seen.
        opCache.fixAtCurrent.setValue(False)
        assert len(gotDirtyKeys) > 0

        # Same request.  Data should be updated now that we're unfrozen.
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()

        # The dirty data intersected 2 outerBlocks, and a total of 6 innerblocks
        # Inner caches are allowed to split up the accesses, so there could be as many as 6
        minAccess = oldAccessCount + 2
        maxAccess = oldAccessCount + 6
        assert opProvider.accessCount >= minAccess
        assert opProvider.accessCount <= maxAccess
        oldAccessCount = opProvider.accessCount

        #####################        

        #### Repeat plain dirty test to ensure fixAtCurrent didn't mess up the block states.

        gotDirtyKeys = []

        # Change some of the input data and mark it dirty
        dirtykey = make_key[0:1, 10:11, 20:21, 0:3, 0:1]
        self.data[dirtykey] = 0.54321
        opProvider.Input.setDirty(dirtykey)

        assert len(gotDirtyKeys) > 0
        
        # Should need access again.
        slicing = make_key[:, 0:50, 15:45, 0:10, :]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()

        # The dirty data intersected 1 outerBlocks and a total of 1 innerblock
        minAccess = oldAccessCount + 1
        maxAccess = oldAccessCount + 1
        assert opProvider.accessCount >= minAccess
        assert opProvider.accessCount <= maxAccess
        oldAccessCount = opProvider.accessCount


if __name__ == "__main__":
    import sys
    logging.getLogger().addHandler( logging.StreamHandler( sys.stdout ) )
    logger.setLevel( logging.DEBUG )
    logging.getLogger("lazyflow.operators.opCxxBlockedArrayCache").setLevel(logging.DEBUG)

    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
