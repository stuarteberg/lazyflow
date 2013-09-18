import logging
logger = logging.getLogger("tests.testBlockedArray")

import numpy

from lazyflow.roi import roiToSlice, getIntersectingBlocks, TinyVector
from lazyflow.request import Request

import blockedarray

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

class TestBlockedArray(object):
    
    def _implMemoryLeaks(self):

        blockshape = (100,100,100)
        access_shape = (100,100,200)
        total_data_shape = (300,1000,1000)

        dtype = numpy.uint8
        control_data = numpy.zeros( total_data_shape, dtype=dtype )

        ba = blockedarray.BlockedArray3uint8( blockshape )
        read_block = numpy.zeros( access_shape, dtype=dtype )

        write_block_starts = getIntersectingBlocks( access_shape, [(0,0,0), total_data_shape] )
        read_block_starts = getIntersectingBlocks( access_shape, [(0,0,0), total_data_shape] )

        set_starts = set( map( tuple, write_block_starts) )
        
        expected_cache_blocks = getIntersectingBlocks( blockshape, [(0,0,0), total_data_shape] )
        num_blocks = len(expected_cache_blocks)
        print "num_blocks will be:", num_blocks
        
        starting_memory_usage_mb = getMemoryUsageMb()
        logger.debug("Starting with memory usage: {} MB".format( starting_memory_usage_mb ))

        write_block = numpy.random.randint( 0, 255, access_shape ).astype( dtype=dtype )

        for _ in range(1):
            numpy.random.shuffle(write_block_starts)
            numpy.random.shuffle(read_block_starts)
            
            shuffled_starts = set( map( tuple, write_block_starts) )
            assert set_starts == shuffled_starts
            
            for write_start, read_start in zip(write_block_starts, read_block_starts):
                write_roi = write_start, write_start+access_shape
                #write_block = numpy.random.randint( 0, 255, access_shape ).astype( dtype=dtype )
                write_block = numpy.zeros( access_shape, dtype=dtype )
                ba.writeSubarray( write_roi[0], write_roi[1], write_block )
                control_data[roiToSlice(*write_roi)] = write_block

                read_roi = read_start, read_start+access_shape
                ba.readSubarray( read_roi[0], read_roi[1], read_block )
                assert (control_data[roiToSlice(*read_roi)] == read_block).all()

        del write_block

        print "numblocks:", ba.numBlocks()
        blocks = ba.blocks((0,0,0), total_data_shape)
        #print "len(blocks[0]) ==", len(blocks[0])
        #print "len(set(blocks[0])) ==", len(set(map(tuple, blocks[0])))
        
        set_block_starts = set(map(tuple, blocks[0]))
        set_expected_starts = set(map(tuple, expected_cache_blocks))
        assert 0 == len(set_block_starts ^ set_expected_starts)
        #print set_expected_starts
        #print set_block_starts


        total_data_mb = numpy.prod(total_data_shape) / 1e6
        logger.debug("Expected cache usage is: {} MB".format( total_data_mb ))
        #off_by_one_blocks = getIntersectingBlocks( blockshape, [(0,0,0), TinyVector(total_data_shape)+1] )
        #off_by_one_mb = len(off_by_one_blocks) * numpy.prod( blockshape ) / 1e6
        #logger.debug("If cache is off-by-one, usage is: {} MB".format( off_by_one_mb ))
        
        tolerance_mb = 10        
        memory_increase_mb = getMemoryUsageMb() - starting_memory_usage_mb
        logger.debug("After test, memory increase is: {} MB".format( memory_increase_mb ))
        assert memory_increase_mb < total_data_mb + tolerance_mb

    def testMemoryLeaks1(self):
        self._implMemoryLeaks()

    def testMemoryLeaks2(self):
        self._implMemoryLeaks()

    def testStartStopWithNumpyArrays(self):
        blockshape = numpy.array( (100,100,100) )
        blockshape = tuple(blockshape)
        ba = blockedarray.BlockedArray3uint8( blockshape )
        

    def testEnumerateBlocks(self):
        blockshape = numpy.array((10,10,10))
        ba = blockedarray.BlockedArray3uint8( blockshape )
        roi = numpy.array([(5,5,5), (15,15,15)])
        blocks = ba.enumerateBlocksInRange( *roi )

        expected = getIntersectingBlocks( blockshape, roi )
        expected = numpy.array( expected )
        expected /= blockshape
        
        assert set(map(tuple, blocks)) == set(map(tuple, expected))

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler( sys.stdout ) )
    logger.setLevel( logging.DEBUG )

    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
