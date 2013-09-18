import time
import threading
import collections
from functools import partial
import logging
logger = logging.getLogger(__name__)

import numpy
import blockedarray

from lazyflow.utility import Timer
from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.request import RequestPool
from lazyflow.operators.opCache import OpCache
from lazyflow.roi import TinyVector, roiFromShape, roiToSlice, getIntersectingBlocks, getBlockBounds

from arrayCacheMemoryMgr import ArrayCacheMemoryMgr

class OpCxxBlockedArrayCache( OpCache ):
    """
    A reimplementation of OpBlockedArrayCache that backed by the blockedarray module.
    """
    # FIXME: Names
    Input = InputSlot()
    innerBlockShape = InputSlot()
    fixAtCurrent = InputSlot(value=False)
    
    # TODO: Eliminate this slot. (Doesn't do anything.)
    outerBlockShape = InputSlot(optional=True)
   
    Output = OutputSlot()
    
    class BlockState(object):
        # These constants must remain in this order!
        DIRTY       = 0
        CLEAN       = 1
        IN_PROCESS  = 2 # Anything >= 2 means "someone is waiting for this data".
                        # Note that states of 4,5,etc. are also valid, depending
                        #   on how many waiting requests there are.

    def __init__(self, *args, **kwargs):
        super( OpCxxBlockedArrayCache, self ).__init__( *args, **kwargs )
        self._blockedarray = None
        self._dirty_while_fixed = False
        self._dirty_blocks = None
        self._lock = threading.Lock() # Regular lock: Must never call wait() while this is held!
        self._block_shape = None
        self._access_times = None
        
        ArrayCacheMemoryMgr.instance.add(self)
    
    def setupOutputs(self):
        assert not numpy.issubdtype(object, self.Input.meta.dtype), \
            "Can't use this operator to cache arrays of objects."
        block_shape = tuple( numpy.minimum( self.innerBlockShape.value, self.Input.meta.shape ) )

        if not self._blockedarray \
        or self._block_shape != block_shape \
        or self.Input.meta.dtype != self.Output.meta.dtype:
            self._block_shape = block_shape
            self._block_size_bytes = numpy.prod(block_shape) * self._getDtypeBytes(self.Input.meta.dtype)
            self._blockedarray = create_blockedarray( self._block_shape, self.Input.meta.dtype )
            self._block_states = self._init_block_states()
            self._access_times = numpy.zeros( self._get_block_state_shape(), dtype=numpy.float32 )
            self._inprocess_requests = {}

        self.Output.meta.assignFrom(self.Input.meta)

    def _getDtypeBytes(self, dtype):
        if type(dtype) is numpy.dtype:
            # Make sure we're dealing with a type (e.g. numpy.float64),
            #  not a numpy.dtype
            dtype = dtype.type
        return dtype().nbytes

    def _init_block_states(self):
        # Create the block_states bookkeeping array and init with all DIRTY.
        block_state_shape = self._get_block_state_shape()
        BlockState = OpCxxBlockedArrayCache.BlockState 
        return BlockState.DIRTY * numpy.ones( block_state_shape, dtype=numpy.uint8 )

    def _get_block_state_shape(self):
        block_shape = self.innerBlockShape.value
        input_shape = self.Input.meta.shape
        block_coords_array = getIntersectingBlocks( block_shape, roiFromShape(input_shape), asarray=True )
        return block_coords_array.shape[:-1]

    def _get_block_roi(self, state_coord):
        block_shape = self.innerBlockShape.value
        input_shape = self.Input.meta.shape
        block_start = TinyVector(state_coord) * block_shape
        return getBlockBounds( input_shape, block_shape, block_start )
    
    def _index_tuple_to_coord_array(self, index_tuple):
        return numpy.array( index_tuple ).transpose()
    
    def _get_state_roi(self, block_roi):
        block_shape = TinyVector(self._block_shape)
        state_roi = (block_roi[0] / block_shape, ( block_roi[1] + (block_shape - 1) ) / block_shape)
        return state_roi

    def execute(self, slot, subindex, roi, result):
        # Determine the corresponding start, stop coords in our state bookkeeping array
        state_roi = self._get_state_roi( (roi.start, roi.stop) )
        
        fixed = self.fixAtCurrent.value
        if fixed:
            # Just provide whatever data we've already got.
            # The blockedarray outputs zeros if it has no data for a block.
            with self._lock:
                self._blockedarray.readSubarray( roi.start, roi.stop, result )

            # If we're giving the user any dirty data while we're fixed, we must 
            #  notify him with a dirty notification when we become unfixed
            self._dirty_while_fixed |= (self._block_states[roiToSlice(*state_roi)] == self.BlockState.DIRTY).any()                
            return result

        # Convenience local vars.
        BlockState = OpCxxBlockedArrayCache.BlockState

        # Use this pool to wait for all requests at once.
        pool = RequestPool()

        with self._lock:
            states = self._block_states[roiToSlice(*state_roi)]
            pending_block_coords = []

            # For blocks that are already being requested by other threads, no need to
            #  make a new request.  Just wait for the requests that are already being processed.
            inprocess_state_indices = numpy.nonzero(states >= BlockState.IN_PROCESS)
            if len(inprocess_state_indices[0]) > 0:
                # Make a list of state indices we're interested in.
                inprocess_state_coords = self._index_tuple_to_coord_array( inprocess_state_indices )
                inprocess_state_coords += state_roi[0] # global coords, not view coords.
                inprocess_state_coords = map(tuple, inprocess_state_coords)
                pending_block_coords += inprocess_state_coords
                
                for state_coord in inprocess_state_coords:
                    self._block_states[ state_coord ] += 1
                    pool.add( self._inprocess_requests[state_coord] )

            # Create new requests for dirty blocks
            dirty_state_indices = numpy.nonzero(states == BlockState.DIRTY)
            if len(dirty_state_indices[0]) > 0:
                # Make a list of state indices we're interested in.
                dirty_state_coords = self._index_tuple_to_coord_array( dirty_state_indices )
                dirty_state_coords += state_roi[0] # global coords, not view coords
                dirty_state_coords = map(tuple, dirty_state_coords)
                pending_block_coords += dirty_state_coords
                
                for state_coord in dirty_state_coords:
                    # Create request
                    block_roi_global = self._get_block_roi( state_coord )
                    req = self.Input( *block_roi_global )

                    # If possible, use the pre-allocated result array as our scratch space to save memory.
                    if (block_roi_global[0] >= roi.start).all() and (block_roi_global[1] <= roi.stop).all():
                        block_roi_in_result = numpy.array(block_roi_global) - roi.start
                        result_slicing = roiToSlice(*block_roi_in_result)
                        req.writeInto( result[result_slicing] )

                    # Save to cache when finished
                    req.notify_finished( partial(self._handle_request_completed, state_coord ) )

                    # Bookkeeping
                    self._block_states[ state_coord ] = BlockState.IN_PROCESS
                    self._inprocess_requests[ state_coord ] = req
                    pool.add( req )
        
        # Wait for all requests (including new and previously in-process).
        # Important: We do not own the lock while we wait.
        pool.wait()

        # Now the cache is up-to-date.  Simply ask it for all the data we need.
        with self._lock:
            # As a potential optimization here, we could skip blocks that 
            #  were dirty (they were already written into result).
            # (For now, that's more trouble than its worth -- it may even hurt performance.)
            self._blockedarray.readSubarray( roi.start, roi.stop, result )

            # Decrement the bookkeeping state for all requests we waited for.
            # A block is not considered "clean" until its data has been transferred to all requests that wanted it.
            for block_state_coord in pending_block_coords:
                assert self._block_states[ block_state_coord ] >= OpCxxBlockedArrayCache.BlockState.IN_PROCESS
                self._block_states[ block_state_coord ] -= 1
                if self._block_states[ block_state_coord ] == OpCxxBlockedArrayCache.BlockState.CLEAN:
                    # No one is waiting for this request data anymore, so it's safe to be removed from the 'in process' list.
                    # (This list determines which blocks can safely be discarded by the memory management thread.)
                    del self._inprocess_requests[block_state_coord]
                    
                    # Track most recent access, used for memory management.
                    self._access_times[block_state_coord] = time.time()

        return result

    def _handle_request_completed(self, block_state_coord, data ):
        block_roi = self._get_block_roi( block_state_coord )
        with self._lock:
            # Write result into cache
            self._blockedarray.writeSubarray( block_roi[0], block_roi[1], data )

            # Memory optimization: Immediately free this request's data
            req = self._inprocess_requests[block_state_coord]
            req.clean()
            del data
    
    def propagateDirty(self, slot, subindex, roi):
        fixed = self.fixAtCurrent.value
        if slot == self.Input:
            with self._lock:
                state_roi = self._get_state_roi( (roi.start, roi.stop) )
                state_slicing = roiToSlice(*state_roi)
                self._block_states[state_slicing] = self.BlockState.DIRTY
                self._access_times[state_slicing] = 0 # Reset.
            if fixed:
                self._dirty_while_fixed = True
            else:
                self.Output.setDirty(roi)
        if slot == self.fixAtCurrent:
            if not fixed and self._dirty_while_fixed:
                # Our input became dirty while we were fixed.
                # Mark the entire output dirty
                self.Output.setDirty()
            else:
                self._dirty_while_fixed = False

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.Input
        assert value.shape == tuple(roi.stop - roi.start), \
            "Roi/shape mismatch: Roi is {}, value has shape {}".format( roi, value.shape )
        
        with self._lock:
            self._blockedarray.writeSubarray( tuple(roi.start), tuple(roi.stop), value, self.eraser.value )
        
        self.Output.setDirty( roi )

    ### Memory management members ###
    BlockStats = collections.namedtuple( "BlockStats", ["last_access", "size_bytes", "attempt_free_fn"] )

    def get_block_stats(self):
        # None of this is protected by our lock.
        # That's okay because The information returned here is used for heuristic purposes only.
        # In _attempt_free_block(), we ensure that the block state hasn't changed since we provided its stats.
        
        BlockStats = OpCxxBlockedArrayCache.BlockStats
        BlockState = OpCxxBlockedArrayCache.BlockState
        input_shape = self.Input.meta.shape

        with self._lock:
            all_stored_block_starts = self._blockedarray.blocks( *roiFromShape( input_shape ) )[0]
        state_coords = all_stored_block_starts / self._block_shape
        states = self._block_states[ tuple(state_coords.transpose()) ]
        coords_and_states = numpy.concatenate( (state_coords, numpy.transpose([states])), axis=1 )
        
        # Filter out 'in process' blocks -- those can't be freed at the moment.
        filtered_coords_and_states = filter( lambda c: c[-1] != BlockState.IN_PROCESS, coords_and_states )
        if not filtered_coords_and_states:
            return []

        filtered_state_coords = numpy.array( filtered_coords_and_states )[:, :-1]
        access_times = self._access_times[ tuple(filtered_state_coords.transpose()) ]
        
        block_stats = []
        for state_coord, access_time in zip( filtered_state_coords, access_times ):
            attempt_free_fn = partial( self._attempt_free_block, state_coord * self._block_shape, access_time )
            block_stats.append( BlockStats( access_time, self._block_size_bytes, attempt_free_fn ) )
        return block_stats
    
    def _attempt_free_block(self, block_start, access_time):
        block_start = TinyVector(block_start)
        with self._lock:
            state_coord = tuple( block_start / self._block_shape )
            new_access_time = self._access_times[state_coord]
            if new_access_time != 0 and new_access_time != access_time:
                return False
            self._blockedarray.deleteSubarray( block_start, block_start + self._block_shape )
            self._access_times[state_coord] = 0
            self._block_states[state_coord] = OpCxxBlockedArrayCache.BlockState.DIRTY
            return True

def create_blockedarray( block_shape, dtype ):
    dimension = len(block_shape)
    lookup = { numpy.uint8 :   { 2 : blockedarray.BlockedArray2uint8,
                                 3 : blockedarray.BlockedArray3uint8,
                                 4 : blockedarray.BlockedArray4uint8,
                                 5 : blockedarray.BlockedArray5uint8 },
               
               numpy.uint32 :  { 2 : blockedarray.BlockedArray2uint32,
                                 3 : blockedarray.BlockedArray3uint32,
                                 4 : blockedarray.BlockedArray4uint32,
                                 5 : blockedarray.BlockedArray5uint32  },

               numpy.float32 : { 2 : blockedarray.BlockedArray2float32,
                                 3 : blockedarray.BlockedArray3float32,
                                 4 : blockedarray.BlockedArray4float32,
                                 5 : blockedarray.BlockedArray5float32  } }

    return lookup[dtype][dimension]( block_shape )
