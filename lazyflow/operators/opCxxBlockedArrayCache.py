import weakref
import threading
from functools import partial
import logging
logger = logging.getLogger(__name__)

import numpy
import blockedarray

from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.request import RequestPool
from lazyflow.operators.opCache import OpCache
from lazyflow.roi import TinyVector, roiFromShape, roiToSlice, getIntersectingBlocks, getBlockBounds

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
        IN_PROCESS  = 0
        DIRTY       = 1
        CLEAN       = 2

    def __init__(self, *args, **kwargs):
        super( OpCxxBlockedArrayCache, self ).__init__( *args, **kwargs )
        self._blockedarray = None
        self._dirty_while_fixed = False
        self._dirty_blocks = None
        self._lock = threading.Lock() # Regular lock: Must never call wait() while this is held!
        self._block_shape = None
    
    def setupOutputs(self):
        assert not numpy.issubdtype(object, self.Input.meta.dtype), \
            "Can't use this operator to cache arrays of objects."
        block_shape = tuple( numpy.minimum( self.innerBlockShape.value, self.Input.meta.shape ) )

        if not self._blockedarray \
        or self._block_shape != block_shape \
        or self.Input.meta.dtype != self.Output.meta.dtype:
            self._block_shape = block_shape
            self._blockedarray = create_blockedarray( self._block_shape, self.Input.meta.dtype )
            self._block_states = self._init_block_states()
            self._inprocess_requests = {}

        self.Output.meta.assignFrom(self.Input.meta)

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

            # For blocks that are already being requested by other threads, no need to
            #  make a new request.  Just wait for the requests that are already being processed.
            inprocess_state_indices = numpy.nonzero(states == BlockState.IN_PROCESS)
            if len(inprocess_state_indices[0]) > 0:
                # Make a list of state indices we're interested in.
                inprocess_state_coords = self._index_tuple_to_coord_array( inprocess_state_indices )
                inprocess_state_coords += state_roi[0] # global coords, not view coords.
                for state_coord in map(tuple, inprocess_state_coords):
                    pool.add( self._inprocess_requests[state_coord] )

            # Create new requests for dirty blocks
            dirty_state_indices = numpy.nonzero(states == BlockState.DIRTY)
            if len(dirty_state_indices[0]) > 0:
                # Make a list of state indices we're interested in.
                dirty_state_coords = self._index_tuple_to_coord_array( dirty_state_indices )
                dirty_state_coords += state_roi[0] # global coords, not view coords
                for state_coord in map(tuple, dirty_state_coords):
                    # Create request
                    block_roi_global = self._get_block_roi( state_coord )
                    req = self.Input( *block_roi_global )
                    pool.add( req )

                    # If possible, use the pre-allocated result array as our scratch space to save memory.
                    if (block_roi_global[0] >= roi.start).all() and (block_roi_global[1] <= roi.stop).all():
                        block_roi_in_result = numpy.array(block_roi_global) - roi.start
                        result_slicing = roiToSlice(*block_roi_in_result)
                        req.writeInto( result[result_slicing] )

                    # Save to cache when finished
                    req.notify_finished( partial(self._handle_request_completed, weakref.ref(req), state_coord ) )

                    # Bookkeeping
                    self._block_states[ state_coord ] = BlockState.IN_PROCESS
                    self._inprocess_requests[ state_coord ] = req
        
        # Wait for all requests (including new and previously in-process).
        pool.wait()

        # Now the cache is up-to-date.  Simply ask it for all the data we need.
        with self._lock:
            # As a potential optimization here, we could skip blocks that 
            #  were dirty (they were already written into result).
            # (For now, that's more trouble than its worth -- it may even hurt performance.)
            self._blockedarray.readSubarray( roi.start, roi.stop, result )

        return result

    def _handle_request_completed(self, weak_request, block_state_coord, data ):
        block_roi = self._get_block_roi( block_state_coord )
        with self._lock:
            # Write result into cache
            self._blockedarray.writeSubarray( block_roi[0], block_roi[1], data )

            # Update bookkeeping members
            del self._inprocess_requests[block_state_coord]
            self._block_states[ block_state_coord ] = OpCxxBlockedArrayCache.BlockState.CLEAN
            
            # Memory optimization: Immediately free this request's data
            weak_request().clean()
            del data


    def propagateDirty(self, slot, subindex, roi):
        fixed = self.fixAtCurrent.value
        if slot == self.Input:
            with self._lock:
                state_roi = self._get_state_roi( (roi.start, roi.stop) )
                self._block_states[roiToSlice(*state_roi)] = self.BlockState.DIRTY                
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
