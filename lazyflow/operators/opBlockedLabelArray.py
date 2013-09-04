import numpy
import vigra
import blockedarray
from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.operators.opCache import OpCache
from lazyflow.roi import roiFromShape, roiToSlice

class OpBlockedLabelArray(OpCache):
    # FIXME All slots: naming conventions
    Input = InputSlot()
    blockShape = InputSlot()
    eraser = InputSlot()
    deleteLabel = InputSlot(optional=True)
    
    Output = OutputSlot()
    nonzeroValues = OutputSlot()
    nonzeroCoordinates = OutputSlot()
    nonzeroBlocks = OutputSlot()
    maxLabel = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super(OpBlockedLabelArray, self).__init__( *args, **kwargs )
        self._blocked_array = None
        self._previous_delete_label = -1
        self._max_label = 0

    def setupOutputs(self):
        blockshape = self.blockShape.value
        if not self._blocked_array \
        or self._blocked_array.blockShape() != blockshape:
        #or self.Input.meta.dtype != self.Output.meta.dtype:
            self._blocked_array = create_blockedarray( blockshape, self.Input.meta.dtype )
            self._blocked_array.setMinMaxTrackingEnabled(True)
        
        # FIXME: Hard-coded as uint8 for now.
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = numpy.uint8
        
        self.maxLabel.meta.shape = (1,)
        self.maxLabel.meta.dtype = self.Output.meta.dtype
        
        self.nonzeroBlocks.meta.shape = (1,)
        self.nonzeroBlocks.meta.dtype = object

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.deleteLabel:
            delete_label = self.deleteLabel.value
            if delete_label == self._previous_delete_label:
                return
            self._previous_delete_label = delete_label
            if delete_label <= 0:
                return
            # Deleted label is converted to 0,
            # All higher labels are shifted down by 1 to keep the labels consecutive
            relabeling = numpy.asarray( range( self._max_label+1 ), numpy.uint8 )
            relabeling = numpy.where( relabeling == delete_label, 0, relabeling )
            relabeling = numpy.where( relabeling > delete_label, relabeling-1, relabeling )
            self._blocked_array.applyRelabeling( relabeling.view( vigra.VigraArray ) )

            # Every pixel was (potentially) touched.
            self.Output.setDirty()
            
            min_label, max_label = self._blocked_array.minMax()
            if max_label != self._max_label:
                self._max_label = max_label
                self.maxLabel.setDirty()

    def execute(self, slot, subindex, roi, result):
        if slot == self.Output:
            return self._executeOutput( roi, result )
        elif slot == self.nonzeroBlocks:
            return self._executeNonzeroBlocks( roi, result )
        elif slot == self.maxLabel:
            return self._executeMaxLabel( result )
        assert False, "Slot not supported: {}".format( slot.name )
    
    def _executeOutput(self, roi, result):
        self._blocked_array.readSubarray( roi.start, roi.stop, result )
        return result
    
    def _executeNonzeroBlocks(self, roi, result):
        # FIXME: We ignore roi and return the entire blocklist, always
        total_roi = roiFromShape( self.Input.meta.shape )
        block_starts, block_stops = self._blocked_array.blocks( *total_roi )
        slicings = []
        for start, stop in zip(block_starts, block_stops):
            slicings.append( roiToSlice( start, stop ) )
        result[0] = slicings
        return result

    def _executeMaxLabel(self, result):
        result[0] = self._max_label
        return result

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.Input
        assert value.shape == tuple(roi.stop - roi.start)
        self._blocked_array.writeSubarrayNonzero( tuple(roi.start), tuple(roi.stop), value, self.eraser.value )
        min_label, max_label = self._blocked_array.minMax()
        
        self.Output.setDirty( roi )
        if max_label != self._max_label:
            self._max_label = max_label
            self.maxLabel.setDirty()

def create_blockedarray( blockshape, dtype ):
    dimension = len(blockshape)
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

    return lookup[dtype][dimension]( blockshape )
