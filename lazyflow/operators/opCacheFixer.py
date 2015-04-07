import numpy
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiFromShape

class OpCacheFixer(Operator):
    Input = InputSlot(allow_mask=True)
    fixAtCurrent = InputSlot(value=False)
    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super(OpCacheFixer, self).__init__(*args, **kwargs)
        self._fixed = False
        self._fixed_dirty_roi = None

    def setupOutputs(self):
        self.Output.meta.assignFrom( self.Input.meta )

    def execute(self, slot, subindex, roi, result):
        if self._fixed:
            # The downstream user doesn't know he's getting fake data.
            # When we become "unfixed", we need to tell him.
            self._expand_fixed_dirty_roi( (roi.start, roi.stop) )
            result[:] = 0
        else:
            self.Input(roi.start, roi.stop).writeInto(result).wait()
        
    def propagateDirty(self, slot, subindex, roi):
        if slot is self.fixAtCurrent:
            # If we're becoming UN-fixed, send out a big dirty notification
            if ( self._fixed and not self.fixAtCurrent.value and
                 self._fixed_dirty_roi and (self._fixed_dirty_roi[1] - self._fixed_dirty_roi[0] > 0).all() ):
                self.Output.setDirty( *self._fixed_dirty_roi )
                self._fixed_dirty_roi = None
            self._fixed = self.fixAtCurrent.value
        elif slot is self.Input:
            if self._fixed:
                # We can't propagate this downstream,
                #  but we need to remember that it was marked dirty.
                # Expand our dirty bounding box.
                self._expand_fixed_dirty_roi( (roi.start, roi.stop) )
            else:
                self.Output.setDirty(roi.start, roi.stop)

    def _init_fixed_dirty_roi(self):
        # Intentionally flipped: nothing is dirty at first.
        entire_roi = roiFromShape(self.Input.meta.shape)
        self._fixed_dirty_roi = (entire_roi[1], entire_roi[0])

    def _expand_fixed_dirty_roi(self, roi):
        if self._fixed_dirty_roi is None:
            self._init_fixed_dirty_roi()
        start, stop = self._fixed_dirty_roi
        start = numpy.minimum(start, roi[0])
        stop = numpy.maximum(stop, roi[1])
        self._fixed_dirty_roi = (start, stop)
        