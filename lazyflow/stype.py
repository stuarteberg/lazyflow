import numpy, vigra
import warnings

from roi import roiToSlice
from lazyflow.utility.helpers import warn_deprecated

import h5py

class SlotType( object ):
    def __init__( self, slot):
        self.slot = slot

    def allocateDestination( self, roi ):
        pass

    def writeIntoDestination( self, destination, value, roi ):
        pass

    def isCompatible(self, value):
        """
        Slot types must implement this method.

        this method should check wether the supplied value
        is compatible with this type trait and should return true
        if this is the case.
        """
        pass

    def setupMetaForValue(self, value):
        """
        Slot types must implement this method.

        this method should extract valuable meta information
        from the provided value and set up the self.slot.meta
        MetaDict accordingly
        """
        pass


    def isConfigured(self):
        """
        Slot types must implement this method.

        it should analyse the .meta property of the slot
        and return wether the neccessary meta information
        is available.
        """
        return False

    def connect(self,slot):
        pass


    def copy_data(self, dst, src):
        """
        Slot types must implement this method

        if should copy all the data from src to dst.
        src and dst must be of the kind which is return by an operator with a slot
        of this type.

        usually dst, is the destination area specified by somebody, and src is the result
        that an operator returns.

        """
        pass

    def check_result_valid(self, roi, result):
        """
        Slot types must implement this method

        it must check wether the provided result is compatible with the user specified roi
        """
        return True


class ArrayLike( SlotType ):
    def allocateDestination( self, roi ):
        shape = roi.stop - roi.start if roi else self.slot.meta.shape
        storage = numpy.ndarray(shape, dtype=self.slot.meta.dtype)
        # if axistags is True:
        #     storage = vigra.VigraArray(storage, storage.dtype, axistags = copy.copy(s))elf.axistags))
        #     #storage = storage.view(vigra.VigraArray)
        #     #storage.axistags = copy.copy(self.axistags)
        return storage

    def writeIntoDestination( self, destination, value, roi ):
        if destination is not None:
            if not isinstance(destination, list):
                assert(roi.dim == destination.ndim), "%r ndim=%r, shape=%r" % (roi.toSlice(), destination.ndim, destination.shape)
            sl = roiToSlice(roi.start, roi.stop)
            try:
                destination[:] = value[sl]
            except TypeError:
                # FIXME: This warning used to be triggered by a corner case that could be encountered by "value slots".
                #        The behavior here isn't truly deprecated.  But we need a better solution for lazyflow 2.0.
                # See ilastik/ilastik#704
                #warn_deprecated("old style slot encountered: non array-like value set -> change SlotType from ArrayLike to proper SlotType")
                destination[:] = value
        else:
            sl = roiToSlice(roi.start, roi.stop)
            try:
                destination = value[sl]
            except:
                # FIXME: This warning used to be triggered by a corner case that could be encountered by "value slots".
                #        The behavior here isn't truly deprecated.  But we need a better solution for lazyflow 2.0.
                # See ilastik/ilastik#704
                #warn_deprecated("old style slot encountered: non array-like value set -> change SlotType from ArrayLike to proper SlotType")
                destination = [value]

            if type(destination) == numpy.ndarray and destination.shape == ():
                # This is necessary because numpy types will not be caught in the except statement above.
                # They don't throw when used with __getitem__
                # e.g. try this:
                # x = np.int64(5)
                # assert type(x[()]) == np.ndarray and x[()].shape == ()

                # FIXME: This warning used to be triggered by a corner case that could be encountered by "value slots".
                #        The behavior here isn't truly deprecated.  But we need a better solution for lazyflow 2.0.
                # See ilastik/ilastik#704
                #warn_deprecated("old style slot encountered: non array-like value set -> change SlotType from ArrayLike to proper SlotType")
                destination = [value]
        return destination



    def isCompatible(self, value):
        warnings.warn("ArrayLike.isCompatible: FIXME here")
        return True


    def setupMetaForValue(self, value):
        if isinstance(value, numpy.ndarray):
            self.slot.meta.shape = value.shape
            self.slot.meta.dtype = value.dtype.type
            if hasattr(value,"axistags"):
                self.slot.meta.axistags = value.axistags
        else:
            self.slot.meta.shape = (1,)
            if isinstance(value, int) or \
               isinstance(value, float) or \
               isinstance(value, numpy.floating) or \
               isinstance(value, numpy.integer) or \
               isinstance(value, numpy.bool_):
                self.slot.meta.dtype = type(value)
            else:
                self.slot.meta.dtype = object

    def isConfigured(self):
        meta = self.slot.meta
        if meta.shape is not None and meta.dtype is not None:
            return True
        else:
            return False


    def copy_data(self, dst, src):
        dst[...] = src[...]

    def check_result_valid(self, roi, result):
        if isinstance(result, numpy.ndarray):
            assert len(roi.start) == result.ndim, "check_result_valid: result has wrong number of dimensions (%d instead of %d)" % (result.ndim, len(roi.start))
            for d in range(result.ndim):
                s = roi.stop[d] - roi.start[d]
                assert result.shape[d] == s, "check_result_valid: result has wrong shape (%d instead of %d) for dimension %d" % (result.shape[d], s, d)
        elif isinstance(result, list):
            s = roi.stop[0] - roi.start[0]
            assert len(result) == s, "check_result_valid: result has wrong shape (%d instead of %d) for dimension %d" % (result.shape[d], s, d)
        elif isinstance(result, h5py.Group):
            # FIXME: this is a hack. the slot
            # OpCompressedCache.OutputHdf5 is not really array-like,
            # because it expects destinations of type h5py.Group.
            pass
        else:
            assert False, "check_result_valid: result type is not supported"

class Opaque(SlotType):
    def allocateDestination(self, roi):
        return None

    def writeIntoDestination(self, destination, value, roi):
        # FIXME: To be similar to the ArrayLike stype, we should return the value wrapped in a list:
        # destination = [value]
        # ...which is also why, for 'Opaque' slots, slot.value is not the same as slot[:].wait().
        # But for now, I'm leaving this alone.
        # "Fixing" this breaks many parts of the object classification workflow 
        #  that are written to use the existing API, awkward as it is.
        # See also: ilastik/ilastik#704, and ilastik/ilastik#705.
        return value

    def isCompatible(self, value):
        return True

    def setupMetaForValue(self, value):
        self.slot.meta.shape = (1,)
        self.slot.meta.dtype = object
        self.slot.meta.axistags = vigra.defaultAxistags(1)

    def isConfigured(self):
        return True
    
    def copy_data(self, dst, src):
        raise("Not Implemented")
