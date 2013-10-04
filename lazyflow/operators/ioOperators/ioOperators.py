#Python
from functools import partial
import os
import math
import logging
import glob
import copy
from itertools import product, chain
from collections import deque, OrderedDict
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)

from lazyflow.utility.bigRequestStreamer import BigRequestStreamer
from lazyflow.roi import roiFromShape

#SciPy
import vigra,numpy,h5py

#lazyflow
from lazyflow.graph import OrderedSignal, Operator, OutputSlot, InputSlot
from lazyflow.roi import roiToSlice

class OpStackLoader(Operator):
    """Imports an image stack.

    Note: This operator does NOT cache the images, so direct access
          via the execute() function is very inefficient, especially
          through the Z-axis. Typically, you'll want to connect this
          operator to a cache whose block size is large in the X-Y
          plane.

    :param globstring: A glob string as defined by the glob module. We
        also support the following special extension to globstring
        syntax: A single string can hold a *list* of globstrings. Each
        separate globstring in the list is separated by two forward
        slashes (//). For, example,

            '/a/b/c.txt///d/e/f.txt//../g/i/h.txt'

        is parsed as

            ['/a/b/c.txt', '/d/e/f.txt', '../g/i/h.txt']

    """
    name = "Image Stack Reader"
    category = "Input"

    inputSlots = [InputSlot("globstring", stype = "string")]
    outputSlots = [OutputSlot("stack")]

    class FileOpenError( Exception ):
        def __init__(self, filename):
            self.filename = filename
            self.msg = "Unable to open file: {}".format(filename)
            super(OpStackLoader.FileOpenError, self).__init__( self.msg )

    def setupOutputs(self):
        self.fileNameList = []
        globStrings = self.globstring.value

        # Parse list into separate globstrings and combine them
        for globString in sorted(globStrings.split("//")):
            self.fileNameList += sorted(glob.glob(globString))

        num_files = len(self.fileNameList)
        if len(self.fileNameList) == 0:
            self.stack.meta.NOTREADY = True
            return
        try:
            self.info = vigra.impex.ImageInfo(self.fileNameList[0])
            self.slices_per_file = vigra.impex.numberImages(self.fileNameList[0])
        except RuntimeError:
            raise OpStackLoader.FileOpenError(self.fileNameList[0])

        slice_shape = self.info.getShape()
        X, Y, C = slice_shape
        if self.slices_per_file == 1:
            # If this is a stack of 2D images, we assume xy slices stacked along z
            Z = num_files
            shape = (Z, Y, X, C)
            axistags = vigra.defaultAxistags('zyxc')
        else:
            # If it's a stack of 3D volumes, we assume xyz blocks stacked along t
            T = num_files
            Z = self.slices_per_file
            shape = (T, Z, Y, X, C)
            axistags = vigra.defaultAxistags('tzyxc')
            
        self.stack.meta.shape = shape
        self.stack.meta.axistags = axistags
        self.stack.meta.dtype = self.info.getDtype()

    def propagateDirty(self, slot, subindex, roi):
        assert slot == self.globstring
        # Any change to the globstring means our entire output is dirty.
        self.stack.setDirty()

    def execute(self, slot, subindex, roi, result):
        if len(self.stack.meta.shape) == 4:
            return self._execute_4d( roi, result )
        elif len(self.stack.meta.shape) == 5:
            return self._execute_5d( roi, result )
        else:
            assert False, "Unexpected output shape: {}".format( self.stack.meta.shape )
        
    def _execute_4d(self, roi, result):
        traceLogger.debug("OpStackLoader: Execute for: " + str(roi))
        # roi is in zyxc order.
        z_start, y_start, x_start, c_start = roi.start
        z_stop, y_stop, x_stop, c_stop = roi.stop

        # Copy each z-slice one at a time.
        for result_z, fileName in enumerate(self.fileNameList[z_start:z_stop]):
            traceLogger.debug( "Reading image: {}".format(fileName) )
            if self.info.getShape() != vigra.impex.ImageInfo(fileName).getShape():
                raise RuntimeError('not all files have the same shape')
            if self.slices_per_file != vigra.impex.numberImages(self.fileNameList[0]):
                raise RuntimeError("Not all files have the same number of slices")

            result[result_z,:,:,:] = vigra.impex.readImage(fileName)[x_start:x_stop,
                                                                     y_start:y_stop,
                                                                     c_start:c_stop].withAxes( *'yxc' )
        return result

    def _execute_5d(self, roi, result):
        # roi is in tzyxc order.
        t_start, z_start, y_start, x_start, c_start = roi.start
        t_stop, z_stop, y_stop, x_stop, c_stop = roi.stop

        # Use *enumerated* range to get global t coords and result t coords
        for result_t, t in enumerate( range( t_start, t_stop ) ):
            file_name = self.fileNameList[t]
            for result_z, z in enumerate( range( z_start, z_stop ) ):
                img = vigra.readImage( file_name, index=z )
                result[result_t, result_z, :, :, :] = img[ x_start:x_stop,
                                                           y_start:y_stop,
                                                           c_start:c_stop ].withAxes( *'yxc' )
        return result
        


class OpStackWriter(Operator):
    name = "Stack File Writer"
    category = "Output"

    Input = InputSlot() # The last two non-singleton axes (except 'c') are the axes of the slices.
                        # Re-order the axes yourself if you want an alternative slicing direction

    FilepathPattern = InputSlot() # A complete filepath including a {slice_index} member and a valid file extension.
    SliceIndexOffset = InputSlot(value=0) # Added to the {slice_index} in the export filename.

    def __init__(self, *args, **kwargs):
        super(OpStackWriter, self).__init__(*args, **kwargs)
        self.progressSignal = OrderedSignal()

    def run_export(self):
        """
        Request the volume in slices (running in parallel), and write each slice to a separate image.
        """
        # Make the directory first if necessary
        export_dir = os.path.split(self.FilepathPattern.value)[0]
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Sliceshape is the same as the input shape, except for the sliced dimension
        tagged_sliceshape = self.Input.meta.getTaggedShape()
        tagged_sliceshape[self._volume_axes[0]] = 1
        slice_shape = (tagged_sliceshape.values())

        # Use a request streamer to automatically request a constant batch of 4 active requests.
        streamer = BigRequestStreamer( self.Input,
                                       roiFromShape( self.Input.meta.shape ),
                                       slice_shape,
                                       batchSize=4 )

        # Write the slices as they come in (possibly out-of-order, but probably not.        
        streamer.resultSignal.subscribe( self._write_slice )
        streamer.progressSignal.subscribe( self.progressSignal )

        logger.debug("Starting Stack Export with slicing shape: {}".format( slice_shape ))
        streamer.execute()

    def setupOutputs(self):
        # If stacking XY images in Z-steps,
        #  then self._volume_axes = 'zxy'
        self._volume_axes = self.get_nonsingleton_axes()
        step_axis = self._volume_axes[0]
        max_slice = self.SliceIndexOffset.value + self.Input.meta.getTaggedShape()[step_axis]
        self._max_slice_digits = int(math.ceil(math.log10(max_slice+1)))

        # Check for errors
        assert len(self._volume_axes) == 3 or len(self._volume_axes) == 4 and 'c' in self._volume_axes[1:], \
            "Exported stacks must have exactly 3 non-singleton dimensions (other than the channel dimension).  "\
            "Your stack dimensions are: {}".format( self.Input.meta.getTaggedShape() )

        # Test to make sure the filepath pattern includes slice index field        
        filepath_pattern = self.FilepathPattern.value
        assert '123456789' in filepath_pattern.format( slice_index=123456789 ), \
            "Output filepath pattern must contain the '{slice_index}' field for formatting.\n"\
            "Your format was: {}".format( filepath_pattern )

    # No output slots...
    def execute(self, slot, subindex, roi, result): pass 
    def propagateDirty(self, slot, subindex, roi): pass

    def get_nonsingleton_axes(self):
        return self.get_nonsingleton_axes_for_tagged_shape( self.Input.meta.getTaggedShape() )

    @classmethod
    def get_nonsingleton_axes_for_tagged_shape(self, tagged_shape):
        # Find the non-singleton axes.
        # The first non-singleton axis is the step axis.
        # The last 2 non-channel non-singleton axes will be the axes of the slices.
        tagged_items = tagged_shape.items()
        filtered_items = filter( lambda (k, v): v > 1, tagged_items )
        filtered_axes = zip( *filtered_items )[0]
        return filtered_axes

    def _write_slice(self, roi, slice_data):
        """
        Write the data from the given roi into a slice image.
        """
        step_axis = self._volume_axes[0]
        input_axes = self.Input.meta.getAxisKeys()
        tagged_roi = OrderedDict( zip( input_axes, zip( *roi ) ) )
        # e.g. tagged_roi={ 'x':(0,1), 'y':(3,4), 'z':(10,20) }
        assert tagged_roi[step_axis][1] - tagged_roi[step_axis][0] == 1,\
            "Expected roi to be a single slice."
        slice_index = tagged_roi[step_axis][0] + self.SliceIndexOffset.value
        filepattern = self.FilepathPattern.value

        # If the user didn't provide custom formatting for the slice field,
        #  auto-format to include zero-padding
        if '{slice_index}' in filepattern:
            filepattern = filepattern.format( slice_index='{' + 'slice_index:0{}'.format(self._max_slice_digits) + '}' )        
        formatted_path = filepattern.format( slice_index=slice_index )
        
        squeezed_data = slice_data.squeeze()
        squeezed_data = vigra.taggedView(squeezed_data, vigra.defaultAxistags("".join(self._volume_axes[1:])))
        assert len(squeezed_data.shape) == len(self._volume_axes)-1

        #logger.debug( "Writing slice image for roi: {}".format( roi ) )
        logger.debug("Writing slice: {}".format(formatted_path) )
        vigra.impex.writeImage( squeezed_data, formatted_path )

class OpStackToH5Writer(Operator):
    name = "OpStackToH5Writer"
    category = "IO"

    GlobString = InputSlot(stype='globstring')
    hdf5Group = InputSlot(stype='object')
    hdf5Path  = InputSlot(stype='string')

    # Requesting the output induces the copy from stack to h5 file.
    WriteImage = OutputSlot(stype='bool')

    def __init__(self, *args, **kwargs):
        super(OpStackToH5Writer, self).__init__(*args, **kwargs)
        self.progressSignal = OrderedSignal()
        self.opStackLoader = OpStackLoader(parent=self)
        self.opStackLoader.globstring.connect( self.GlobString )

    def setupOutputs(self):
        self.WriteImage.meta.shape = (1,)
        self.WriteImage.meta.dtype = object

    def propagateDirty(self, slot, subindex, roi):
        # Any change to our inputs means we're dirty
        assert slot == self.GlobString or slot == self.hdf5Group or slot == self.hdf5Path
        self.WriteImage.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        # Copy the data image-by-image
        stackTags = self.opStackLoader.stack.meta.axistags
        zAxis = stackTags.index('z')
        dataShape=self.opStackLoader.stack.meta.shape
        numImages = self.opStackLoader.stack.meta.shape[zAxis]
        axistags = self.opStackLoader.stack.meta.axistags
        dtype = self.opStackLoader.stack.meta.dtype
        if type(dtype) is numpy.dtype:
            # Make sure we're dealing with a type (e.g. numpy.float64),
            #  not a numpy.dtype
            dtype = dtype.type
        
        index_ = axistags.index('c')
        if index_ >= len(dataShape):
            numChannels = 1
        else:
            numChannels = dataShape[ index_]
        
        # Set up our chunk shape: Aim for a cube that's roughly 300k in size
        dtypeBytes = dtype().nbytes
        cubeDim = math.pow( 300000 / (numChannels * dtypeBytes), (1/3.0) )
        cubeDim = int(cubeDim)

        chunkDims = {}
        chunkDims['t'] = 1
        chunkDims['x'] = cubeDim
        chunkDims['y'] = cubeDim
        chunkDims['z'] = cubeDim
        chunkDims['c'] = numChannels

        # h5py guide to chunking says chunks of 300k or less "work best"
        assert chunkDims['x'] * chunkDims['y'] * chunkDims['z'] * numChannels * dtypeBytes  <= 300000
        
        chunkShape = ()
        for i in range( len(dataShape) ):
            axisKey = axistags[i].key
            # Chunk shape can't be larger than the data shape
            chunkShape += ( min( chunkDims[axisKey], dataShape[i] ), )
        
        # Create the dataset
        internalPath = self.hdf5Path.value
        internalPath = internalPath.replace('\\', '/') # Windows fix
        group = self.hdf5Group.value
        if internalPath in group:
            del group[internalPath]
        
        data = group.create_dataset(internalPath,
                                    #compression='gzip',
                                    #compression_opts=4,
                                    shape=dataShape,
                                    dtype=dtype,
                                    chunks=chunkShape)
        # Now copy each image
        self.progressSignal(0)
        
        for z in range(numImages):
            # Ask for an entire z-slice (exactly one whole image from the stack)
            slicing = [slice(None)] * len(stackTags)
            slicing[zAxis] = slice(z, z+1)
            data[tuple(slicing)] = self.opStackLoader.stack[slicing].wait()
            self.progressSignal( z*100 / numImages )

        data.attrs['axistags'] = axistags.toJSON()

        # We're done
        result[...] = True

        self.progressSignal(100)

        return result

class OpH5WriterBigDataset(Operator):
    name = "H5 File Writer BigDataset"
    category = "Output"

    inputSlots = [InputSlot("hdf5File"), # Must be an already-open hdf5File (or group) for writing to
                  InputSlot("hdf5Path", stype = "string"),
                  InputSlot("Image"),
                  InputSlot("CompressionEnabled", value=True)]

    outputSlots = [OutputSlot("WriteImage")]

    loggingName = __name__ + ".OpH5WriterBigDataset"
    logger = logging.getLogger(loggingName)
    traceLogger = logging.getLogger("TRACE." + loggingName)

    def __init__(self, *args, **kwargs):
        super(OpH5WriterBigDataset, self).__init__(*args, **kwargs)
        self.progressSignal = OrderedSignal()

    def setupOutputs(self):
        self.outputs["WriteImage"].meta.shape = (1,)
        self.outputs["WriteImage"].meta.dtype = object

        self.f = self.inputs["hdf5File"].value
        hdf5Path = self.inputs["hdf5Path"].value
        
        # On windows, there may be backslashes.
        hdf5Path = hdf5Path.replace('\\', '/')

        hdf5GroupName, datasetName = os.path.split(hdf5Path)
        if hdf5GroupName == "":
            g = self.f
        else:
            if hdf5GroupName in self.f:
                g = self.f[hdf5GroupName]
            else:
                g = self.f.create_group(hdf5GroupName)

        dataShape=self.Image.meta.shape
        taggedShape = self.Image.meta.getTaggedShape()
        dtype = self.Image.meta.dtype
        if type(dtype) is numpy.dtype:
            # Make sure we're dealing with a type (e.g. numpy.float64),
            #  not a numpy.dtype
            dtype = dtype.type

        numChannels = 1
        if 'c' in taggedShape:
            numChannels = taggedShape['c']

        # Set up our chunk shape: Aim for a cube that's roughly 300k in size
        dtypeBytes = dtype().nbytes
        cubeDim = math.pow( 300000 / (numChannels * dtypeBytes), (1/3.0) )
        cubeDim = int(cubeDim)

        chunkDims = {}
        chunkDims['t'] = 1
        chunkDims['x'] = cubeDim
        chunkDims['y'] = cubeDim
        chunkDims['z'] = cubeDim
        chunkDims['c'] = numChannels
        
        # h5py guide to chunking says chunks of 300k or less "work best"
        assert chunkDims['x'] * chunkDims['y'] * chunkDims['z'] * numChannels * dtypeBytes  <= 300000

        chunkShape = ()
        for i in range( len(dataShape) ):
            axisKey = self.Image.meta.axistags[i].key
            # Chunk shape can't be larger than the data shape
            chunkShape += ( min( chunkDims[axisKey], dataShape[i] ), )

        self.chunkShape = chunkShape
        if datasetName in g.keys():
            del g[datasetName]
        kwargs = { 'shape' : dataShape, 'dtype' : dtype, 'chunks' : self.chunkShape }
        if self.CompressionEnabled.value:
            kwargs['compression'] = 'gzip' # <-- Would be nice to use lzf compression here, but that is h5py-specific.
            kwargs['compression_opts'] = 1 # <-- Optimize for speed, not disk space.
        self.d=g.create_dataset(datasetName, **kwargs)

        if self.Image.meta.drange is not None:
            self.d.attrs['drange'] = self.Image.meta.drange

    def execute(self, slot, subindex, rroi, result):
        self.progressSignal(0)
        
        slicings=self.computeRequestSlicings()
        numSlicings = len(slicings)

        self.logger.debug( "Dividing work into {} pieces".format( len(slicings) ) )

        # Throttle: Only allow 10 outstanding requests at a time.
        # Otherwise, the whole set of requests can be outstanding and use up ridiculous amounts of memory.        
        activeRequests = deque()
        activeSlicings = deque()
        # Start by activating 10 requests 
        for i in range( min(10, len(slicings)) ):
            s = slicings.pop()
            activeSlicings.append(s)
            self.logger.debug( "Creating request for slicing {}".format(s) )
            activeRequests.append( self.inputs["Image"][s] )
        
        counter = 0

        while len(activeRequests) > 0:
            # Wait for a request to finish
            req = activeRequests.popleft()
            s=activeSlicings.popleft()
            data = req.wait()
            if data.flags.c_contiguous:
                self.d.write_direct(data.view(numpy.ndarray), dest_sel=s)
            else:
                self.d[s] = data
            
            req.clean() # Discard the data in the request and allow its children to be garbage collected.

            if len(slicings) > 0:
                # Create a new active request
                s = slicings.pop()
                activeSlicings.append(s)
                activeRequests.append( self.inputs["Image"][s] )
            
            # Since requests finish in an arbitrary order (but we always block for them in the same order),
            # this progress feedback will not be smooth.  It's the best we can do for now.
            self.progressSignal( 100*counter/numSlicings )
            self.logger.debug( "request {} out of {} executed".format( counter, numSlicings ) )
            counter += 1

        # Save the axistags as a dataset attribute
        self.d.attrs['axistags'] = self.Image.meta.axistags.toJSON()

        # We're finished.
        result[0] = True

        self.progressSignal(100)

    def computeRequestSlicings(self):
        #TODO: reimplement the request better
        shape=numpy.asarray(self.inputs['Image'].meta.shape)

        chunkShape = numpy.asarray(self.chunkShape)

        # Choose a request shape that is a multiple of the chunk shape
        axistags = self.Image.meta.axistags
        multipliers = { 'x':5, 'y':5, 'z':5, 't':1, 'c':100 } # For most problems, there is little advantage to breaking up the channels.
        multiplier = [multipliers[tag.key] for tag in axistags ]
        shift = chunkShape * numpy.array(multiplier)
        shift=numpy.minimum(shift,shape)
        start=numpy.asarray([0]*len(shape))

        stop=shift
        reqList=[]

        #shape = shape - (numpy.mod(numpy.asarray(shape),
        #                  shift))

        for indices in product(*[range(0, stop, step)
                        for stop,step in zip(shape, shift)]):

            start=numpy.asarray(indices)
            stop=numpy.minimum(start+shift,shape)
            reqList.append(roiToSlice(start,stop))
        return reqList

    def propagateDirty(self, slot, subindex, roi):
        # The output from this operator isn't generally connected to other operators.
        # If someone is using it that way, we'll assume that the user wants to know that 
        #  the input image has become dirty and may need to be written to disk again.
        self.WriteImage.setDirty(slice(None))

if __name__ == '__main__':
    from lazyflow.graph import Graph
    import h5py
    import sys

    traceLogger.addHandler(logging.StreamHandler(sys.stdout))
    traceLogger.setLevel(logging.DEBUG)
    traceLogger.debug("HELLO")

    f = h5py.File('/tmp/flyem_sample_stack.h5')
    internalPath = 'volume/data'

    # OpStackToH5Writer
    graph = Graph()
    opStackToH5 = OpStackToH5Writer()
    opStackToH5.GlobString.setValue('/tmp/flyem_sample_stack/*.png')
    opStackToH5.hdf5Group.setValue(f)
    opStackToH5.hdf5Path.setValue(internalPath)

    success = opStackToH5.WriteImage.value
    assert success

