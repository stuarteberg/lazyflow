from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpImageReader, OpBlockedArrayCache
from opStreamingHdf5Reader import OpStreamingHdf5Reader
from opNpyFileReader import OpNpyFileReader
from lazyflow.operators.ioOperators import OpStackLoader, OpBlockwiseFilesetReader, OpRESTfulBlockwiseFilesetReader
from lazyflow.utility.jsonConfig import JsonConfigParser

import h5py
import vigra
import os
import logging

class OpInputDataReader(Operator):
    """
    This operator can read input data of any supported type.
    The data format is determined from the file extension.
    """
    name = "OpInputDataReader"
    category = "Input"

    h5Exts = ['h5', 'hdf5', 'ilp']
    npyExts = ['npy']
    blockwiseExts = ['json']
    vigraImpexExts = vigra.impex.listExtensions().split()
    SupportedExtensions = h5Exts + npyExts + vigraImpexExts + blockwiseExts

    # FilePath is inspected to determine data type.
    # For hdf5 files, append the internal path to the filepath,
    #  e.g. /mydir/myfile.h5/internal/path/to/dataset
    # For stacks, provide a globstring, e.g. /mydir/input*.png
    # Other types are determined via file extension
    WorkingDirectory = InputSlot(stype='filestring', optional=True)
    FilePath = InputSlot(stype='filestring')
    Output = OutputSlot()
    
    loggingName = __name__ + ".OpInputDataReader"
    logger = logging.getLogger(loggingName)

    class DatasetReadError(Exception):
        pass

    def __init__(self, *args, **kwargs):
        super(OpInputDataReader, self).__init__(*args, **kwargs)
        self.internalOperator = None
        self.internalOutput = None
        self._file = None

    def cleanUp(self):
        super(OpInputDataReader, self).cleanUp()
        if self._file is not None:
            self._file.close()
            self._file = None

    def setupOutputs(self):
        """
        Inspect the file name and instantiate and connect an internal operator of the appropriate type.
        TODO: Handle datasets of non-standard (non-5d) dimensions.
        """
        filePath = self.FilePath.value
        assert type(filePath) == str, "Error: filePath is not of type str.  It's of type {}".format(type(filePath))

        # Does this look like a relative path?
        useRelativePath = not os.path.isabs(filePath)

        if useRelativePath:
            # If using a relative path, we need both inputs before proceeding
            if not self.WorkingDirectory.ready():
                return
            else:
                # Convert this relative path into an absolute path
                filePath = os.path.normpath(os.path.join(self.WorkingDirectory.value, filePath)).replace('\\','/')

        # Clean up before reconfiguring
        if self.internalOperator is not None:
            self.Output.disconnect()
            self.internalOperator.cleanUp()
            self.internalOperator = None
            self.internalOutput = None
        if self._file is not None:
            self._file.close()

        openFuncs = [ self._attemptOpenAsStack,
                      self._attemptOpenAsHdf5,
                      self._attemptOpenAsNpy,
                      self._attemptOpenAsBlockwiseFileset,
                      self._attemptOpenAsRESTfulBlockwiseFileset,
                      self._attemptOpenWithVigraImpex ]

        # Try every method of opening the file until one works.
        iterFunc = openFuncs.__iter__()
        while self.internalOperator is None:
            try:
                openFunc = iterFunc.next()
            except StopIteration:
                break
            self.internalOperator, self.internalOutput = openFunc(filePath)

        if self.internalOutput is None:
            raise RuntimeError("Can't read " + filePath + " because it has an unrecognized format.")

        # Directly connect our own output to the internal output
        self.Output.connect( self.internalOutput )
    
    def _attemptOpenAsStack(self, filePath):
        if '*' in filePath:
            stackReader = OpStackLoader(parent=self)
            stackReader.globstring.setValue(filePath)
            return (stackReader, stackReader.stack)
        else:
            return (None, None)

    def _attemptOpenAsHdf5(self, filePath):
        # Check for an hdf5 extension
        h5Exts = OpInputDataReader.h5Exts + ['ilp']
        h5Exts = ['.' + ex for ex in h5Exts]
        ext = None
        for x in h5Exts:
            if x in filePath:
                ext = x

        if ext is None:
            return (None, None)

        externalPath = filePath.split(ext)[0] + ext
        internalPath = filePath.split(ext)[1]

        if not os.path.exists(externalPath):
            raise OpInputDataReader.DatasetReadError("Input file does not exist: " + externalPath)

        # Open the h5 file in read-only mode
        try:
            h5File = h5py.File(externalPath, 'r')
        except Exception as e:
            msg = "Unable to open HDF5 File: {}".format( externalPath )
            if hasattr(e, 'message'):
                msg += e.message
            raise OpInputDataReader.DatasetReadError( msg )
        self._file = h5File

        h5Reader = OpStreamingHdf5Reader(parent=self)
        h5Reader.Hdf5File.setValue(h5File)

        # Can't set the internal path yet if we don't have one
        assert internalPath != '', \
            "When using hdf5, you must append the hdf5 internal path to the "\
            "data set to your filename, e.g. myfile.h5/volume/data  "\
            "No internal path provided for dataset in file: {}".format( externalPath )

        try:
            h5Reader.InternalPath.setValue(internalPath)
        except OpStreamingHdf5Reader.DatasetReadError as e:
            msg = "Error reading HDF5 File: {}".format(externalPath)
            msg += e.msg
            raise OpInputDataReader.DatasetReadError( msg )

        return (h5Reader, h5Reader.OutputImage)

    def _attemptOpenAsNpy(self, filePath):
        fileExtension = os.path.splitext(filePath)[1].lower()
        fileExtension = fileExtension.lstrip('.') # Remove leading dot

        # Check for numpy extension
        if fileExtension not in OpInputDataReader.npyExts:
            return (None, None)
        else:
            try:
                # Create an internal operator
                npyReader = OpNpyFileReader(parent=self)
                npyReader.FileName.setValue(filePath)
                return (npyReader, npyReader.Output)
            except OpNpyFileReader.DatasetReadError as e:
                raise OpInputDataReader.DatasetReadError( *e.args )

    def _attemptOpenAsBlockwiseFileset(self, filePath):
        fileExtension = os.path.splitext(filePath)[1].lower()
        fileExtension = fileExtension.lstrip('.') # Remove leading dot

        if fileExtension in OpInputDataReader.blockwiseExts:
            opReader = OpBlockwiseFilesetReader(parent=self)
            try:
                # This will raise a SchemaError if this is the wrong type of json config.
                opReader.DescriptionFilePath.setValue( filePath )
                return (opReader, opReader.Output)
            except JsonConfigParser.SchemaError:
                opReader.cleanUp()
            except OpBlockwiseFilesetReader.MissingDatasetError as e:
                raise OpInputDataReader.DatasetReadError(*e.args)
        return (None, None)

    def _attemptOpenAsRESTfulBlockwiseFileset(self, filePath):
        fileExtension = os.path.splitext(filePath)[1].lower()
        fileExtension = fileExtension.lstrip('.') # Remove leading dot

        if fileExtension in OpInputDataReader.blockwiseExts:
            opReader = OpRESTfulBlockwiseFilesetReader(parent=self)
            try:
                # This will raise a SchemaError if this is the wrong type of json config.
                opReader.DescriptionFilePath.setValue( filePath )
                return (opReader, opReader.Output)
            except JsonConfigParser.SchemaError:
                opReader.cleanUp()
            except OpRESTfulBlockwiseFilesetReader.MissingDatasetError as e:
                raise OpInputDataReader.DatasetReadError(*e.args)
        return (None, None)

    def _attemptOpenWithVigraImpex(self, filePath):
        fileExtension = os.path.splitext(filePath)[1].lower()
        fileExtension = fileExtension.lstrip('.') # Remove leading dot

        if fileExtension not in OpInputDataReader.vigraImpexExts:
            return (None, None)

        if not os.path.exists(filePath):
            raise OpInputDataReader.DatasetReadError("Input file does not exist: " + filePath)

        vigraReader = OpImageReader(parent=self)
        vigraReader.Filename.setValue(filePath)

        # Cache the image instead of reading the hard disk for every access.
        imageCache = OpBlockedArrayCache(parent=self)
        imageCache.Input.connect(vigraReader.Image)
        
        # 2D: Just one block for the whole image
        cacheBlockShape = vigraReader.Image.meta.shape
        
        taggedShape = vigraReader.Image.meta.getTaggedShape()
        if 'z' in taggedShape.keys():
            # 3D: blocksize is one slice.
            taggedShape['z'] = 1
            cacheBlockShape = tuple(taggedShape.values())
        
        imageCache.fixAtCurrent.setValue( False ) 
        imageCache.innerBlockShape.setValue( cacheBlockShape ) 
        imageCache.outerBlockShape.setValue( cacheBlockShape ) 
        assert imageCache.Output.ready()
        
        return (imageCache, imageCache.Output)

    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here because our output is directly connected..."

    def propagateDirty(self, slot, subindex, roi):
        # Output slots are directly conncted to internal operators
        pass

    @classmethod
    def getInternalDatasets(cls, filePath):
        """
        Search the given file for internal datasets, and return their internal paths as a list.
        For now, it is assumed that the file is an hdf5 file.
        
        Returns: A list of the internal datasets in the file, or None if the format doesn't support internal datasets.
        """
        datasetNames = None
        ext = os.path.splitext(filePath)[1][1:]
        
        # HDF5. Other formats don't contain more than one dataset (as far as we're concerned).
        if ext in OpInputDataReader.h5Exts:
            datasetNames = []
            # Open the file as a read-only so we can get a list of the internal paths
            with h5py.File(filePath, 'r') as f:
                # Define a closure to collect all of the dataset names in the file.
                def accumulateDatasetPaths(name, val):
                    if type(val) == h5py._hl.dataset.Dataset and 3 <= len(val.shape) <= 5:
                        datasetNames.append( '/' + name )    
                # Visit every group/dataset in the file            
                f.visititems(accumulateDatasetPaths)        

        return datasetNames



















