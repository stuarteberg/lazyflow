# -*- coding: utf-8 -*-
###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#		   http://ilastik.org/license/
###############################################################################
#Python
import os
from collections import deque
import itertools
import math
import traceback
from functools import partial
import logging
import copy
import time
import fastfilters
logger = logging.getLogger(__name__)

#SciPy
import numpy, vigra

#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot, OrderedSignal
from lazyflow import roi
from lazyflow.roi import sliceToRoi, roiToSlice
from lazyflow.request import RequestPool
from operators import OpArrayPiper
from lazyflow.rtype import SubRegion
from generic import OpMultiArrayStacker, popFlagsFromTheKey

def zfill_num(n, stop):
    """ Make int strings same length.

    >>> zfill_num(1, 100) # len('99') == 2
    '01'

    >>> zfill_num(1, 101) # len('100') == 3
    '001'

    """
    return str(n).zfill(len(str(stop - 1)))

def makeOpXToMulti(n):
    """A factory for creating OpXToMulti classes."""
    assert n > 0

    class OpXToMulti(Operator):
        category = "Misc"
        name = "{} Element to Multislot".format(n)

        if n == 1:
            inputSlots = [InputSlot('Input')]
        else:
            names = list("Input{}".format(zfill_num(i, n))
                         for i in range(n))
            inputSlots = list(InputSlot(name, optional=True)
                                   for name in names)

        outputSlots = [OutputSlot("Outputs", level=1)]

        def _sorted_inputs(self, filterReady=False):
            """Returns self.inputs.values() sorted by keys.

               :param filterReady: only return slots that are ready.

            """
            keys = sorted(self.inputs.keys())
            slots = list(self.inputs[k] for k in keys)
            if filterReady:
                slots = list(s for s in slots if s.ready())
            return slots

        def _do_assignfrom(self, inslots):
            for inslot, outslot in zip(inslots, self.outputs['Outputs']):
                outslot.meta.assignFrom(inslot.meta)

        def setupOutputs(self):
            inslots = self._sorted_inputs(filterReady=True)
            self.outputs["Outputs"].resize(len(inslots))
            self._do_assignfrom(inslots)

        def execute(self, slot, subindex, roi, result):
            key = roiToSlice(roi.start, roi.stop)
            index = subindex[0]
            inslots = self._sorted_inputs(filterReady=True)
            if index < len(inslots):
                return inslots[index][key].wait()

        def propagateDirty(self, islot, subindex, roi):
            inslots = self._sorted_inputs(filterReady=True)
            index = inslots.index(islot)
            self.outputs["Outputs"][index].setDirty(roi)
            readyslots = list(s for s in inslots[:index] if s.ready())
            self._do_assignfrom(readyslots)

        def setInSlot(self, slot, subindex, roi, value):
            # Nothing to do here: All inputs are directly connected to an input slot.
            pass

    return OpXToMulti

Op1ToMulti = makeOpXToMulti(1)
Op5ToMulti = makeOpXToMulti(5)
Op50ToMulti = makeOpXToMulti(50)

class OpPixelFeaturesPresmoothed(Operator):
    name="OpPixelFeaturesPresmoothed"
    category = "Vigra filter"

    supportFastFilters = InputSlot(value=True)

    inputSlots = [InputSlot("Input"),
                  InputSlot("Matrix"),
                  InputSlot("Scales"),
                  InputSlot("FeatureIds")] # The selection of features to compute

    outputSlots = [OutputSlot("Output"),        # The entire block of features as a single image (many channels)
                   OutputSlot("Features", level=1)] # Each feature image listed separately, with feature name provided in metadata

    # Specify a default set & order for the features we compute
    DefaultFeatureIds = [ 'GaussianSmoothing',
                          'LaplacianOfGaussian',
                          'GaussianGradientMagnitude',
                          'DifferenceOfGaussians',
                          'StructureTensorEigenvalues',
                          'HessianOfGaussianEigenvalues' ]

    WINDOW_SIZE = 3.5
    
    class InvalidScalesError(Exception):
        def __init__(self, invalid_scales):
            self.invalid_scales = invalid_scales

    def __init__(self, *args, **kwargs):
        Operator.__init__(self, *args, **kwargs)
        self.source = OpArrayPiper(parent=self)
        self.source.inputs["Input"].connect(self.inputs["Input"])

        # Give our feature IDs input a default value (connected out of the box, but can be changed)
        self.inputs["FeatureIds"].setValue( self.DefaultFeatureIds )

    def getInvalidScales(self):
        """
        Check each of the scales the user selected against the shape of the input dataset (in space only).
        Return a list of the selected scales that are too large for the input dataset.
        
        .. note:: This function is NOT called automatically.  Clients are expected to call it after 
                  configuring the operator, before they attempt to execute() the operator.
                  If this function returns a non-empty list of scales, then calling execute()
                  would generate errors.
        """
        invalid_scales = []
        for j, scale in enumerate(self.scales):
            if self.matrix[:,j].any():
                tagged_shape = self.Input.meta.getTaggedShape()
                spatial_axes_shape = filter( lambda (k,v): k in 'xyz', tagged_shape.items() )
                spatial_shape = zip( *spatial_axes_shape )[1]
                
                if (scale * self.WINDOW_SIZE > numpy.array(spatial_shape)).any():
                    invalid_scales.append( scale )
        return invalid_scales

    def setupOutputs(self):
        assert self.Input.meta.getAxisKeys()[-1] == 'c', "This code assumes channel is the last axis"

        self.scales = self.inputs["Scales"].value
        self.matrix = self.inputs["Matrix"].value

        if not isinstance(self.matrix, numpy.ndarray):
            raise RuntimeError("OpPixelFeatures: Please input a numpy.ndarray as 'Matrix'")

        dimCol = len(self.scales)
        dimRow = len(self.inputs["FeatureIds"].value)

        assert dimRow== self.matrix.shape[0], "Please check the matrix or the scales they are not the same (scales = %r, matrix.shape = %r)" % (self.scales, self.matrix.shape)
        assert dimCol== self.matrix.shape[1], "Please check the matrix or the scales they are not the same (scales = %r, matrix.shape = %r)" % (self.scales, self.matrix.shape)

        featureNameArray =[]
        oparray = []
        for j in range(dimRow):
            oparray.append([])
            featureNameArray.append([])

        self.newScales = []
        
        for j in range(dimCol):
            destSigma = 1.0
            if self.scales[j] > destSigma:
                self.newScales.append(destSigma)
            else:
                self.newScales.append(self.scales[j])

            logger.debug("Replacing scale %f with new scale %f" %(self.scales[j], self.newScales[j]))
        
        for i, featureId in enumerate(self.inputs["FeatureIds"].value):
            if featureId == 'GaussianSmoothing':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpGaussianSmoothingFF(self))
                    else:
                        oparray[i].append(OpGaussianSmoothing(self))

                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    oparray[i][j].inputs["sigma"].setValue(self.newScales[j])
                    featureNameArray[i].append("Gaussian Smoothing (σ=" + str(self.scales[j]) + ")")

            elif featureId == 'LaplacianOfGaussian':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpLaplacianOfGaussianFF(self))
                    else:
                        oparray[i].append(OpLaplacianOfGaussian(self))
                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    oparray[i][j].inputs["scale"].setValue(self.newScales[j])
                    featureNameArray[i].append("Laplacian of Gaussian (σ=" + str(self.scales[j]) + ")")

            elif featureId == 'StructureTensorEigenvalues':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpStructureTensorEigenvaluesFF(self))
                    else:
                        oparray[i].append(OpStructureTensorEigenvalues(self))
                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    # Note: If you need to change the inner or outer scale,
                    #  you must make a new feature (with a new feature ID) and
                    #  leave this feature here to preserve backwards compatibility
                    oparray[i][j].inputs["innerScale"].setValue(self.newScales[j])
                    #FIXME, FIXME, FIXME
                    #sigma1 = [x*0.5 for x in self.newScales[j]]
                    #oparray[i][j].inputs["outerScale"].setValue(sigma1)
                    oparray[i][j].inputs["outerScale"].setValue(self.newScales[j]*0.5)
                    featureNameArray[i].append("Structure Tensor Eigenvalues (σ=" + str(self.scales[j]) + ")")

            elif featureId == 'HessianOfGaussianEigenvalues':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpHessianOfGaussianEigenvaluesFF(self))
                    else:
                        oparray[i].append(OpHessianOfGaussianEigenvalues(self))
                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    oparray[i][j].inputs["scale"].setValue(self.newScales[j])
                    featureNameArray[i].append("Hessian of Gaussian Eigenvalues (σ=" + str(self.scales[j]) + ")")

            elif featureId == 'GaussianGradientMagnitude':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpGaussianGradientMagnitudeFF(self))
                    else:
                        oparray[i].append(OpGaussianGradientMagnitude(self))
                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    oparray[i][j].inputs["sigma"].setValue(self.newScales[j])
                    featureNameArray[i].append("Gaussian Gradient Magnitude (σ=" + str(self.scales[j]) + ")")

            elif featureId == 'DifferenceOfGaussians':
                for j in range(dimCol):
                    if self.supportFastFilters.value:
                        oparray[i].append(OpDifferenceOfGaussiansFF(self))
                    else:
                        oparray[i].append(OpDifferenceOfGaussians(self))
                    oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                    # Note: If you need to change sigma0 or sigma1, you must make a new
                    #  feature (with a new feature ID) and leave this feature here
                    #  to preserve backwards compatibility
                    oparray[i][j].inputs["sigma0"].setValue(self.newScales[j])
                    #FIXME, FIXME, FIXME
                    #sigma1 = [x*0.66 for x in self.newScales[j]]
                    #oparray[i][j].inputs["sigma1"].setValue(sigma1)
                    oparray[i][j].inputs["sigma1"].setValue(self.newScales[j]*0.66)
                    featureNameArray[i].append("Difference of Gaussians (σ=" + str(self.scales[j]) + ")")

        channelCount = 0
        featureCount = 0
        self.Features.resize( 0 )
        self.featureOutputChannels = []
        channel_names = []
        #connect individual operators
        for i in range(dimRow):
            for j in range(dimCol):
                if self.matrix[i,j]:
                    # Feature names are provided via metadata
                    oparray[i][j].outputs["Output"].meta.description = featureNameArray[i][j]

                    # Prepare the individual features
                    featureCount += 1
                    self.Features.resize( featureCount )

                    featureMeta = oparray[i][j].outputs["Output"].meta
                    featureChannels = featureMeta.shape[ featureMeta.axistags.index('c') ]

                    if featureChannels == 1:
                        channel_names.append( featureNameArray[i][j] )
                    else:
                        for feature_channel_index in range(featureChannels):
                            channel_names.append( featureNameArray[i][j] + " [{}]".format(feature_channel_index) )
                    
                    self.Features[featureCount-1].meta.assignFrom( featureMeta )
                    self.Features[featureCount-1].meta.axistags["c"].description = "" # Discard any semantics related to the input channels
                    self.Features[featureCount-1].meta.display_mode = "" # Discard any semantics related to the input channels
                    self.featureOutputChannels.append( (channelCount, channelCount + featureChannels) )
                    channelCount += featureChannels


        if self.matrix.any():
            self.maxSigma = 0
            #determine maximum sigma
            for i in range(dimRow):
                for j in range(dimCol):
                    val=self.matrix[i,j]
                    if val:
                        self.maxSigma = max(self.scales[j],self.maxSigma)

            self.featureOps = oparray

        # Output meta is a modified copy of the input meta
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = numpy.float32
        self.Output.meta.axistags["c"].description = "" # Discard any semantics related to the input channels
        self.Output.meta.display_mode = "grayscale"
        self.Output.meta.channel_names = channel_names
        self.Output.meta.shape = self.Input.meta.shape[:-1] + (channelCount,)
        self.Output.meta.ideal_blockshape = self._get_ideal_blockshape()
        
        # FIXME: Features are float, so we need AT LEAST 4 bytes per output channel,
        #        but vigra functions may use internal RAM as well.
        self.Output.meta.ram_usage_per_requested_pixel = 4.0 * self.Output.meta.shape[-1]

    def _get_ideal_blockshape(self):
        tagged_blockshape = self.Output.meta.getTaggedShape()
        if 't' in tagged_blockshape:
            # There is no advantage to grouping time slices in a single request.
            tagged_blockshape['t'] = 1
        for k in 'xyz':
            # There is no natural blockshape for spatial dimensions.
            if k in tagged_blockshape:
                tagged_blockshape[k] = 0
        input_blockshape = self.Input.meta.ideal_blockshape
        if input_blockshape is None:
            input_blockshape = (0,) * len( self.Input.meta.shape )
        output_blockshape = tagged_blockshape.values()
        final_blockshape = numpy.maximum( input_blockshape, output_blockshape )
        return tuple( final_blockshape )

    def propagateDirty(self, inputSlot, subindex, roi):
        if inputSlot == self.Input:
            channelAxis = self.Input.meta.axistags.index('c')
            numChannels = self.Input.meta.shape[channelAxis]
            dirtyChannels = roi.stop[channelAxis] - roi.start[channelAxis]
            
            # If all the input channels were dirty, the dirty output region is a contiguous block
            if dirtyChannels == numChannels:
                dirtyKey = list(roiToSlice(roi.start, roi.stop))
                dirtyKey[channelAxis] = slice(None)
                dirtyRoi = sliceToRoi(dirtyKey, self.Output.meta.shape)
                self.Output.setDirty(dirtyRoi[0], dirtyRoi[1])
            else:
                # Only some input channels were dirty, 
                #  so we must mark each dirty output region separately.
                numFeatures = self.Output.meta.shape[channelAxis] / numChannels
                for featureIndex in range(numFeatures):
                    startChannel = numChannels*featureIndex + roi.start[channelAxis]
                    stopChannel = startChannel + roi.stop[channelAxis]
                    dirtyRoi = copy.copy(roi)
                    dirtyRoi.start[channelAxis] = startChannel
                    dirtyRoi.stop[channelAxis] = stopChannel
                    self.Output.setDirty(dirtyRoi)

        elif (inputSlot == self.Matrix
              or inputSlot == self.Scales 
              or inputSlot == self.FeatureIds):
            self.Output.setDirty(slice(None))
        else:
            assert False, "Unknown dirty input slot."
            

    def execute(self, slot, subindex, rroi, result):
        assert slot == self.Features or slot == self.Output
        if slot == self.Features:
            key = roiToSlice(rroi.start, rroi.stop)
            index = subindex[0]
            key = list(key)
            channelIndex = self.Input.meta.axistags.index('c')
            
            # Translate channel slice to the correct location for the output slot.
            key[channelIndex] = slice(self.featureOutputChannels[index][0] + key[channelIndex].start,
                                      self.featureOutputChannels[index][0] + key[channelIndex].stop)
            rroi = SubRegion(self.Output, pslice=key)

            # Get output slot region for this channel
            return self.execute(self.Output, (), rroi, result)
        elif slot == self.outputs["Output"]:
            key = rroi.toSlice()
            
            logger.debug("OpPixelFeaturesPresmoothed: request %s" % (rroi.pprint(),))
            
            cnt = 0
            written = 0
            assert (rroi.stop<=self.outputs["Output"].meta.shape).all()
            flag = 'c'
            channelAxis=self.inputs["Input"].meta.axistags.index('c')
            axisindex = channelAxis
            oldkey = list(key)
            oldkey.pop(axisindex)


            inShape  = self.inputs["Input"].meta.shape
            hasChannelAxis = (self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Channels) > 0)
            #if (self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Channels) == 0):
            #    noChannels = True
            inAxistags = self.inputs["Input"].meta.axistags
                
            shape = self.outputs["Output"].meta.shape
            axistags = self.outputs["Output"].meta.axistags

            result = result.view(vigra.VigraArray)
            result.axistags = copy.copy(axistags)


            hasTimeAxis = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Time)
            timeAxis=self.inputs["Input"].meta.axistags.index('t')

            subkey = popFlagsFromTheKey(key,axistags,'c')
            subshape=popFlagsFromTheKey(shape,axistags,'c')
            at2 = copy.copy(axistags)
            at2.dropChannelAxis()
            subshape=popFlagsFromTheKey(subshape,at2,'t')
            subkey = popFlagsFromTheKey(subkey,at2,'t')

            oldstart, oldstop = roi.sliceToRoi(key, shape)

            start, stop = roi.sliceToRoi(subkey,subkey)
            maxSigma = max(0.7,self.maxSigma)  #we use 0.7 as an approximation of not doing any smoothing
            #smoothing was already applied previously
            
            # The region of the smoothed image we need to give to the feature filter (in terms of INPUT coordinates)
            # 0.7, because the features receive a pre-smoothed array and don't need much of a neighborhood 
            vigOpSourceStart, vigOpSourceStop = roi.enlargeRoiForHalo(start, stop, subshape, 0.7, self.WINDOW_SIZE)
            
            
            # The region of the input that we need to give to the smoothing operator (in terms of INPUT coordinates)
            newStart, newStop = roi.enlargeRoiForHalo(vigOpSourceStart, vigOpSourceStop, subshape, maxSigma, self.WINDOW_SIZE)
            
            newStartSmoother = roi.TinyVector(start - vigOpSourceStart)
            newStopSmoother = roi.TinyVector(stop - vigOpSourceStart)
            roiSmoother = roi.roiToSlice(newStartSmoother, newStopSmoother)

            # Translate coordinates (now in terms of smoothed image coordinates)
            vigOpSourceStart = roi.TinyVector(vigOpSourceStart - newStart)
            vigOpSourceStop = roi.TinyVector(vigOpSourceStop - newStart)

            readKey = roi.roiToSlice(newStart, newStop)

            writeNewStart = start - newStart
            writeNewStop = writeNewStart +  stop - start

            treadKey=list(readKey)

            if hasTimeAxis:
                if timeAxis < channelAxis:
                    treadKey.insert(timeAxis, key[timeAxis])
                else:
                    treadKey.insert(timeAxis-1, key[timeAxis])
            if  self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Channels) == 0:
                treadKey =  popFlagsFromTheKey(treadKey,axistags,'c')
            else:
                treadKey.insert(channelAxis, slice(None,None,None))

            treadKey=tuple(treadKey)

            req = self.inputs["Input"][treadKey]
            
            sourceArray = req.wait()
            req.clean()
            #req.result = None
            req.destination = None
            if sourceArray.dtype != numpy.float32:
                sourceArrayF = sourceArray.astype(numpy.float32)
                try:
                    sourceArray.resize((1,), refcheck = False)
                except:
                    pass
                del sourceArray
                sourceArray = sourceArrayF
                
            #if (self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Channels) == 0):
                #add a channel dimension to make the code afterwards more uniform
            #    sourceArray = sourceArray.view(numpy.ndarray)
            #    sourceArray = sourceArray.reshape(sourceArray.shape+(1,))
            sourceArrayV = sourceArray.view(vigra.VigraArray)
            sourceArrayV.axistags =  copy.copy(inAxistags)
            
            dimCol = len(self.scales)
            dimRow = self.matrix.shape[0]

            sourceArraysForSigmas = [None]*dimCol

            #connect individual operators
            try:
                for j in range(dimCol):
                    hasScale = False
                    for i in range(dimRow):
                        if self.matrix[i,j]:
                            hasScale = True
                    if not hasScale:
                        continue
                    destSigma = 1.0
                    if self.scales[j] > destSigma:
                        tempSigma = math.sqrt(self.scales[j]**2 - destSigma**2)
                    else:
                        destSigma = 0.0
                        tempSigma = self.scales[j]
                    vigOpSourceShape = list(vigOpSourceStop - vigOpSourceStart)
                    if hasTimeAxis:
    
                        if timeAxis < channelAxis:
                            vigOpSourceShape.insert(timeAxis, ( oldstop - oldstart)[timeAxis])
                        else:
                            vigOpSourceShape.insert(timeAxis-1, ( oldstop - oldstart)[timeAxis])
                        vigOpSourceShape.insert(channelAxis, inShape[channelAxis])
    
                        sourceArraysForSigmas[j] = numpy.ndarray(tuple(vigOpSourceShape),numpy.float32)
                        for i,vsa in enumerate(sourceArrayV.timeIter()):
                            droi = (tuple(vigOpSourceStart._asint()), tuple(vigOpSourceStop._asint()))
                            tmp_key = getAllExceptAxis(len(sourceArraysForSigmas[j].shape),timeAxis, i)
                            
                            if self.supportFastFilters.value:
                                vsa  = numpy.ascontiguousarray(vsa)
                                buffer = fastfilters.gaussianSmoothing(vsa, tempSigma, window_size = self.WINDOW_SIZE )
                                droi = roiToSlice(*droi)
                                logger.info("1Presmoothing sigma: {}, roi: {}".format(tempSigma, droi))
                                sourceArraysForSigmas[j][tmp_key] = buffer[droi]
                                
                                if abs(vsa.min()) > 1000 or abs(vsa.max()) > 1000 or abs(buffer.min()) > 1000 or abs(buffer.max()) > 1000:                        
                                    import h5py
                                    
                                    logger.info('Saving debugging in and out files.')
                                    fin = h5py.File('/groups/branson/home/cervantesj/Desktop/presmooth_in.h5','w')
                                    fin['data'] = vsa
                                    fin.close()
           
                                    fout = h5py.File('/groups/branson/home/cervantesj/Desktop/presmooth_out.h5','w')
                                    fout['data'] = buffer
                                    fout.close()
                            
                            else:
                                sourceArraysForSigmas[j][tmp_key] = vigra.filters.gaussianSmoothing(vsa,tempSigma, roi = droi, window_size = self.WINDOW_SIZE )

                    else:
                        droi = (tuple(vigOpSourceStart._asint()), tuple(vigOpSourceStop._asint()))
                        
                        if self.supportFastFilters.value:
                            sourceArrayV  = numpy.ascontiguousarray(sourceArrayV)
                            buffer = fastfilters.gaussianSmoothing(sourceArrayV, tempSigma, window_size = self.WINDOW_SIZE )
                            droi = roiToSlice(*droi)
                            logger.info("1Presmoothing sigma: {}, roi: {}".format(tempSigma, droi))
                            sourceArraysForSigmas[j] = buffer[droi]
                            
                            if abs(sourceArrayV.min()) > 1000 or abs(sourceArrayV.max()) > 1000 or abs(buffer.min()) > 1000 or abs(buffer.max()) > 1000:                        
                                import h5py
                                
                                logger.info('Saving debugging in and out files.')
                                fin = h5py.File('/groups/branson/home/cervantesj/Desktop/presmooth_in.h5','w')
                                fin['data'] = sourceArrayV
                                fin.close()
       
                                fout = h5py.File('/groups/branson/home/cervantesj/Desktop/presmooth_out.h5','w')
                                fout['data'] = buffer
                                fout.close()
                        else:
                            sourceArraysForSigmas[j] = vigra.filters.gaussianSmoothing(sourceArrayV, sigma = tempSigma, roi = droi, window_size = self.WINDOW_SIZE)
                        
            except RuntimeError as e:
                if e.message.find('kernel longer than line') > -1:
                    message = "Feature computation error:\nYour image is too small to apply a filter with sigma=%.1f. Please select features with smaller sigmas." % self.scales[j]
                    raise RuntimeError(message)
                else:
                    raise e

            del sourceArrayV
            try:
                sourceArray.resize((1,), refcheck = False)
            except ValueError:
                # Sometimes this fails, but that's okay.
                logger.debug("Failed to free array memory.")                
            del sourceArray

            closures = []

            #connect individual operators
            for i in range(dimRow):
                for j in range(dimCol):
                    val=self.matrix[i,j]
                    if val:
                        vop= self.featureOps[i][j]
                        oslot = vop.outputs["Output"]
                        req = None
                        #inTagKeys = [ax.key for ax in oslot.meta.axistags]
                        #print inTagKeys, flag
                        if hasChannelAxis:
                            slices = oslot.meta.shape[axisindex]
                            if cnt + slices >= rroi.start[axisindex] and rroi.start[axisindex]-cnt<slices and rroi.start[axisindex]+written<rroi.stop[axisindex]:
                                begin = 0
                                if cnt < rroi.start[axisindex]:
                                    begin = rroi.start[axisindex] - cnt
                                end = slices
                                if cnt + end > rroi.stop[axisindex]:
                                    end -= cnt + end - rroi.stop[axisindex]
                                key_ = copy.copy(oldkey)
                                key_.insert(axisindex, slice(begin, end, None))
                                reskey = [slice(None, None, None) for x in range(len(result.shape))]
                                reskey[axisindex] = slice(written, written+end-begin, None)
                                
                                destArea = result[tuple(reskey)]
                                #readjust the roi for the new source array
                                roiSmootherList = list(roiSmoother)
                                
                                roiSmootherList.insert(axisindex, slice(begin, end, None))
                                
                                if hasTimeAxis:
                                    # The time slice in the ROI doesn't matter:
                                    # The sourceArrayParameter below overrides the input data to be used.
                                    roiSmootherList.insert(timeAxis, 0)
                                roiSmootherRegion = SubRegion(oslot, pslice=roiSmootherList)
                                
                                closure = partial(oslot.operator.execute, oslot, (), roiSmootherRegion, destArea, sourceArray = sourceArraysForSigmas[j])
                                closures.append(closure)

                                written += end - begin
                            cnt += slices
                        else:
                            if cnt>=rroi.start[axisindex] and rroi.start[axisindex] + written < rroi.stop[axisindex]:
                                reskey = [slice(None, None, None) for x in range(len(result.shape))]
                                slices = oslot.meta.shape[axisindex]
                                reskey[axisindex]=slice(written, written+slices, None)
                                #print "key: ", key, "reskey: ", reskey, "oldkey: ", oldkey, "resshape:", result.shape
                                #print "roiSmoother:", roiSmoother
                                destArea = result[tuple(reskey)]
                                #print "destination area:", destArea.shape
                                logger.debug(oldkey, destArea.shape, sourceArraysForSigmas[j].shape)
                                oldroi = SubRegion(oslot, pslice=oldkey)
                                #print "passing roi:", oldroi
                                closure = partial(oslot.operator.execute, oslot, (), oldroi, destArea, sourceArray = sourceArraysForSigmas[j])
                                closures.append(closure)

                                written += 1
                            cnt += 1
            pool = RequestPool()
            for c in closures:
                r = pool.request(c)
            pool.wait()
            pool.clean()

            for i in range(len(sourceArraysForSigmas)):
                if sourceArraysForSigmas[i] is not None:
                    try:
                        sourceArraysForSigmas[i].resize((1,))
                    except:
                        sourceArraysForSigmas[i] = None

###################################################3
class OpPixelFeaturesInterpPresmoothed(Operator):
    name="OpPixelFeaturesPresmoothed"
    category = "Vigra filter"

    
    inputSlots = [InputSlot("Input"),
                  InputSlot("Matrix"),
                  InputSlot("Scales"),
                  InputSlot("FeatureIds"),
                  InputSlot("InterpolationScaleZ")] # The selection of features to compute

    outputSlots = [OutputSlot("Output"),        # The entire block of features as a single image (many channels)
                   OutputSlot("Features", level=1)] # Each feature image listed separately, with feature name provided in metadata

    # Specify a default set & order for the features we compute
    DefaultFeatureIds = [ 'GaussianSmoothing',
                          'LaplacianOfGaussian',
                          'StructureTensorEigenvalues',
                          'HessianOfGaussianEigenvalues',
                          'GaussianGradientMagnitude',
                          'DifferenceOfGaussians' ]
    
    WINDOW_SIZE = 3.5
    
    def __init__(self, *args, **kwargs):
        Operator.__init__(self, *args, **kwargs)
        self.source = OpArrayPiper(parent=self)
        self.source.inputs["Input"].connect(self.inputs["Input"])

        self.stacker = OpMultiArrayStacker(parent=self)

        self.multi = Op50ToMulti(parent=self)

        self.stacker.inputs["Images"].connect(self.multi.outputs["Outputs"])

        # Give our feature IDs input a default value (connected out of the box, but can be changed)
        self.inputs["FeatureIds"].setValue( self.DefaultFeatureIds )

    def getInvalidScales(self):
        """
        Check each of the scales the user selected against the shape of the input dataset (in space only).
        Return a list of the selected scales that are too large for the input dataset.
        
        .. note:: This function is NOT called automatically.  Clients are expected to call it after 
                  configuring the operator, before they attempt to execute() the operator.
                  If this function returns a non-empty list of scales, then calling execute()
                  would generate errors.
        """
        invalid_scales = []
        z_scale = self.InterpolationScaleZ.value
        
        tagged_shape = self.Input.meta.getTaggedShape()
        tagged_shape['z'] = tagged_shape['z']*z_scale
        spatial_axes_shape = filter( lambda (k,v): k in 'xyz', tagged_shape.items() )
        spatial_shape = zip( *spatial_axes_shape )[1]
        
        for j, scale in enumerate(self.scales):
            if self.matrix[:,j].any():
                
                
                if (scale * self.WINDOW_SIZE > numpy.array(spatial_shape)).any():
                    invalid_scales.append( scale )
        return invalid_scales


    def setupOutputs(self):
        if self.inputs["Scales"].connected() and self.inputs["Matrix"].connected():

            self.stacker.inputs["Images"].disconnect()
            self.scales = self.inputs["Scales"].value
            self.matrix = self.inputs["Matrix"].value

            if not isinstance(self.matrix, numpy.ndarray):
                raise RuntimeError("OpPixelFeatures: Please input a numpy.ndarray as 'Matrix'")

            dimCol = len(self.scales)
            dimRow = len(self.inputs["FeatureIds"].value)

            assert dimRow== self.matrix.shape[0], "Please check the matrix or the scales they are not the same (scales = %r, matrix.shape = %r)" % (self.scales, self.matrix.shape)
            assert dimCol== self.matrix.shape[1], "Please check the matrix or the scales they are not the same (scales = %r, matrix.shape = %r)" % (self.scales, self.matrix.shape)

            featureNameArray =[]
            oparray = []
            for j in range(dimRow):
                oparray.append([])
                featureNameArray.append([])

            self.newScales = []
            
            for j in range(dimCol):
                destSigma = 1.0
                if self.scales[j] > destSigma:
                    self.newScales.append(destSigma)
                else:
                    self.newScales.append(self.scales[j])

                logger.debug("Replacing scale %f with new scale %f" %(self.scales[j], self.newScales[j]))
            
            for i, featureId in enumerate(self.inputs["FeatureIds"].value):
                if featureId == 'GaussianSmoothing':
                    for j in range(dimCol):
                        oparray[i].append(OpGaussianSmoothing(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        oparray[i][j].inputs["sigma"].setValue(self.newScales[j])
                        featureNameArray[i].append("Gaussian Smoothing (σ=" + str(self.scales[j]) + ")")

                elif featureId == 'LaplacianOfGaussian':
                    for j in range(dimCol):
                        oparray[i].append(OpLaplacianOfGaussian(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        oparray[i][j].inputs["scale"].setValue(self.newScales[j])
                        featureNameArray[i].append("Laplacian of Gaussian (σ=" + str(self.scales[j]) + ")")

                elif featureId == 'StructureTensorEigenvalues':
                    for j in range(dimCol):
                        oparray[i].append(OpStructureTensorEigenvalues(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        # Note: If you need to change the inner or outer scale,
                        #  you must make a new feature (with a new feature ID) and
                        #  leave this feature here to preserve backwards compatibility
                        oparray[i][j].inputs["innerScale"].setValue(self.newScales[j])
                        oparray[i][j].inputs["outerScale"].setValue(self.newScales[j]*0.5)
                        featureNameArray[i].append("Structure Tensor Eigenvalues (σ=" + str(self.scales[j]) + ")")

                elif featureId == 'HessianOfGaussianEigenvalues':
                    for j in range(dimCol):
                        oparray[i].append(OpHessianOfGaussianEigenvalues(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        oparray[i][j].inputs["scale"].setValue(self.newScales[j])
                        featureNameArray[i].append("Hessian of Gaussian Eigenvalues (σ=" + str(self.scales[j]) + ")")

                elif featureId == 'GaussianGradientMagnitude':
                    for j in range(dimCol):
                        oparray[i].append(OpGaussianGradientMagnitude(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        oparray[i][j].inputs["sigma"].setValue(self.newScales[j])
                        featureNameArray[i].append("Gaussian Gradient Magnitude (σ=" + str(self.scales[j]) + ")")

                elif featureId == 'DifferenceOfGaussians':
                    for j in range(dimCol):
                        oparray[i].append(OpDifferenceOfGaussians(self))
                        oparray[i][j].inputs["Input"].connect(self.source.outputs["Output"])
                        # Note: If you need to change sigma0 or sigma1, you must make a new
                        #  feature (with a new feature ID) and leave this feature here
                        #  to preserve backwards compatibility
                        oparray[i][j].inputs["sigma0"].setValue(self.newScales[j])
                        oparray[i][j].inputs["sigma1"].setValue(self.newScales[j]*0.66)
                        featureNameArray[i].append("Difference of Gaussians (σ=" + str(self.scales[j]) + ")")

            #disconnecting all Operators
            for islot in self.multi.inputs.values():
                islot.disconnect()

            channelCount = 0
            featureCount = 0
            self.Features.resize( 0 )
            self.featureOutputChannels = []
            #connect individual operators
            for i in range(dimRow):
                for j in range(dimCol):
                    if self.matrix[i,j]:
                        # Feature names are provided via metadata
                        oparray[i][j].outputs["Output"].meta.description = featureNameArray[i][j]
                        self.multi.inputs["Input%02d" %(i*dimCol+j)].connect(oparray[i][j].outputs["Output"])
                        logger.debug("connected  Input%02d of self.multi" %(i*dimCol+j))

                        # Prepare the individual features
                        featureCount += 1
                        self.Features.resize( featureCount )

                        featureMeta = oparray[i][j].outputs["Output"].meta
                        featureChannels = featureMeta.shape[ featureMeta.axistags.index('c') ]
                        self.Features[featureCount-1].meta.assignFrom( featureMeta )
                        self.featureOutputChannels.append( (channelCount, channelCount + featureChannels) )
                        channelCount += featureChannels
            
            #additional connection with FakeOperator
            if (self.matrix==0).all():
                fakeOp = OpGaussianSmoothing(parent=self)
                fakeOp.inputs["Input"].connect(self.source.outputs["Output"])
                fakeOp.inputs["sigma"].setValue(10)
                self.multi.inputs["Input%02d" %(i*dimCol+j+1)].connect(fakeOp.outputs["Output"])
                self.multi.inputs["Input%02d" %(i*dimCol+j+1)].disconnect()
                stackerShape = list(self.Input.meta.shape)
                stackerShape[ self.Input.meta.axistags.index('c') ] = 0
                self.stacker.Output.meta.shape = tuple(stackerShape)
                self.stacker.Output.meta.axistags = self.Input.meta.axistags
            else:
                self.stacker.inputs["AxisFlag"].setValue('c')
                self.stacker.inputs["AxisIndex"].setValue(self.source.outputs["Output"].meta.axistags.index('c'))
                self.stacker.inputs["Images"].connect(self.multi.outputs["Outputs"])
    
                self.maxSigma = 0
                #determine maximum sigma
                for i in range(dimRow):
                    for j in range(dimCol):
                        val=self.matrix[i,j]
                        if val:
                            self.maxSigma = max(self.scales[j],self.maxSigma)
    
                self.featureOps = oparray

            # Output meta is a modified copy of the input meta
            self.Output.meta.assignFrom(self.Input.meta)
            self.Output.meta.dtype = numpy.float32
            self.Output.meta.axistags = self.stacker.Output.meta.axistags
            self.Output.meta.shape = self.stacker.Output.meta.shape

    def propagateDirty(self, inputSlot, subindex, roi):
        if inputSlot == self.Input:
            channelAxis = self.Input.meta.axistags.index('c')
            numChannels = self.Input.meta.shape[channelAxis]
            dirtyChannels = roi.stop[channelAxis] - roi.start[channelAxis]
            
            # If all the input channels were dirty, the dirty output region is a contiguous block
            if dirtyChannels == numChannels:
                dirtyKey = roiToSlice(roi.start, roi.stop)
                dirtyKey[channelAxis] = slice(None)
                dirtyRoi = sliceToRoi(dirtyKey, self.Output.meta.shape)
                self.Output.setDirty(dirtyRoi[0], dirtyRoi[1])
            else:
                # Only some input channels were dirty, 
                #  so we must mark each dirty output region separately.
                numFeatures = self.Output.meta.shape[channelAxis] / numChannels
                for featureIndex in range(numFeatures):
                    startChannel = numChannels*featureIndex + roi.start[channelAxis]
                    stopChannel = startChannel + roi.stop[channelAxis]
                    dirtyRoi = copy.copy(roi)
                    dirtyRoi.start[channelAxis] = startChannel
                    dirtyRoi.stop[channelAxis] = stopChannel
                    self.Output.setDirty(dirtyRoi)

        elif (inputSlot == self.Matrix
              or inputSlot == self.Scales 
              or inputSlot == self.FeatureIds
              or inputSlot == self.InterpolationScaleZ):
            self.Output.setDirty(slice(None))
        else:
            assert False, "Unknown dirty input slot."
            

    def execute(self, slot, subindex, rroi, result):
        assert slot == self.Features or slot == self.Output
        if slot == self.Features:
            key = roiToSlice(rroi.start, rroi.stop)
            index = subindex[0]
            subslot = self.Features[index]
            key = list(key)
            channelIndex = self.Input.meta.axistags.index('c')
            
            # Translate channel slice to the correct location for the output slot.
            key[channelIndex] = slice(self.featureOutputChannels[index][0] + key[channelIndex].start,
                                      self.featureOutputChannels[index][0] + key[channelIndex].stop)
            rroi = SubRegion(self.Output, pslice=key)
    
            # Get output slot region for this channel
            return self.execute(self.Output, (), rroi, result)
        elif slot == self.outputs["Output"]:
            key = rroi.toSlice()
            cnt = 0
            written = 0
            assert (rroi.stop<=self.outputs["Output"].meta.shape).all()
            assert self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Channels)!=0, "Data without channels is not yet supported"
            flag = 'c'
            channelAxis=self.inputs["Input"].meta.axistags.index('c')
            assert self.inputs["Input"].meta.shape[channelAxis]==1, "Multichannel data is not yet supported"
            
            #assert len(self.inputs["Input"].meta.shape)==4, "Only 3d data, as the interpolation is in z"
            axisindex = channelAxis
            oldkey = list(key)
            oldkey.pop(axisindex)

            inShape  = self.inputs["Input"].meta.shape
            shape = self.outputs["Output"].meta.shape
            axistags = self.inputs["Input"].meta.axistags

            result = result.view(vigra.VigraArray)
            result.axistags = copy.copy(axistags)

            hasTimeAxis = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Time)
            timeAxis=self.inputs["Input"].meta.axistags.index('t')

            subkey = popFlagsFromTheKey(key,axistags,'c')
            subshape=popFlagsFromTheKey(shape,axistags,'c')
            at2 = copy.copy(axistags)
            at2.dropChannelAxis()
            subshape=popFlagsFromTheKey(subshape,at2,'t')
            subkey = popFlagsFromTheKey(subkey,at2,'t')

            oldstart, oldstop = roi.sliceToRoi(key, shape)

            start, stop = roi.sliceToRoi(subkey,subkey)
            
            maxSigma = max(0.7,self.maxSigma)
            #maxSigma = max(1., self.maxSigma)
            # The region of the smoothed image we need to give to the feature filter (in terms of INPUT coordinates)
            # all this has to be done for the interpolated array!
            zaxis = axistags.index('z')
            scaleZ = self.InterpolationScaleZ.value
            newRangeZ = scaleZ*(shape[zaxis]-1)+1
            interpShape = list(copy.copy(popFlagsFromTheKey(shape, axistags, 'c')))
            interpShape[zaxis] = numpy.long(newRangeZ)
            #TODO: this insanity can most probably be avoided by using taggedShape
            #FIXME: we assume that time is first. Whatever.
            if hasTimeAxis:
                assert timeAxis==0
                interpShape = interpShape[1:]
            interp_start = copy.copy(start)
            interp_stop = copy.copy(stop)
            interp_start[zaxis] = scaleZ*interp_start[zaxis]
            interp_stop[zaxis] = scaleZ*interp_stop[zaxis]-1
            
            vigOpSourceStart, vigOpSourceStop = roi.enlargeRoiForHalo(interp_start, interp_stop, interpShape, 0.7, window = self.WINDOW_SIZE)
            
            # The region of the input that we need to give to the smoothing operator (in terms of INPUT coordinates)
            newStart, newStop = roi.enlargeRoiForHalo(vigOpSourceStart, vigOpSourceStop, interpShape, maxSigma, window = self.WINDOW_SIZE)
            
            vigOpOffset = start - vigOpSourceStart
            newStartSmoother = roi.TinyVector(interp_start - vigOpSourceStart)
            newStopSmoother = roi.TinyVector(interp_stop - vigOpSourceStart)
            roiSmoother = roi.roiToSlice(newStartSmoother, newStopSmoother)
            
            # Translate coordinates (now in terms of smoothed image coordinates)
            vigOpSourceStart = roi.TinyVector(vigOpSourceStart - newStart)
            vigOpSourceStop = roi.TinyVector(vigOpSourceStop - newStart)

            #adjust the readkey, as we read from the non-interpolated image
            newStartNI = copy.copy(newStart)
            newStopNI = copy.copy(newStop)
            newStartNI[zaxis] = numpy.floor(float(newStart[zaxis])/scaleZ)
            newStopNI[zaxis] = numpy.ceil(float(newStop[zaxis])/scaleZ)
            readKey = roi.roiToSlice(newStartNI, newStopNI)
            
            #interpolation is applied on a region read with the above key. In x-y it should just read everything
            newStartI = copy.copy(newStart)
            newStopI = copy.copy(newStop)
            newStopI = newStopI - newStartI
            newStartI = newStartI - newStartI
            newStartI[zaxis] = newStart[zaxis]-scaleZ*newStartNI[zaxis]
            newStopI[zaxis] = newStop[zaxis]-scaleZ*newStartNI[zaxis]
            readKeyInterp = roi.roiToSlice(newStartI, newStopI)

            writeNewStart = start - newStart
            writeNewStop = writeNewStart +  stop - start

            treadKey=list(readKey)
            treadKeyInterp = list(readKeyInterp)

            if hasTimeAxis:
                if timeAxis < channelAxis:
                    treadKey.insert(timeAxis, key[timeAxis])
                    treadKeyInterp.insert(timeAxis, key[timeAxis])
                else:
                    treadKey.insert(timeAxis-1, key[timeAxis])
                    treadKey.insert(timeAxis-1, key[timeAxis])
            if  self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Channels) == 0:
                treadKey =  popFlagsFromTheKey(treadKey,axistags,'c')
                treadKeyInterp =  popFlagsFromTheKey(treadKeyInterp,axistags,'c')
            else:
                treadKey.insert(channelAxis, slice(None,None,None))
                treadKeyInterp.insert(channelAxis, slice(None,None,None))

            treadKey=tuple(treadKey)
            req = self.inputs["Input"][treadKey]
            sourceArray = req.wait()
            
            #req.result = None
            req.clean()
            req.destination = None
            if sourceArray.dtype != numpy.float32:
                sourceArrayF = sourceArray.astype(numpy.float32)
                del sourceArray
                sourceArray = sourceArrayF
            sourceArrayV = sourceArray.view(vigra.VigraArray)
            sourceArrayV.axistags =  copy.copy(axistags)
            
            ########## new stuff #####################
            zaxis = axistags.index('z')
            scaleZ = self.InterpolationScaleZ.value
            newRangeZ = scaleZ*(sourceArrayV.shape[zaxis]-1)+1
            interpShape = list(sourceArrayV.shape)
            interpShape[zaxis] = numpy.long(newRangeZ)
            interpShape = popFlagsFromTheKey(interpShape, axistags, 'c')
            interpShape = popFlagsFromTheKey(interpShape, at2, 't')
            #FIXME: this won't work with multichannel data. Don't care for now.
            
            sourceArrayVInterp = vigra.sampling.resizeVolumeSplineInterpolation(sourceArrayV.squeeze(), shape=interpShape)
            interpShapeFull = sourceArrayVInterp.shape+(1,)
            if hasTimeAxis:
                interpShapeFull = (1,)+interpShapeFull
            sourceArrayVInterp = numpy.ndarray.reshape(sourceArrayVInterp, interpShapeFull)
            sourceArrayVInterp.axistags = copy.copy(axistags)
            sourceArrayVInterp = sourceArrayVInterp[treadKeyInterp]

            dimCol = len(self.scales)
            dimRow = self.matrix.shape[0]

            sourceArraysForSigmas = [None]*dimCol

            #connect individual operators
            for j in range(dimCol):
                hasScale = False
                for i in range(dimRow):
                    if self.matrix[i,j]:
                        hasScale = True
                if not hasScale:
                    continue
                destSigma = 1.0
                if self.scales[j] > destSigma:
                    tempSigma = math.sqrt(self.scales[j]**2 - destSigma**2)
                else:
                    destSigma = 0.0
                    tempSigma = self.scales[j]
                vigOpSourceShape = list(vigOpSourceStop - vigOpSourceStart)
                if hasTimeAxis:

                    if timeAxis < channelAxis:
                        vigOpSourceShape.insert(timeAxis, ( oldstop - oldstart)[timeAxis])
                    else:
                        vigOpSourceShape.insert(timeAxis-1, ( oldstop - oldstart)[timeAxis])
                    vigOpSourceShape.insert(channelAxis, inShape[channelAxis])
                    logger.debug( "vigOpSourceShape: {}".format( vigOpSourceShape ) )
                    sourceArraysForSigmas[j] = numpy.ndarray(tuple(vigOpSourceShape),numpy.float32)
                    for i,vsa in enumerate(sourceArrayVInterp.timeIter()):
                        droi = (tuple(vigOpSourceStart._asint()), tuple(vigOpSourceStop._asint()))
                        tmp_key = getAllExceptAxis(len(sourceArraysForSigmas[j].shape),timeAxis, i)
                        sourceArraysForSigmas[j][tmp_key] = vigra.filters.gaussianSmoothing(vsa,tempSigma, roi = droi, window_size = self.WINDOW_SIZE )
                else:
                    droi = (tuple(vigOpSourceStart._asint()), tuple(vigOpSourceStop._asint()))
                    try:
                        sourceArraysForSigmas[j] = vigra.filters.gaussianSmoothing(sourceArrayVInterp, sigma = tempSigma, roi = droi, window_size = self.WINDOW_SIZE)
                    except RuntimeError:
                        logger.error( "interpolated array: {} {}".format( sourceArrayVInterp.shape, sourceArrayVInterp.axistags ) )
                        logger.error( "source array: {} {}".format( sourceArrayV.shape, sourceArrayV.axistags ) )
                        logger.error( "droi: {}".format( droi ) )
                        raise
            del sourceArrayV
            del sourceArrayVInterp
            try:
                sourceArray.resize((1,), refcheck = False)
            except ValueError:
                # Sometimes this fails, but that's okay.
                logger.debug("Failed to free array memory.")                
            del sourceArray

            closures = []

            #connect individual operators
            for i in range(dimRow):
                for j in range(dimCol):
                    val=self.matrix[i,j]
                    if val:
                        vop= self.featureOps[i][j]
                        oslot = vop.outputs["Output"]
                        req = None
                        inTagKeys = [ax.key for ax in oslot.meta.axistags]
                        if flag in inTagKeys:
                            slices = oslot.meta.shape[axisindex]
                            if cnt + slices >= rroi.start[axisindex] and rroi.start[axisindex]-cnt<slices and rroi.start[axisindex]+written<rroi.stop[axisindex]:
                                begin = 0
                                if cnt < rroi.start[axisindex]:
                                    begin = rroi.start[axisindex] - cnt
                                end = slices
                                if cnt + end > rroi.stop[axisindex]:
                                    end -= cnt + end - rroi.stop[axisindex]
                                    
                                #call feature computation per slice, only for the original data slices
                                nz = scaleZ*(oldkey[zaxis].stop-oldkey[zaxis].start)
                                roiSmootherList = list(roiSmoother)
                                zrange = range(roiSmootherList[zaxis].start, roiSmootherList[zaxis].stop, scaleZ)
                                
                                for iz, z in enumerate(zrange):
                                    
                                    #key_ = copy.copy(oldkey)
                                    key_ = list(oldkey)
                                    key_.insert(axisindex, slice(begin, end, None))
                                    
                                    #readjust the roi for the new source array?
                                    
                                    newRoi = copy.copy(roiSmootherList)
                                    newRoi.insert(axisindex, slice(begin, end, None))
                                    newRoi[zaxis] = slice(z, z+1, None)
                                    newRoi = SubRegion(None, pslice=newRoi)
                                    #print "roi smoother:", roiSmoother
                                    
                                    zStart, zStop = roi.enlargeRoiForHalo((z,), (z+1,), (sourceArraysForSigmas[j].shape[zaxis],), 0.7, self.WINDOW_SIZE)
                                    
                                    sourceKey = []
                                    sourceKey.insert(axistags.index('x'), slice(None, None, None))
                                    sourceKey.insert(axistags.index('y'), slice(None, None, None))
                                    sourceKey.insert(zaxis, slice(zStart, zStop, None))
                                    
                                    reskey = [slice(None, None, None) for x in range(len(result.shape))]
                                    reskey[axisindex] = slice(written, written+end-begin, None)
                                    reskey[zaxis] = slice(iz, iz+1, None)
                                    
                                    destArea = result[tuple(reskey)]
                                    roi_ = SubRegion(oslot, pslice=key_)
                                    
                                    #print "passing to filter:", sourceArraysForSigmas[j][0, 0, zStart:zStop, 0]                                
                                    #closure = partial(oslot.operator.execute, oslot, (), roi_, destArea, sourceArray = sourceArraysForSigmas[j][sourceKey])
                                    closure = partial(oslot.operator.execute, oslot, (), newRoi, destArea, sourceArraysForSigmas[j][sourceKey])
                                    closures.append(closure)
                                    
                                written += end - begin
                            cnt += slices
                        else:
                            if cnt>=rroi.start[axisindex] and rroi.start[axisindex] + written < rroi.stop[axisindex]:
                                reskey = copy.copy(oldkey)
                                reskey.insert(axisindex, written)
                                #print "key: ", key, "reskey: ", reskey, "oldkey: ", oldkey
                                #print "result: ", result.shape, "inslot:", inSlot.shape

                                destArea = result[tuple(reskey)]
                                logger.debug(oldkey, destArea.shape, sourceArraysForSigmas[j].shape)
                                oldroi = SubRegion(oslot, pslice=oldkey)
                                closure = partial(oslot.operator.execute, oslot, (), oldroi, destArea, sourceArray = sourceArraysForSigmas[j])
                                closures.append(closure)

                                written += 1
                            cnt += 1
            pool = RequestPool()
            for c in closures:
                r = pool.request(c)
            pool.wait()
            pool.clean()

            for i in range(len(sourceArraysForSigmas)):
                if sourceArraysForSigmas[i] is not None:
                    try:
                        sourceArraysForSigmas[i].resize((1,))
                    except:
                        sourceArraysForSigmas[i] = None



def getAllExceptAxis(ndim,index,slicer):
    res= [slice(None, None, None)] * ndim
    res[index] = slicer
    return tuple(res)

class OpBaseVigraFilter(OpArrayPiper):
    inputSlots = [InputSlot("Input"), InputSlot("sigma", stype = "float")]
    outputSlots = [OutputSlot("Output")]

    name = "OpBaseVigraFilter"
    category = "Vigra filter"

    vigraFilter = None
    outputDtype = numpy.float32
    inputDtype = numpy.float32
    supportsOut = True
    window_size_feature = 2
    window_size_smoother = 3.5
    supportsRoi = False
    supportsWindow = False

    def execute(self, slot, subindex, rroi, result, sourceArray=None):
        assert len(subindex) == self.Output.level == 0
        key = roiToSlice(rroi.start, rroi.stop)

        kwparams = {}
        for islot in self.inputs.values():
            if islot.name != "Input":
                kwparams[islot.name] = islot.value

        if self.inputs.has_key("sigma"):
            sigma = self.inputs["sigma"].value
        elif self.inputs.has_key("scale"):
            sigma = self.inputs["scale"].value
        elif self.inputs.has_key("sigma0"):
            sigma = self.inputs["sigma0"].value
        elif self.inputs.has_key("innerScale"):
            sigma = self.inputs["innerScale"].value

        windowSize = 3.5
        if self.supportsWindow:
            kwparams['window_size']=self.window_size_feature
            windowSize = self.window_size_smoother

        largestSigma = max(0.7,sigma) #we use 0.7 as an approximation of not doing any smoothing
        #smoothing was already applied previously

        shape = self.outputs["Output"].meta.shape

        axistags = self.inputs["Input"].meta.axistags
        hasChannelAxis = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Channels)
        channelAxis=self.inputs["Input"].meta.axistags.index('c')
        hasTimeAxis = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Time)
        timeAxis=self.inputs["Input"].meta.axistags.index('t')
        zAxis = self.inputs["Input"].meta.axistags.index('z')

        subkey = popFlagsFromTheKey(key,axistags,'c')
        subshape=popFlagsFromTheKey(shape,axistags,'c')
        at2 = copy.copy(axistags)
        at2.dropChannelAxis()
        subshape=popFlagsFromTheKey(subshape,at2,'t')
        subkey = popFlagsFromTheKey(subkey,at2,'t')

        oldstart, oldstop = roi.sliceToRoi(key, shape)
        
        start, stop = roi.sliceToRoi(subkey,subkey)
        
        if sourceArray is not None and zAxis<len(axistags):
            if timeAxis>zAxis:
                subshape[at2.index('z')]=sourceArray.shape[zAxis]
            else:
                subshape[at2.index('z')-1]=sourceArray.shape[zAxis]
        
        newStart, newStop = roi.enlargeRoiForHalo(start, stop, subshape, 0.7, window = windowSize)
        
        readKey = roi.roiToSlice(newStart, newStop)

        writeNewStart = start - newStart
        writeNewStop = writeNewStart +  stop - start

        if (writeNewStart == 0).all() and (newStop == writeNewStop).all():
            fullResult = True
        else:
            fullResult = False

        writeKey = roi.roiToSlice(writeNewStart, writeNewStop)
        writeKey = list(writeKey)
        if timeAxis < channelAxis:
            writeKey.insert(channelAxis-1, slice(None,None,None))
        else:
            writeKey.insert(channelAxis, slice(None,None,None))
        writeKey = tuple(writeKey)

        #print writeKey

        channelsPerChannel = self.resultingChannels()

        if self.supportsRoi is False and largestSigma > 5:
            logger.warn("WARNING: operator", self.name, "does not support roi !!")

        i2 = 0
        for i in range(int(numpy.floor(1.0 * oldstart[channelAxis]/channelsPerChannel)),int(numpy.ceil(1.0 * oldstop[channelAxis]/channelsPerChannel))):
            newReadKey = list(readKey) #add channel and time axis if needed
            if hasTimeAxis:
                if channelAxis > timeAxis:
                    newReadKey.insert(timeAxis, key[timeAxis])
                else:
                    newReadKey.insert(timeAxis-1, key[timeAxis])
            if hasChannelAxis:
                newReadKey.insert(channelAxis, slice(i, i+1, None))
                
            if sourceArray is None:
                req = self.inputs["Input"][newReadKey]
                t = req.wait()
            else:
                if hasChannelAxis:
                    t = sourceArray[getAllExceptAxis(len(newReadKey),channelAxis,slice(i,i+1,None) )]
                else:
                    fullkey = [slice(None, None, None)]*len(newReadKey)
                    t = sourceArray[fullkey]

            t = numpy.require(t, dtype=self.inputDtype)
            t = t.view(vigra.VigraArray)
            t.axistags = copy.copy(axistags)
            t = t.insertChannelAxis()

            sourceBegin = 0

            if oldstart[channelAxis] > i * channelsPerChannel:
                sourceBegin = oldstart[channelAxis] - i * channelsPerChannel
            sourceEnd = channelsPerChannel
            if oldstop[channelAxis] < (i+1) * channelsPerChannel:
                sourceEnd = channelsPerChannel - ((i+1) * channelsPerChannel - oldstop[channelAxis])
            destBegin = i2
            destEnd = i2 + sourceEnd - sourceBegin

            if channelsPerChannel>1:
                tkey=getAllExceptAxis(len(shape),channelAxis,slice(destBegin,destEnd,None))
                resultArea = result[tkey]
            else:
                tkey=getAllExceptAxis(len(shape),channelAxis,slice(i2,i2+1,None))
                resultArea = result[tkey]

            i2 += destEnd-destBegin

            supportsOut = self.supportsOut
            if (destEnd-destBegin != channelsPerChannel):
                supportsOut = False

            supportsOut = False #disable for now due to vigra crashes! #FIXME
            for step,image in enumerate(t.timeIter()):
                nChannelAxis = channelAxis - 1

                if timeAxis > channelAxis or not hasTimeAxis:
                    nChannelAxis = channelAxis
                twriteKey=getAllExceptAxis(image.ndim, nChannelAxis, slice(sourceBegin,sourceEnd,None))

                if hasTimeAxis > 0:
                    tresKey  = getAllExceptAxis(resultArea.ndim, timeAxis, step)
                else:
                    tresKey  = slice(None, None,None)

                #print tresKey, twriteKey, resultArea.shape, temp.shape
                vres = resultArea[tresKey]
                               
                if supportsOut:
                    if self.supportsRoi:
                        vroi = (tuple(writeNewStart._asint()), tuple(writeNewStop._asint()))
                        try:
                            vres = vres.view(vigra.VigraArray)
                            vres.axistags = copy.copy(image.axistags)
                            logger.debug( "FAST LANE {} {} {} {}".format( self.name, vres.shape, image[twriteKey].shape, vroi ) )
                            temp = self.vigraFilter(image[twriteKey], roi = vroi,out=vres, **kwparams)
                        except:
                            logger.error( "{} {} {} {}".format(self.name, image.shape, vroi, kwparams) )
                            raise
                    else:
                        try:
                            temp = self.vigraFilter(image, **kwparams)
                        except:
                            logger.error( "{} {} {} {}".format(self.name, image.shape, vroi, kwparams) )
                            raise
                        temp=temp[writeKey]
                else:
                    if self.supportsRoi:
                        vroi = (tuple(writeNewStart._asint()), tuple(writeNewStop._asint()))
                        try:
                            temp = self.vigraFilter(image, roi = vroi, **kwparams)
                            logger.info("Filter params: {}".format(kwparams))
                        except Exception, e:
                            logger.error( "EXCEPT 2.1 {} {} {} {}".format( self.name, image.shape, vroi, kwparams ) )
                            traceback.print_exc(e)
                            import sys
                            sys.exit(1)
                    else:
                        try:
                            temp = self.vigraFilter(image, **kwparams)
                        except Exception, e:
                            logger.error( "EXCEPT 2.2 {} {} {} {}".format( self.name, image.shape, kwparams ) )
                            traceback.print_exc(e)
                            sys.exit(1)
                        temp=temp[writeKey]


                    try:
                        vres[:] = temp[twriteKey]
                        if abs(image.min()) > 1000 or abs(image.max()) > 1000 or abs(vres.min()) > 1000 or abs(vres.max()) > 1000:                        
                        #if abs(vres.min()) > 1000 or abs(vres.max()) > 1000:
                            import h5py
                            
                            logger.info('Saving debugging in and out files.')
                            fin = h5py.File('/groups/branson/home/cervantesj/Desktop/testin2.h5','w')
                            fin['data'] = image
                            fin.close()
   
                            fout = h5py.File('/groups/branson/home/cervantesj/Desktop/testout2.h5','w')
                            fout['data'] = vres
                            fout.close()
                            
                            
                        logger.info("name: {}, Roi: {}, Min: {}, Max: {}".format(self.name, twriteKey, vres.min(), vres.max()))
                    except:
                        logger.error( "EXCEPT3 {} {} {}".format( vres.shape, temp.shape, twriteKey ) )
                        logger.error( "EXCEPT3 {} {} {}".format( resultArea.shape,  tresKey, twriteKey ) )
                        logger.error( "EXCEPT3 {} {} {}".format( step, t.shape, timeAxis ) )
                        raise
                
                #print "(in.min=",image.min(),",in.max=",image.max(),") (vres.min=",vres.min(),",vres.max=",vres.max(),")"


    def setupOutputs(self):
        
        # Output meta starts with a copy of the input meta, which is then modified
        self.Output.meta.assignFrom(self.Input.meta)
        
        numChannels  = 1
        inputSlot = self.inputs["Input"]
        if inputSlot.meta.axistags.axisTypeCount(vigra.AxisType.Channels) > 0:
            channelIndex = self.inputs["Input"].meta.axistags.channelIndex
            numChannels = self.inputs["Input"].meta.shape[channelIndex]
            inShapeWithoutChannels = popFlagsFromTheKey( self.inputs["Input"].meta.shape,self.inputs["Input"].meta.axistags,'c')
        else:
            inShapeWithoutChannels = inputSlot.meta.shape
            channelIndex = len(inputSlot.meta.shape)

        self.outputs["Output"].meta.dtype = self.outputDtype
        p = self.inputs["Input"].partner
        at = copy.copy(inputSlot.meta.axistags)

        if at.axisTypeCount(vigra.AxisType.Channels) == 0:
            at.insertChannelAxis()

        self.outputs["Output"].meta.axistags = at

        channelsPerChannel = self.resultingChannels()
        inShapeWithoutChannels = list(inShapeWithoutChannels)
        inShapeWithoutChannels.insert(channelIndex,numChannels * channelsPerChannel)
        self.outputs["Output"].meta.shape = tuple(inShapeWithoutChannels)

        if self.outputs["Output"].meta.axistags.axisTypeCount(vigra.AxisType.Channels) == 0:
            self.outputs["Output"].meta.axistags.insertChannelAxis()

        # The output data range is not necessarily the same as the input data range.
        if 'drange' in self.Output.meta:
            del self.Output.meta['drange']

    def resultingChannels(self):
        raise RuntimeError('resultingChannels() not implemented')


#difference of Gaussians
def differenceOfGausssians(image,sigma0, sigma1,window_size, roi, out = None):
    """ difference of gaussian function"""
    return (vigra.filters.gaussianSmoothing(image,sigma0,window_size=window_size,roi = roi)-vigra.filters.gaussianSmoothing(image,sigma1,window_size=window_size,roi = roi))


def firstHessianOfGaussianEigenvalues(image, **kwargs):
    return vigra.filters.hessianOfGaussianEigenvalues(image, **kwargs)[...,0:1]

def coherenceOrientationOfStructureTensor(image,sigma0, sigma1, window_size, out = None):
    """
    coherence Orientation of Structure tensor function:
    input:  M*N*1ch VigraArray
            sigma corresponding to the inner scale of the tensor
            scale corresponding to the outher scale of the tensor

    output: M*N*2 VigraArray, the firest channel correspond to coherence
                              the second channel correspond to orientation
    """

    #FIXME: make more general

    #assert image.spatialDimensions==2, "Only implemented for 2 dimensional images"
    assert len(image.shape)==2 or (len(image.shape)==3 and image.shape[2] == 1), "Only implemented for 2 dimensional images"

    st=vigra.filters.structureTensor(image, sigma0, sigma1, window_size = window_size)
    i11=st[:,:,0]
    i12=st[:,:,1]
    i22=st[:,:,2]

    if out is not None:
        assert out.shape[0] == image.shape[0] and out.shape[1] == image.shape[1] and out.shape[2] == 2
        res = out
    else:
        res=numpy.ndarray((image.shape[0],image.shape[1],2))

    res[:,:,0]=numpy.sqrt( (i22-i11)**2+4*(i12**2))/(i11-i22)
    res[:,:,1]=numpy.arctan(2*i12/(i22-i11))/numpy.pi +0.5


    return res

class OpDifferenceOfGaussiansFF(OpBaseVigraFilter):
    name = "DifferenceOfGaussiansFF"
    
    def differenceOfGausssiansFF(image, sigma0, sigma1, window_size):
        image = numpy.ascontiguousarray(image)
        return (fastfilters.gaussianSmoothing(image, sigma0, window_size) - fastfilters.gaussianSmoothing(image, sigma1, window_size) )
    
    vigraFilter = staticmethod(differenceOfGausssiansFF)
    
    outputDtype = numpy.float32
    supportsOut = False
    supportsWindow = True
    supportsRoi = False
    inputSlots = [InputSlot("Input"), InputSlot("sigma0", stype = "float"), InputSlot("sigma1", stype = "float")]

    def resultingChannels(self):
        return 1

class OpDifferenceOfGaussians(OpBaseVigraFilter):
    name = "DifferenceOfGaussians"
    
    vigraFilter = staticmethod(differenceOfGausssians)
    
    outputDtype = numpy.float32
    supportsOut = False
    supportsWindow = True
    supportsRoi = True
    inputSlots = [InputSlot("Input"), InputSlot("sigma0", stype = "float"), InputSlot("sigma1", stype = "float")]

    def resultingChannels(self):
        return 1


class OpGaussianSmoothingFF(OpBaseVigraFilter):
    name = "GaussianSmoothingFF"

    vigraFilter =  staticmethod(fastfilters.gaussianSmoothing)
       
    outputDtype = numpy.float32
    supportsRoi = False
    supportsWindow = True
    supportsOut = False

    def resultingChannels(self):
        return 1

class OpGaussianSmoothing(OpBaseVigraFilter):
    name = "GaussianSmoothing"
            
    vigraFilter = staticmethod(vigra.filters.gaussianSmoothing)
       
    outputDtype = numpy.float32
    supportsRoi = True
    supportsWindow = True
    supportsOut = True

    def resultingChannels(self):
        return 1

class OpHessianOfGaussianEigenvaluesFF(OpBaseVigraFilter):
    name = "HessianOfGaussianEigenvaluesFF"
    
    vigraFilter = staticmethod(fastfilters.hessianOfGaussianEigenvalues)
    
    outputDtype = numpy.float32
    supportsRoi = False
    supportsWindow = True
    supportsOut = False
    inputSlots = [InputSlot("Input"), InputSlot("scale", stype = "float")]

    def resultingChannels(self):
        temp = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
        return temp

class OpHessianOfGaussianEigenvalues(OpBaseVigraFilter):
    name = "HessianOfGaussianEigenvalues"
    
    vigraFilter = staticmethod(vigra.filters.hessianOfGaussianEigenvalues)
    
    outputDtype = numpy.float32
    supportsRoi = True
    supportsWindow = True
    supportsOut = True
    inputSlots = [InputSlot("Input"), InputSlot("scale", stype = "float")]

    def resultingChannels(self):
        temp = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
        return temp


class OpStructureTensorEigenvaluesFF(OpBaseVigraFilter):
    name = "StructureTensorEigenvaluesFF"
    
    vigraFilter = staticmethod(fastfilters.structureTensorEigenvalues)
    
    outputDtype = numpy.float32
    supportsRoi = False
    supportsWindow = True
    supportsOut = False
    inputSlots = [InputSlot("Input"), InputSlot("innerScale", stype = "float"),InputSlot("outerScale", stype = "float")]

    def resultingChannels(self):
        temp = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
        return temp


class OpStructureTensorEigenvalues(OpBaseVigraFilter):
    name = "StructureTensorEigenvalues"
    
    vigraFilter = staticmethod(vigra.filters.structureTensorEigenvalues)
    
    outputDtype = numpy.float32
    supportsRoi = True
    supportsWindow = True
    supportsOut = True
    inputSlots = [InputSlot("Input"), InputSlot("innerScale", stype = "float"),InputSlot("outerScale", stype = "float")]

    def resultingChannels(self):
        temp = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
        return temp


class OpHessianOfGaussianEigenvaluesFirst(OpBaseVigraFilter):
    name = "First Eigenvalue of Hessian Matrix"
    vigraFilter = staticmethod(firstHessianOfGaussianEigenvalues)
    outputDtype = numpy.float32
    supportsOut = False
    supportsWindow = True
    supportsRoi = True

    inputSlots = [InputSlot("Input"), InputSlot("scale", stype = "float")]

    def resultingChannels(self):
        return 1


class OpHessianOfGaussian(OpBaseVigraFilter):
    name = "HessianOfGaussian"
    vigraFilter = staticmethod(vigra.filters.hessianOfGaussian)
    outputDtype = numpy.float32
    supportsWindow = True
    supportsRoi = True
    supportsOut = True

    def resultingChannels(self):
        temp = self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
        return temp


class OpGaussianGradientMagnitudeFF(OpBaseVigraFilter):
    name = "GaussianGradientMagnitudeFF"
    
    vigraFilter = staticmethod(fastfilters.gaussianGradientMagnitude)
    
    outputDtype = numpy.float32
    supportsRoi = False
    supportsWindow = True
    supportsOut = False

    def resultingChannels(self):
        return 1
    

class OpGaussianGradientMagnitude(OpBaseVigraFilter):
    name = "GaussianGradientMagnitude"
    
    vigraFilter = staticmethod(vigra.filters.gaussianGradientMagnitude)
    
    outputDtype = numpy.float32
    supportsRoi = True
    supportsWindow = True
    supportsOut = True

    def resultingChannels(self):
        return 1


class OpLaplacianOfGaussianFF(OpBaseVigraFilter):
    name = "LaplacianOfGaussianFF"
    
    vigraFilter = staticmethod(fastfilters.laplacianOfGaussian)
    
    outputDtype = numpy.float32
    supportsOut = False
    supportsRoi = False
    supportsWindow = True
    inputSlots = [InputSlot("Input"), InputSlot("scale", stype = "float")]

    def resultingChannels(self):
        return 1


class OpLaplacianOfGaussian(OpBaseVigraFilter):
    name = "LaplacianOfGaussian"
    
    vigraFilter = staticmethod(vigra.filters.laplacianOfGaussian)
    
    outputDtype = numpy.float32
    supportsOut = True
    supportsRoi = True
    supportsWindow = True
    inputSlots = [InputSlot("Input"), InputSlot("scale", stype = "float")]

    def resultingChannels(self):
        return 1

class OpImageReader(Operator):
    """
    Read an image using vigra.impex.readImage().
    Supports 2D images (output as xyc) and also multi-page tiffs (output as zyxc).
    """
    Filename = InputSlot(stype="filestring")
    Image = OutputSlot()
    
    def setupOutputs(self):
        filename = self.Filename.value

        info = vigra.impex.ImageInfo(filename)
        assert [tag.key for tag in info.getAxisTags()] == ['x', 'y', 'c']

        shape_xyc = info.getShape()
        shape_yxc = (shape_xyc[1], shape_xyc[0], shape_xyc[2])

        self.Image.meta.dtype = info.getDtype()
        self.Image.meta.prefer_2d = True

        numImages = vigra.impex.numberImages(filename)
        if numImages == 1:
            # For 2D, we use order yxc.
            self.Image.meta.shape = shape_yxc
            v_tags = info.getAxisTags()
            self.Image.meta.axistags = vigra.AxisTags( [v_tags[k] for k in 'yxc'] )
        else:
            # For 3D, we use zyxc
            # Insert z-axis shape
            shape_zyxc = (numImages,) + shape_yxc
            self.Image.meta.shape = shape_zyxc

            # Insert z tag
            z_tag = vigra.defaultAxistags('z')[0]
            tags_xyc = [tag for tag in info.getAxisTags()]
            tags_zyxc = [z_tag] +  list(reversed(tags_xyc[:-1])) + tags_xyc[-1:]
            self.Image.meta.axistags = vigra.AxisTags( tags_zyxc )

    def execute(self, slot, subindex, rroi, result):
        filename = self.Filename.value

        if 'z' in self.Image.meta.getAxisKeys():
            # Copy from each image slice into the corresponding slice of the result.
            roi_zyxc = numpy.array( [rroi.start, rroi.stop] )
            for z_global, z_result in zip( range(*roi_zyxc[:,0]), 
                                           range(result.shape[0]) ):
                full_slice = vigra.impex.readImage(filename, index=z_global)
                full_slice = full_slice.transpose(1,0,2) # xyc -> yxc
                assert full_slice.shape == self.Image.meta.shape[1:]
                result[z_result] = full_slice[roiToSlice( *roi_zyxc[:,1:] )]
        else:
            full_slice = vigra.impex.readImage(filename).transpose(1,0,2) # xyc -> yxc
            assert full_slice.shape == self.Image.meta.shape
            roi_yxc = numpy.array( [rroi.start, rroi.stop] )
            result[:] = full_slice[roiToSlice( *roi_yxc )]
        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Filename:
            self.Image.setDirty()
        else:
            assert False, "Unknown dirty input slot."

