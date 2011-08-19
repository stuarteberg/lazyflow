import numpy

from lazyflow.graph import Operators, Operator, InputSlot, OutputSlot, MultiInputSlot, MultiOutputSlot
from lazyflow.roi import sliceToRoi, roiToSlice, block_view
from Queue import Empty
from collections import deque
import greenlet, threading
import vigra
import copy



class OpTrainRandomForest(Operator):
    name = "TrainRandomForest"
    description = "Train a random forest on multiple images"
    category = "Learning"
    
    inputSlots = [MultiInputSlot("Images"),MultiInputSlot("Labels"), InputSlot("fixClassifier", stype="bool")]
    outputSlots = [OutputSlot("Classifier")]
    
    def notifyConnectAll(self):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"]._dtype = object
            self.outputs["Classifier"]._shape = (1,)
            self.outputs["Classifier"]._axistags  = "classifier"
            self.outputs["Classifier"].setDirty((slice(0,1,None),))            
             
    
    def notifySubConnect(self, slots, indexes):
        if self.inputs["fixClassifier"].connected():
            if self.inputs["fixClassifier"].value == False:
                self.outputs["Classifier"]._dtype = object
                self.outputs["Classifier"]._shape = (1,)
                self.outputs["Classifier"]._axistags  = "classifier"
                self.outputs["Classifier"].setDirty((slice(0,1,None),))            
             
    def getOutSlot(self, slot, key, result):
        
        featMatrix=[]
        labelsMatrix=[]
        print "outslot of training operator", len(self.inputs["Labels"])
        for i,labels in enumerate(self.inputs["Labels"]):
            print "labels...", i, labels.shape
            if labels.shape is not None:
                print "labels processing..."
                labels=labels[:].allocate().wait()
                print "labels read"
                
                indexes=numpy.nonzero(labels[...,0].view(numpy.ndarray))
                #print "checking number of labels:"
                #tmpind = numpy.where(labels==1)
                #print "label1: ", len(tmpind[0])
                #tmpind = numpy.where(labels==2)
                #print "label1: ", len(tmpind[0])
                #tmpind = numpy.where(labels==3)
                #print "label1: ", len(tmpind[0])
                #tmpind = numpy.where(labels==4)
                #print "label1: ", len(tmpind[0])
                #tmpind = numpy.where(labels==5)
                #print "label1: ", len(tmpind[0])
                
                
                #indexes=numpy.nonzero(labels.view(numpy.ndarray))
                print "length of index array: ", len(indexes), len(indexes[0])
                #Maybe later request only part of the region?
                
                image=self.inputs["Images"][i][:].allocate().wait()
                print "features read"
                print "OpTrainRandomForest:", image.shape, labels.shape
                
                features=image[indexes]
                labels=labels[indexes]
                
                featMatrix.append(features)
                labelsMatrix.append(labels)
        

        featMatrix=numpy.concatenate(featMatrix,axis=0)
        labelsMatrix=numpy.concatenate(labelsMatrix,axis=0)
        
        print "featMatrix.shape:", featMatrix.shape
        print "labelsMatrix.shape:", labelsMatrix.shape
        
        RF=vigra.learning.RandomForest(100)        
        try:
            RF.learnRF(featMatrix.astype(numpy.float32),labelsMatrix.astype(numpy.uint32))
        except:
            print "ERROR: couldnt learn classifier"
            print featMatrix, labelsMatrix
            print featMatrix.shape, featMatrix.dtype
            print labelsMatrix.shape, labelsMatrix.dtype            
            
        result[0]=RF
        
    def setInSlot(self, slot, key, value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def setSubInSlot(self,slots,indexes, key,value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def notifySubSlotDirty(self, slots, indexes, key):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))    

    def notifyDirty(self, slot, key):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))            

class OpPredictRandomForest(Operator):
    name = "PredictRandomForest"
    description = "Predict on multiple images"
    category = "Learning"
    
    inputSlots = [InputSlot("Image"),InputSlot("Classifier"),InputSlot("LabelsCount",stype='integer')]
    outputSlots = [OutputSlot("PMaps")]
    
    def notifyConnectAll(self):
        inputSlot = self.inputs["Image"]    
        nlabels=self.inputs["LabelsCount"].value
        
        
        """
        self.outputs["PMaps"].resize(len(inputSlot)) #clearAllSlots()
        for i,islot in enumerate(self.inputs["Images"]):
            oslot = self.outputs["PMaps"][i]
            if islot.partner is not None:
                oslot._dtype = numpy.float32
                oslot._shape = islot.shape[:-1]+(nlabels,)
                oslot._axistags = islot.axistags
        
        """
        oslot = self.outputs["PMaps"]
        islot=self.inputs["Image"]

        print "OPPREDICTRANDOMFOREST: ", nlabels, islot.shape

        oslot._dtype = numpy.float32
        
        oslot._axistags = islot.axistags
        oslot._shape = islot.shape[:-1]+(nlabels,)
    """    
    def notifySubConnect(self, slots, indexes):
        print "OpClassifier notifySubConnect"
        self.notifyConnectAll()                 
    """
        
        

    def getOutSlot(self,slot, key, result):
        nlabels=self.inputs["LabelsCount"].value

        RF=self.inputs["Classifier"].value
        assert RF.labelCount() == nlabels, "ERROR: OpPredictRandomForest, labelCount differs from true labelCount! %r vs. %r" % (RF.labelCount(), nlabels)        
                
        newKey = key[:-1]
        newKey += (slice(0,self.inputs["Image"].shape[-1],None),)
        
        res = self.inputs["Image"][newKey].allocate().wait()
               
        shape=res.shape
        prod = 1
        for i,e in enumerate(shape):
            if i < len(shape) - 1:
                prod *= e            

        features=res.reshape(prod, shape[-1])
        

        prediction=RF.predictProbabilities(features.astype(numpy.float32))        
        
        prediction = prediction.reshape(*(shape[:-1] + (RF.labelCount(),)))
        
        result[:]=prediction[...,key[-1]]

            
    def notifyDirty(self, slot, key):
        if slot == self.inputs["Classifier"]:
            print "OpPredict: Classifier changed, setting dirty"
            self.outputs["PMaps"].setDirty(slice(None,None,None))     
        elif slot == self.inputs["Image"]:
            nlabels=self.inputs["LabelsCount"].value
            self.outputs["PMaps"].setDirty(key[:-1] + (slice(0,nlabels,None),))
            
            
            
            

        