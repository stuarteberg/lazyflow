from __future__ import print_function
from builtins import range
import os
import h5py
from lazyflow.graph import Graph
from lazyflow.operators import OpBlockedArrayCache
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader

import psutil
this_process = psutil.Process(os.getpid())


# TODO: Play around with the parameters in cacheMemoryManager.py and memory.py

def test_reader_only():
    graph = Graph()

    with h5py.File('/tmp/blabla.h5') as f:        
        opReader = OpStreamingHdf5Reader(graph=graph)
        opReader.Hdf5File.setValue(f)
        opReader.InternalPath.setValue('frames')

        # Manual approach:
        for t in range(30000):
            frame_result = opReader[t:t+1, 0:1000, 0:1000, 0:1].wait()
            print("Finished with frame:", t)
            print("Current memory usage:", this_process.memory_info().rss)
        

def test_reader_and_streamer():
    graph = Graph()

    with h5py.File('/tmp/blabla.h5') as f:        
        opReader = OpStreamingHdf5Reader(graph=graph)
        opReader.Hdf5File.setValue(f)
        opReader.InternalPath.setValue('frames')

        # Streamer approach (actually used during batch processing)        
        streamer = BigRequestStreamer( opReader.OutputImage, ( (0,0,0,0), (30000,1000,1000,1) ) )
         
        def print_progress( progress ):
            print("streamer progress:", progress)
            print("Current memory usage:", this_process.memory_info().rss)
        streamer.progressSignal.subscribe()
 
        def handle_block_result(roi, result):
            print("Finished with frame:", roi[0][0])
        streamer.resultSignal.subscribe( handle_block_result )

        print("starting streamer...")     
        streamer.execute()
        print("DONE.")


def test_reader_and_OpBlockedArrayCache():
    graph = Graph()

    with h5py.File('/tmp/blabla.h5') as f:        
        opReader = OpStreamingHdf5Reader(graph=graph)
        opReader.Hdf5File.setValue(f)
        opReader.InternalPath.setValue('frames')

        opCache = OpBlockedArrayCache(graph=graph)
        opCache.BlockShape.setValue( (1,1000,1000,1) )
        opCache.Input.connect( opReader.OutputImage )

        # Manual approach:
        for t in range(30000):
            frame_result = opCache[t:t+1, 0:1000, 0:1000, 0:1].wait()
            print("Finished with frame:", t)
            print("Current memory usage:", this_process.memory_info().rss)




def test_reader_and_cache_and_streamer():
    graph = Graph()

    with h5py.File('/tmp/blabla.h5') as f:        
        opReader = OpStreamingHdf5Reader(graph=graph)
        opReader.Hdf5File.setValue(f)
        opReader.InternalPath.setValue('frames')

        opCache = OpBlockedArrayCache(graph=graph)
        opCache.BlockShape.setValue( (1,1000,1000,1) )
        opCache.Input.connect( opReader.OutputImage )

        # Streamer approach (actually used during batch processing)        
        streamer = BigRequestStreamer( opCache.Output, ( (0,0,0,0), (30000,1000,1000,1) ) )
         
        def print_progress( progress ):
            print("streamer progress:", progress)
            print("Current memory usage:", this_process.memory_info().rss)
        streamer.progressSignal.subscribe()
 
        def handle_block_result(roi, result):
            print("Finished with frame:", roi[0][0])
        streamer.resultSignal.subscribe( handle_block_result )

        print("starting streamer...")     
        streamer.execute()
        print("DONE.")

if __name__ == "__main__":
    test_reader_only()
    #test_reader_and_OpBlockedArrayCache()
    #test_cache_and_streamer()
