import h5py
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

from lazyflow.utility.io.dvidVolume import DvidVolume

class H5CutoutRequestHandler(BaseHTTPRequestHandler):
    """
    Suports the following DVID REST calls:
    
    Meta info:
        GET  /api/node/<UUID>/<data name>/info
    
    Cutout volume:
        GET  /api/node/<UUID>/<data name>/<dims>/<size>/<offset>[/<format>]
    """
    def do_GET(self):
        params = self.path.split('/')
        if params[0] == '':
            params = params[1:]
        
        if len(params) != 8 or \
           params[0] != 'api' or \
           params[1] != 'node':
            self.send_error(400, "Bad query syntax: {}".format( self.path ))
            return
        
        uuid, data_name, dims_str, roi_shape_str, roi_start_str, format = params[2:]
        
        # For this mock server, we assume the data can be found inside our file at /uuid/data_name
        dataset_path = uuid + '/' + data_name
        if dataset_path not in self.server.h5_file:
            self.send_error(404, "Couldn't find dataset: {}".format( dataset_path ))
            return

        dataset = self.server.h5_file[dataset_path]
        roi_start = tuple( int(x) for x in roi_start_str.split('_') )
        roi_shape = tuple( int(x) for x in roi_shape_str.split('_') )
        
        roi_stop = tuple( numpy.array(roi_start) + roi_shape )        
        slicing = tuple( slice(x,y) for x,y in zip(roi_start, roi_stop) )
        
        self.send_header("Content-type", "binary/imagedata")
        self.end_headers()

        # Reverse here because API uses fortran order, but data is stored in C-order
        data = dataset[tuple(reversed(slicing))]
        axistags = vigra.AxisTags.fromJSON( dataset.attrs['axistags'] )
        v_array = vigra.taggedView( data, axistags )
        self.encode_from_vigra_array( v_array, self.wfile )

    def encode_from_vigra_array(self, v_array, stream):
        buf = numpy.getbuffer(v_array)
        remaining_bytes = len(buf)
        while remaining_bytes > 0:
            next_chunk = min( remaining_bytes, 1000 )
            stream.write( buf[-remaining_bytes:-(remaining_bytes-next_chunk)] )
            remaining_bytes -= next_chunk
    
    @classmethod
    def get_dataset_metainfo(cls, dataset):
        shape = dataset.shape
        dtype = dataset.dtype
        # Tricky business here:
        # The dataset is stored as a C-order-array, but DVID wants fortran order.
        c_tags = vigra.AxisTags.fromJSON( dataset.attrs['axistags'] )
        f_tags = vigra.AxisTags( list(reversed(c_tags)) )
        return DvidVolume.MetaInfo( shape, dtype, f_tags )

class H5Server(HTTPServer):
    def __init__(self, h5filepath, *args, **kwargs):
        HTTPServer.__init__(self, *args, **kwargs)
        self.h5filepath = h5filepath
    
    def serve_forever(self):
        with h5py.File( self.h5filepath ) as h5_file:
            self.h5_file = h5_file
            HTTPServer.serve_forever(self)

if __name__ == "__main__":
    server_address = ('', 8000)
    server = H5Server( 'test.h5', server_address, H5CutoutRequestHandler )
    server.serve_forever()

    print "DONE!"
