import json
import numpy
import h5py
import vigra
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
    
    VOLUME_MIMETYPE = "binary/imagedata"
    
    def do_GET(self):
        params = self.path.split('/')
        if params[0] == '':
            params = params[1:]

        if len(params) < 5:
            self.send_error(400, "Bad query syntax: {}".format( self.path ))
            return

        uuid, data_name = params[2:4]

        dataset_path = uuid + '/' + data_name
        if dataset_path not in self.server.h5_file:
            self.send_error(404, "Couldn't find dataset: {} in file {}".format( dataset_path, self.server.h5_file.filename ))
            return

        # For this mock server, we assume the data can be found inside our file at /uuid/data_name
        dataset = self.server.h5_file[dataset_path]

        if len(params) == 5:
            self.do_get_info(params, dataset)
        elif len(params) == 8:
            self.do_get_data(params, dataset)
        else:
            self.send_error(400, "Bad query syntax: {}".format( self.path ))
            return

    def do_get_info(self, params, dataset):
        assert len(params) == 5
        cmd = params[4]
        if cmd != 'info':
            self.send_error(400, "Bad query syntax: {}".format( self.path ))
            return
        
        metainfo = self.get_dataset_metainfo(dataset)
        json_text = self.get_dataset_metainfo_json(metainfo)

        self.send_response(200)
        self.send_header("Content-type", "text/json")
        self.send_header("Content-length", str(len(json_text)))
        self.end_headers()
        self.wfile.write( json_text )

    def do_get_data(self, params, dataset):
        assert len(params) == 8
        if params[0] != 'api' or \
           params[1] != 'node':
            self.send_error(400, "Bad query syntax: {}".format( self.path ))
            return
        
        dims_str, roi_shape_str, roi_start_str, fmt = params[4:]

        dataset_ndims = len(dataset.shape)
        expected_dims_str = "_".join( map(str, range( dataset_ndims-1 )) )
        if dims_str != expected_dims_str:
            self.send_error(400, "For now, queries must include all dataset axes.  Your query requested dims: {}".format( dims_str ))
            return
        
        roi_start = tuple( int(x) for x in roi_start_str.split('_') )
        roi_shape = tuple( int(x) for x in roi_shape_str.split('_') )
        
        roi_stop = tuple( numpy.array(roi_start) + roi_shape )        
        slicing = tuple( slice(x,y) for x,y in zip(roi_start, roi_stop) )
        
        # Reverse here because API uses fortran order, but data is stored in C-order
        data = dataset[tuple(reversed(slicing))]
        axistags = vigra.AxisTags.fromJSON( dataset.attrs['axistags'] )
        v_array = vigra.taggedView( data, axistags )
        buf = numpy.getbuffer(v_array)

        self.send_response(200)
        self.send_header("Content-type", self.VOLUME_MIMETYPE)
        self.send_header("Content-length", str(len(buf)))
        self.end_headers()

        self.send_buffer( buf, self.wfile )

    def send_buffer(self, buf, stream):
        remaining_bytes = len(buf)
        while remaining_bytes > 0:
            next_chunk = min( remaining_bytes, 1000 )
            next_bytes = buf[len(buf)-remaining_bytes:len(buf)-(remaining_bytes-next_chunk)]
            stream.write( next_bytes )
            remaining_bytes -= next_chunk
    
    @classmethod
    def get_dataset_metainfo(cls, dataset):
        shape = dataset.shape
        dtype = dataset.dtype.type
        # Tricky business here:
        # The dataset is stored as a C-order-array, but DVID wants fortran order.
        c_tags = vigra.AxisTags.fromJSON( dataset.attrs['axistags'] )
        f_tags = vigra.AxisTags( list(reversed(c_tags)) )
        return DvidVolume.MetaInfo( shape, dtype, f_tags )

    @classmethod
    def get_dataset_metainfo_json(cls, metainfo):
        metadict = {}
        metadict["axes"] = []
        for tag, size in zip(metainfo.axistags, metainfo.shape):
            if tag.key == "c":
                continue
            axisdict = {}
            axisdict["label"] = tag.key.upper()
            axisdict["resolution"] = tag.resolution
            axisdict["units"] = "nanometers" # FIXME: Hardcoded for now
            axisdict["size"] = size
            metadict["axes"].append( axisdict )
        metadict["values"] = []
        
        num_channels = metainfo.shape[ metainfo.axistags.channelIndex ]
        for _ in range( num_channels ):
            metadict["values"].append( { "type" : metainfo.dtype.__name__,
                                         "label" : "" } ) # FIXME: No label for now.
        return json.dumps( metadict )


class H5Server(HTTPServer):
    def __init__(self, h5filepath, *args, **kwargs):
        HTTPServer.__init__(self, *args, **kwargs)
        self.h5filepath = h5filepath
    
    def serve_forever(self):
        with h5py.File( self.h5filepath, 'r' ) as h5_file: # FIXME: Read-only for now (we don't yet support PUT)
            self.h5_file = h5_file
            HTTPServer.serve_forever(self)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python {} <filename.h5>\n".format( sys.argv[0] ))
        sys.exit(1)
    
    filename = sys.argv[1]
    
    server_address = ('', 8000)
    server = H5Server( filename, server_address, H5CutoutRequestHandler )
    server.serve_forever()

    print "DONE!"
