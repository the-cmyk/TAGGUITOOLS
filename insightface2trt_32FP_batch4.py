# check sources only if in trouble: https://medium.com/@penolove15/face-recognition-with-arcface-with-tensorrt-abb544738e39
# 1 - install tensorrt https://developer.nvidia.com/tensorrt
# 2 - install ONNX (maybe it is important to be 1.4.1)
# 3 - wget https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.onnx
# 4 - Update onnx_file_path and run this file to create TRT engine
# if it doesnt work, try repeating inside a NVIDIA/tensorrt container (atleast 19.09)
# 5 - run this file to build engine and run inference
# python
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

batch_size = 4
TRT_LOGGER = trt.Logger()
onnx_file_path = "/insightface/resnet100.onnx"
def get_engine(onnx_file_path, engine_file_path="arc_FP32_batch4.engine"):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            #builder.fp16_mode = True
            #builder.strict_type_constraints = True
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = batch_size
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None

            #network.get_input(0).shape = [1, 3, 112, 112]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
    
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


class Preprocess(object):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for insightface.
    """

    def __init__(self, input_resolution):
        self.input_resolution = input_resolution

    def process(self, input_image_path):
        image_raw, image_resized = self._load_and_resize(input_image_path)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        return image_raw, image_preprocessed

    def _load_and_resize(self, input_image_path):
        image_raw = Image.open(input_image_path)
        
        new_resolution = (
            self.input_resolution[1],
            self.input_resolution[0])
        image_resized = image_raw.resize(
            new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, dtype=np.float32, order='C')
        return image

def inference_mem_trt7(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def inference_mem_trt6(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def inference():
    input_image_path = "/insightface/sample-images/t1.jpg"
    input_image_path2 = "/insightface/sample-images/t2.jpg"

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution = (112, 112)
    # Create a pre-processor object by specifying the required input resolution
    preprocessor = Preprocess(input_resolution)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    image_raw, image2 = preprocessor.process(input_image_path2)
    images = np.dstack((image, image2, image, image2))
    print("shape of images:")
    print(images)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = np.ascontiguousarray(images.astype(np.float32))
        import time
        start_time = time.time()
        for i in range(10):
            trt_outputs = inference_mem_trt6(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
        print("--- %s seconds ---" % ((time.time() - start_time)/10/batch_size))
    # Return the host output. 
    return trt_outputs

    

print("Running inference...")
output = inference()
#print(output[0])
print(len(output[0]))