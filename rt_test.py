import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import torch
import time
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
from data import test_dataset
from skimage import io, transform
from torch.autograd import Variable
# logger to capture errors, warnings, and other information during the build and inference phases
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

TRT_LOGGER = trt.Logger()


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def preprocess_image(img_path):
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data

def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1


def save_output(image_name, pred, d_dir, o_dir):
	pred = pred.squeeze()
	pred = pred.cpu().data.numpy()
	#th = 0.2
	#pred[pred > th] = 1
	#pred[pred <= th] = 0

	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)

	mask = transform.resize(pred, (image.shape[0],image.shape[1]), anti_aliasing=False, mode = 'constant', order=0)
	mask = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
	#kernel = np.ones((3, 3), np.uint8)
	#mask = cv2.erode(mask, kernel, iterations=4)
	olay = image * mask


	#pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	io.imsave(o_dir+imidx+'.jpg', olay)
	io.imsave(d_dir + imidx + '.jpg', mask)


def build_engine(onnx_file_path):
# initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    #config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.max_workspace_size = 16 << 30
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    #builder.max_workspace_size = 16 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    # if builder.platform_has_fast_fp16:
    #
    #     builder.fp16_mode = True

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_serialized_network(network, config=config)
    #context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine


def main(makeengine=False):
    if makeengine:
        # initialize TensorRT engine and parse ONNX model
        serialized_engine = build_engine('basnet.onnx')
        #serialized_engine = engine.serialize()
        with open('basnet.engine', 'wb') as f:
            f.write(serialized_engine)

    with open('basnet.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    # preprocess input data
    #host_input = np.array(preprocess_image("ILSVRC2012_test_00000232.jpg").numpy(), dtype=np.float32, order='C')
    # for i in range(10):
    #     cuda.memcpy_htod_async(device_input, host_input, stream)
    #     # run inference
    #     start = time.time()
    #     context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    #     cuda.memcpy_dtoh_async(host_output, device_output, stream)
    #     stream.synchronize()
    #     print(time.time()-start)
    #     # postprocess results
    #     print(host_output.shape, output_shape)
    #     output_data = torch.Tensor(host_output).reshape(output_shape[2], output_shape[3])
    #     print(output_data.shape)
    #     pred = normPRED(output_data)
    #     print(pred)

    image_dir = '/home/hypevr/Desktop/data_0616/xy/other/image/'  # '/media/hypevr/KEY/tonaci_selected/'#'./test_data/test_images/'
    prediction_dir = '/home/hypevr/Desktop/data_0616/xy/other/mask/'  # '/media/hypev/KEY/tonaci_selected_masks/'
    olay_dir = '/home/hypevr/Desktop/data_0616/xy/other/olay/'  # '/media/hypevr/KEY/tonaci_selected_olay/'
    model_dir = './saved_models/basnet_bsi_human2_fr0.2_pb_0.2/basnet_213.pth'  # refine/
    plate_dir = '/home/hypevr/Desktop/data_0616/xy/3/back'
    test_loader = test_dataset(image_dir, image_dir, 352, True)
    for i in range(test_loader.size):
        image_orig, host_input, gt, name = test_loader.load_data()
        host_input = host_input.numpy()
        print(host_input.shape)
        host_input = np.tile(host_input, (1, 1, 1, 1))
        host_input = np.transpose(host_input, (0, 1, 2, 3))
        #host_input = host_input.transpose((0, 3, 1, 2))
        host_input = np.array(host_input, order='C')
        print(host_input.shape)
        #host_input = Variable(host_input)
        #print(host_input)
        ##inputs_test = data_test[0]
        cuda.memcpy_htod_async(device_input, host_input, stream)

        #inputs_test = inputs_test.type(torch.FloatTensor)
        start = time.time()
        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        print(time.time() - start)
        output_data = torch.Tensor(host_output).reshape(output_shape[0], output_shape[1], output_shape[2], output_shape[3])
        #pred = normPRED(output_data)
        # pred = overlay(image_resized, pred.squeeze().cpu().data.numpy())
        # save results to test_results folder
        save_output(image_dir + name, output_data, prediction_dir, olay_dir)


if __name__ == "__main__":
    # execute only if run as a script
    main(True)
