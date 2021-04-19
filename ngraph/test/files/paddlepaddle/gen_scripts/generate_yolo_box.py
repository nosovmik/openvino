#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

data_type = 'float32'

def yolo_box(name : str, x, img_size, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        node_img_size = pdpd.static.data(name='img_size', shape=img_size.shape, dtype='int32')
        boxes, scores = pdpd.vision.ops.yolo_box(node_x,
                                                node_img_size,
                                                anchors=attrs['anchors'],
                                                class_num=attrs['class_num'],
                                                conf_thresh=attrs['conf_thresh'],
                                                downsample_ratio=attrs['downsample_ratio'],
                                                clip_bbox=attrs['clip_bbox'],
                                                name=None, 
                                                scale_x_y=attrs['scale_x_y'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'img_size': img_size},
            fetch_list=[boxes, scores])             

        saveModel(name, exe, feedkeys=['x', 'img_size'], fetchlist=[boxes, scores], inputs=[x, img_size], outputs=outs)

    return

def main():
    # yolo_box
    pdpd_attrs = {
            'anchors': [116, 90, 156, 198, 373, 326],
            'class_num': 2,
            'conf_thresh': 0.80,
            'downsample_ratio': 32,
            'clip_bbox': True,
            'scale_x_y': 1.0
    }

    num_anchors = int(len(pdpd_attrs['anchors'])/2)
    N, C, H, W = 1, (num_anchors * (5+pdpd_attrs['class_num'])), 3, 3

    data = np.arange(N*C*H*W).astype(data_type)
    data_NCHW = data.reshape(N, C, H, W)
    data_ImSize = np.arange(N*2).astype('int32').reshape(N, 2)
    print(data_NCHW.shape)    

    yolo_box('yolo_box_test1', data_NCHW, data_ImSize, pdpd_attrs)   


if __name__ == "__main__":
    main()     