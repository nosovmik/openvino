#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def pool2d(name : str, x, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    exclusive = False
    if 'exclusive' in attrs:
        exclusive = attrs['exclusive']
    global_pool=False
    if 'global_pool' in attrs:
        global_pool = attrs['global_pool']

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    out = pdpd.fluid.layers.pool2d(node_x, pool_size=attrs['kernel_size'],
                                   pool_type=attrs['type'],
                                   pool_stride=attrs['stride'],
                                   pool_padding=attrs['pool_padding'],
                                   global_pooling=global_pool,
                                   exclusive=exclusive)

    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])             

    saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]


def main():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]]).astype(np.float32)

    # maxPool
    pdpd_max_attrs = {
        'kernel_size': [3, 3],
        'type': 'max',
        'stride': 1,
        'pool_padding': 0
    }    
    pool2d("maxPool", data, pdpd_max_attrs)

    # maxGlobalPool
    spatial_shape = np.ndim(data)
    pdpd_max_global_attrs = {
        'kernel_size': [spatial_shape, spatial_shape],
        'type': 'max',
        'stride': 1,
        'pool_padding': 0,
        'global_pool': True}

    pool2d("maxGlobalPool", data, pdpd_max_global_attrs)

    # avgPool
    pdpd_avg_attrs = {'kernel_size': [3, 3],
                      'type': 'avg',
                      'stride': 1,
                      'pool_padding': 1,
                      'exclusive': True}
    ng_avg_attrs = {
        'kernel_size': [3, 3],
        'type': 'avg',
        'stride': [1, 1],
        'padding': [1, 1],
        'exclude_pad': True
    }
    pool2d("avgPool", data, pdpd_avg_attrs)   

if __name__ == "__main__":
    main()     