
import numpy as np
import paddle as pdpd

def saveModel(name, exe, feedkeys:list, fetchlist:list, inputs:list, outputs:list, **kwargv):
    for key, value in kwargv.items():
            print ("%s == %s" %(key, value))

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)                

    print("\n\n------------- %s -----------\n" % (name))
    for i, input in enumerate(inputs):
        print("INPUT %s :" % (feedkeys[i]), input.shape, input.dtype)
        print(input)
    print("\n")
    for i, output in enumerate(outputs):
        print("OUTPUT %s :" % (fetchlist[i]),output.shape, output.dtype)
        print(output)            

    # composited model + scattered model
    pdpd.fluid.io.save_inference_model("../models/"+name, feedkeys, fetchlist, exe)
    pdpd.fluid.io.save_inference_model("../models/"+name, feedkeys, fetchlist, exe, model_filename=name+".pdmodel", params_filename=name+".pdiparams")   