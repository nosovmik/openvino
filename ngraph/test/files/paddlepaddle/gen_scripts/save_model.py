
import numpy as np
import paddle as pdpd


#print numpy array like C structure
def print_alike(arr):
    shape = arr.shape
    rank = len(shape)
    print("shape: ", shape, "rank: %d" %(rank))

    #for idx, value in np.ndenumerate(arr):
    #    print(idx, value)
    print(arr)
    
    def print_array(arr, end=' '):
        shape = arr.shape
        rank = len(arr.shape)
        if rank > 1:
            #print("{")
            line = "{"
            for i in range(arr.shape[0]):
                line += print_array(arr[i,:], end="},\n" if i < arr.shape[0]-1 else "}")
            line += end
            return line
        else:
            line = "{"           
            for i in range(arr.shape[0]):              
                line += str(arr[i])
                line += ", " if i < shape[0]-1 else ' '
            line += end
            #print(line)
            return line
            

    print(print_array(arr, "}"))


def saveModel(name, exe, feedkeys:list, fetchlist:list, inputs:list, outputs:list, **kwargv):
    for key, value in kwargv.items():
            print ("%s == %s" %(key, value))           

    print("\n\n------------- %s -----------\n" % (name))
    for i, input in enumerate(inputs):
        print("INPUT %s :" % (feedkeys[i]), input.shape, input.dtype)
        print_alike(input)
    print("\n")
    for i, output in enumerate(outputs):
        print("OUTPUT %s :" % (fetchlist[i]),output.shape, output.dtype)
        print_alike(output)            

    # composited model + scattered model
    pdpd.fluid.io.save_inference_model("../models/"+name, feedkeys, fetchlist, exe)
    pdpd.fluid.io.save_inference_model("../models/"+name, feedkeys, fetchlist, exe, model_filename=name+".pdmodel", params_filename=name+".pdiparams")   


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)  

    #x = np.random.randn(2,3).astype(np.float32)
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]], 
    [[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]]]).astype(np.float32)
    print_alike(x)  