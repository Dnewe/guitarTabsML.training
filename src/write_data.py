import os
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from utils.fs_io import write_json, create_dir, write_ndarray_to_csv, write_lines_to_textfile
from typing import List
import numpy as np

dataconfig = DataConfig()
modelconfig = ModelConfig()

def write_metamodel(modeldir:str):
    write_json(os.path.join(modeldir, 'metamodel.json'), modelconfig.to_dict())

def write_metadata(modeldir:str):
    write_json(os.path.join(modeldir, 'metadata.json'), dataconfig.to_dict())


def create_modeldir(input:str, outdir:str):
    modeldir = os.path.join(outdir, "model_" + os.path.basename(os.path.dirname(input)))
    create_dir(modeldir)
    return modeldir


def write_modelcsvfiles(modeldir:str, W:List[np.ndarray]):
    for i in range(len(W)):
        write_ndarray_to_csv(os.path.join(modeldir, f"W{i+1}.csv"), W[i])


def write_infotxt(modeldir:str, acc:float, loss:float):
    configtxt = [f'Accuracy on test data: {round(acc,4)*100}%', 
                 f'Loss on test data: {round(loss,4)}']
    write_lines_to_textfile(os.path.join(modeldir, 'info.txt'), configtxt)


def write_data(data_path, outdir_path, model, acc, loss, dynLinePlot): 
    modeldir = create_modeldir(data_path, outdir_path)

    print('Saving Image..')
    dynLinePlot.save(os.path.join(modeldir,'acc_vs_it.png'))

    print("Writing model data..")
    W = [layer.W for layer in model.layers]
    write_modelcsvfiles(modeldir, W=W)
    write_metadata(modeldir)
    write_metamodel(modeldir)
    write_infotxt(modeldir, acc, loss)