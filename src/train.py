import config
import os
from read_data import read_data
from gradient_descent import gradient_descent, forward_prop, get_accuracy, get_predictions
from utils.fs_io import create_dir, write_ndarray_to_csv, write_lines_to_textfile


def create_modeldir(input:str, outdir:str):
    modeldir = os.path.join(outdir, os.path.basename(input).replace('.csv',''))
    create_dir(modeldir)
    return modeldir


def write_modelcsvfiles(modeldir:str, **kwargs):
    for key, value in kwargs.items():
        write_ndarray_to_csv(os.path.join(modeldir, f"model_{key}.csv"), value)


def write_configtxt(modeldir:str):
    configtxt = [f'Train data: {config.TRAIN_PROP*100}%',
                 f'',
                 f'Iterations: {config.ITERATIONS}',
                 f'Alpha: {config.ALPHA}',
                 f'',
                 f'Layer1 size: {config.SIZE_LAYER1}']
    write_lines_to_textfile(os.path.join(modeldir, 'config.txt'), configtxt)


def run(args):
    datacsv_path = args.input
    outdir_path = args.output

    print("Reading data..")
    Y_dev, X_dev, Y_train, X_train = read_data(datacsv_path)

    print("Training..")
    w1, b1, w2, b2 = gradient_descent(X_train, Y_train, config.ITERATIONS, config.ALPHA)

    print("Finished training")
    _,_,_,a2 = forward_prop(w1,b1,w2,b2, X_dev)
    print(f"Accuracy on test set: {get_accuracy(get_predictions(a2), Y_dev)}")

    print("Writing model data..")
    modeldir = create_modeldir(datacsv_path, outdir_path)
    write_modelcsvfiles(modeldir, W1=w1, b1=b1, W2=w2, b2=b2)
    write_configtxt(modeldir)