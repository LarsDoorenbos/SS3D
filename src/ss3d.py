import os
import math
import torch.nn as nn
import pickle
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
from efficientnet_pytorch import EfficientNet
import click
import numpy as np
import torch
import utils
import numpy_dataset
from nifti_io import ni_load, ni_save
from evalresults import eval_dir

sigmoid = lambda x: 1 / (1 + math.exp(-x))

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)                

class SS3D:
    def __init__(
        self,
        input_shape,
        data_dir=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.data_dir = data_dir
        self.input_shape = input_shape

        network = EfficientNet.from_pretrained('efficientnet-b0')

        network.eval()

        if hasattr(network, "fc"):
            network.fc = nn.Identity()

        network = nn.DataParallel(network)
        network.to(self.device)

        self.model = network

    def train(self, dataset):
        _, trainset = numpy_dataset.get_numpy3d_dataset(
            base_dir=self.data_dir,            
            num_processes=8,
            pin_memory=True,
            batch_size=64,
            mode="train",
            target_size=self.input_shape[2],
        )

        print(len(trainset))

        for dim in range(0, 3):
            print('Dim: ', dim)
            means = {}
            covs = {}

            stepsize = 120
            for sli in range(0, trainset[0].shape[dim], stepsize):
                for idx in range(len(trainset)):
                    volume = trainset[idx]
                    if dim == 2:
                        x = volume[:, :, sli:sli + stepsize]
                        imgs = torch.cat([np.repeat(x[:, :, i][None, None], 3, axis=1) for i in range(x.shape[2])])                 
                    elif dim == 1:
                        x = volume[:, sli:sli + stepsize, :]
                        imgs = torch.cat([np.repeat(x[:, i][None, None], 3, axis=1) for i in range(x.shape[1])])
                    else:
                        x = volume[sli:sli + stepsize, :, :]
                        imgs = torch.cat([np.repeat(x[i][None, None], 3, axis=1) for i in range(x.shape[0])])

                    imgs[:, 0] = (imgs[:, 0] - 0.485) / 0.229
                    imgs[:, 1] = (imgs[:, 1] - 0.456) / 0.224
                    imgs[:, 2] = (imgs[:, 2] - 0.406) / 0.225

                    if idx == 0:
                        spectra = utils.get_single_spectrum(imgs, self.model)[None]
                    else:
                        spectra = np.concatenate((spectra, utils.get_single_spectrum(imgs, self.model)[None]), axis=0)

                mean = np.mean(spectra, axis=0)
                print(mean.shape)
                
                for i in range(mean.shape[0]):
                    offset = 0
                    while offset < spectra.shape[0]:
                        try:
                            covI = LedoitWolf().fit(spectra[offset:, i]).precision_                 
                            break
                        except:
                            print('fail', str(sli + i), 'now trying', offset)
                            offset += 1
                            
                    means[str(sli + i)] = mean[i]
                    covs[str(sli + i)] = covI   

            save_obj(means, dataset + '/means' + str(dim))
            save_obj(covs, dataset + '/covs' + str(dim))

    def score_pixels(self, np_array, size, name, dataset):
        tr = numpy_dataset.get_transforms_3d(size)
        np_array = tr(np_array[None])

        score = np.zeros(np_array[0].shape) + 10000000
        sample_score = 0

        for dim in range(3):
            blackList = []
            scoreList = []
            means = load_obj(dataset + '/means' + str(dim))
            covs = load_obj(dataset + '/covs' + str(dim))
            
            stepsize = 75
            for sli in range(0, np_array[0].shape[dim], stepsize):
                if dim == 2:
                    x = np_array[0][:, :, sli:sli + stepsize]
                    imgs = torch.cat([np.repeat(x[:, :, i][None, None], 3, axis=1) for i in range(x.shape[2])])
                elif dim == 1:
                    x = np_array[0][:, sli:sli + stepsize, :]
                    imgs = torch.cat([np.repeat(x[:, i][None, None], 3, axis=1) for i in range(x.shape[1])])
                else:
                    x = np_array[0][sli:sli + stepsize, :, :]
                    imgs = torch.cat([np.repeat(x[i][None, None], 3, axis=1) for i in range(x.shape[0])])

                for idx in range(len(imgs)):
                    blackList.append(len(np.argwhere(imgs[idx, 0] < 0.01).flatten()) / (224*224) * 50)
            
                imgs[:, 0] = (imgs[:, 0] - 0.485) / 0.229
                imgs[:, 1] = (imgs[:, 1] - 0.456) / 0.224
                imgs[:, 2] = (imgs[:, 2] - 0.406) / 0.225

                spectra = utils.get_single_spectrum(imgs, self.model)   
                print(spectra.shape)

                for idx in range(len(imgs)):
                    if blackList[sli + idx] > 95:
                        maha = 0
                    else:
                        maha = 10000000

                        for window in range(0, 1):
                            loc = sli + idx + window
                            if loc > -1 and loc < 224:
                                tempMaha = mahalanobis(spectra[idx], means[str(loc)], covs[str(loc)])
                                maha = min(tempMaha, maha)

                    scoreList.append(maha)

            del means
            del covs

            sample_score += np.sum(np.array(scoreList))

            scoreList = savgol_filter(scoreList, 21, 3)

            for idx in range(score.shape[0]):
                if dim == 0:
                    score[idx, :, :] = np.minimum(score[idx, :, :], scoreList[idx])
                elif dim == 1:
                    score[:, idx, :] = np.minimum(score[:, idx, :], scoreList[idx])
                elif dim == 2:
                    score[:, :, idx] = np.minimum(score[:, :, idx], scoreList[idx])

        score = score / 300
        if dataset == 'brain':
            score = zoom(score, (256 / 224, 256 / 224, 256 / 224))
        else:
            score = zoom(score, (512 / 224, 512 / 224, 512 / 224))

        return score, sigmoid((sample_score - 10000) / 20000)

@click.option(
    "-r", "--run", default="train", type=click.Choice(["train", "predict", "test", "all", "predtest"], case_sensitive=False)
)
@click.option("--target-size", type=click.IntRange(1, 512, clamp=True), default=224)
@click.option("-t", "--test-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-p", "--pred-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-s", "--dataset", type=click.Choice(["brain", "abdominal"], case_sensitive=False))
@click.command()
def main(
    mode="pixel",
    run="train",
    target_size=224,
    test_dir=None,
    pred_dir=None,
    data_dir=None,
    dataset=None
):

    input_shape = (8, 1, target_size, target_size)

    maha3d_algo = SS3D(
        input_shape,
        data_dir=data_dir,
    )

    if run == "train" or run == "all":
        print('Training')
        maha3d_algo.train(dataset)

    if run == "predict" or run == "all" or run == 'predtest':
        if pred_dir is None:
            pred_dir = "predictions"
            os.makedirs(pred_dir, exist_ok=True)
        elif pred_dir is None:
            print("Please either give a log/ output dir or a prediction dir")
            exit(0) 

        for idx, f_name in enumerate(os.listdir(test_dir)):
            ni_file = os.path.join(test_dir, f_name)
            ni_data, ni_aff = ni_load(ni_file)
            print(ni_file)

            pixel_scores, sample_score = maha3d_algo.score_pixels(ni_data, target_size, str(idx), dataset)
            ni_save(os.path.join(pred_dir, f_name), pixel_scores, ni_aff)

            with open(os.path.join(pred_dir, f_name + ".txt"), "w") as target_file:
                target_file.write(str(sample_score))       

    if run == "test" or run == "all" or run == 'predtest':

        if pred_dir is None:
            print("Please either give a prediction dir")
            exit(0)
        if test_dir is None:
            print(
                "Please either give a test dir which contains the test samples "
                "and for which a test_dir_label folder exists"
            )
            exit(0)

        mode = 'sample'
        test_dir = test_dir[:-1] if test_dir.endswith("/") else test_dir
        AUCscore, APscore = eval_dir(pred_dir=pred_dir, label_dir=test_dir + f"_label/{mode}", mode=mode)

        print('Sample level score: AUC:', AUCscore * 100, 'AP:', APscore * 100)
        mode = 'pixel'
        test_dir = test_dir[:-1] if test_dir.endswith("/") else test_dir
        AUCscore, APscore = eval_dir(pred_dir=pred_dir, label_dir=test_dir + f"_label/{mode}", mode=mode)

        print('Pixel level score: AUC:', AUCscore * 100, 'AP:', APscore * 100)


if __name__ == "__main__":

    main()
