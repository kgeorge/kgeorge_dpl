import sys
import os
import numpy as np
import urllib, tarfile
import re
import pickle




def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        d = pickle.load(fp)
    return d

class CifarData(object):
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    #regular expression that matches a datafile
    DOWNLOADED_DATA_FILEPATH_REG_EXPRESSION = re.compile('^data_batch_\d+')
    #cifar-10 consist of 32 x 32 x 3 rgb images
    LOADED_IMG_HEIGHT = 32
    LOADED_IMG_WIDTH = 32
    LOADED_IMG_DEPTH = 3

    def __init__(self):
        #training and validate datasets as numpy n-d arrays,
        #apropriate portions of which are ready to be fed to the placeholder variables
        self.train_dataset={'data':[], 'labels':[]}
        self.validate_dataset={'data':[], 'labels':[]}
        self.test_dataset={'data':{}, 'labels':[]}
        self.label_names_for_validation_and_test=None
        self.verbose=True
        pass

    def v_print_(self, *args):
        if self.verbose:
            print(args)

    def maybe_download_and_extract(self, dest_directory):
        """Download and extract the tarball from Alex's website."""
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = CifarData.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.urlretrieve(CifarData.DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        self.v_print_('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        with tarfile.open(filepath, 'r:gz') as t:
            dataset_dir = os.path.join(dest_directory, t.getmembers()[0].name)
            t.extractall(dest_directory)
        return dataset_dir


    def convert_to_rgb_img_data(self, index=-1, data=None):
        assert(index < data.shape[0])
        image_holder = np.zeros(shape=[data.shape[1],data.shape[2], data.shape[3]], dtype=np.float32)
        image_holder[:, :, :] = data[index, :, :, :]
        plt.imshow(image_holder)



    def prepare_input_ (self, data=None, labels=None):
        epsilon=1.0e-8
        scale=55.0
        assert(data.shape[1] == CifarData.LOADED_IMG_HEIGHT * CifarData.LOADED_IMG_WIDTH * CifarData.LOADED_IMG_DEPTH)
        assert(data.shape[0] == labels.shape[0])
        #do mean normaization across all samples
        mu = np.mean(data, axis=1)[:, np.newaxis]
        assert(mu.shape == (data.shape[0], 1))
        data = data - mu
        normalizer = np.sqrt((data**2).sum(axis=1))/scale
        normalizer[normalizer < epsilon] = 1.0
        assert(normalizer.shape[0]== (data.shape[0]))
        normalizer = normalizer[:, np.newaxis]
        data = data/normalizer

        data = data.reshape([-1,CifarData.LOADED_IMG_DEPTH, CifarData.LOADED_IMG_HEIGHT, CifarData.LOADED_IMG_WIDTH])
        data = data.transpose([0, 2, 3, 1])
        data = data.astype(np.float32)
        return data, labels

    def post_load_summary_(self):
        def get_details_dataset(dataset):
            return 'data:  shape={sh}  , dtype={dt}, labels: shape={lsh}, dtype={ldt}'.format(sh=dataset['data'].shape, dt=dataset['data'].dtype, lsh=dataset['labels'].shape, ldt=dataset['labels'].dtype)
        self.v_print_ ('train: %s' % get_details_dataset(self.train_dataset))
        self.v_print_ ('validate: %s' % get_details_dataset(self.validate_dataset))
        self.v_print_ ('test: %s' % get_details_dataset(self.test_dataset))
        self.v_print_ ('label names for images used %r' % self.label_names_for_validation_and_test)

    def load_and_preprocess_input(self, dataset_dir=None,
                                  n_train_samples=-1,
                                  n_validate_samples=-1,
                                  n_test_samples=-1):
        assert(os.path.isdir(dataset_dir))
        #need to have a valid value for number of  validate and test samples
        assert(n_test_samples >= 0)
        assert(n_validate_samples >=0)
        trn_all_data=[]
        trn_all_labels=[]
        vldte_and_test_data=[]
        vldte_and_test_labels=[]

        #for loading train dataset, iterate through the directory to get matching data file
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                m=CifarData.DOWNLOADED_DATA_FILEPATH_REG_EXPRESSION.match(f)
                if m:
                    relpath = os.path.join(root, f)
                    d=unpickle(os.path.join(root, f))
                    trn_all_data.append(d['data'])
                    trn_all_labels.append(d['labels'])
        #concatenate all the  data in various files into one ndarray of shape
        #data.shape == (no_of_samples, 3072), where 3072=CifarData.LOADED_IMG_DEPTH x CifarData.LOADED_IMG_HEIGHT x CifarData.LOADED_IMG_WIDTH
        #labels.shape== (no_of_samples)
        trn_all_data, trn_all_labels = (np.concatenate(trn_all_data).astype(np.float32),
                                              np.concatenate(trn_all_labels).astype(np.int32))

        #we need only n_train_samples                                            )
        if n_train_samples >= 0:
            trn_all_data = trn_all_data[0:n_train_samples,:]
            trn_all_labels = trn_all_labels[0:n_train_samples,:]

        #load the only test data set for validation and testing
        #use only the first n_validate_samples samples for validating
        test_temp=unpickle(os.path.join(dataset_dir, 'test_batch'))
        assert((n_validate_samples+n_test_samples) < test_temp['data'].shape[0])
        vldte_and_test_data=test_temp['data'][0:(n_validate_samples+n_test_samples), :]
        vldte_and_test_labels=test_temp['labels'][0:(n_validate_samples+n_test_samples)]
        vldte_and_test_data, vldte_and_test_labels =  (np.concatenate([vldte_and_test_data]).astype(np.float32),
                                                 np.concatenate([vldte_and_test_labels]).astype(np.int32))
         #transform the test images in the same manner as the train images
        self.train_dataset['data'], self.train_dataset['labels'] = self.prepare_input_(data=trn_all_data, labels=trn_all_labels)
        validate_and_test_data, validate_and_test_labels = self.prepare_input_(data=vldte_and_test_data, labels=vldte_and_test_labels)

        self.validate_dataset['data'] = validate_and_test_data[0:n_validate_samples, :, :, :]
        self.validate_dataset['labels'] = validate_and_test_labels[0:n_validate_samples]
        self.test_dataset['data'] = validate_and_test_data[n_validate_samples:(n_validate_samples+n_test_samples), :, :, :]
        self.test_dataset['labels'] = validate_and_test_labels[n_validate_samples:(n_validate_samples+n_test_samples)]
        
        
        #load all label-names
        self.label_names_for_validation_and_test=unpickle(os.path.join(dataset_dir, 'batches.meta'))['label_names'] 
    
        self.post_load_summary_()


if __name__ == "__main__":
    pass
