import data
import torchvision.transforms as transforms


# The dataset is already downloaded on the cluster
dataset_dir = "/mounts/Datasets2/Pascal-VOC2012/"
download = False

# How do we preprocess the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'none'}

# How do we preprocess the targets
target_transform_params = {'target_mode': 'preprocessed'}

# The post-processing of the image
image_transform = transforms.ToTensor()


train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)

# Display the first element of the dataset
# i.e. a pair made of an image and the slightly preprocessed targets
print(train_dataset[0])