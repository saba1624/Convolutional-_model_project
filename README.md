# Convolutional-_model_project
1. Convolutional model for object recognition in images, designed for drones or robots, utilizing a classical convolutional model and YOLO 6
2. Change the storage path for image files on lines 13 and 14. Modify the trainset and testset variables as follows:
   
   trainset = datasets.CIFAR10(root='./new_data_folder', train=True, download=True, transform=transform)
   testset = datasets.CIFAR10(root='./new_data_folder', train=False, download=True, transform=transform)
3. I am continuously uploading the material; the project's deadline is August 5th, 2023.
