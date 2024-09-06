## CODE SUMMARY

This code defines a custom Convolutional Neural Network (CNN) architecture using PyTorch, with a complete pipeline for training and testing the model on a dataset. 
The CustomCNN class, which inherits from torch.nn.Module, consists of four convolutional layers, each followed by batch normalization, Leaky ReLU activation, and max pooling layers to downsample feature maps. 
After these layers, the feature maps are flattened, and a fully connected layer with dropout is applied to produce the final output. 


The model is designed for classification tasks, where the number of output classes is set using the num_classes parameter, and the forward pass defines how the input tensor flows through the network layers. 
The training pipeline includes train_step and test_step functions to compute loss and accuracy for both training and testing datasets, using PyTorch's device capabilities for either GPU or CPU execution. 
The results from each epoch, including training and test loss as well as accuracies, are stored in the results dictionary and are plotted side by side to visualize model performance over time.


Additionally, the AdamW optimizer is employed, along with a ReduceLROnPlateau scheduler, to adjust the learning rate when validation loss plateaus. 
The training loop is encapsulated in the train function, running for a set number of epochs and returning the results for further analysis.









