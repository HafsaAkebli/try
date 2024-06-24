import torch



# define the device (GPU, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


#read the patches from the server that I created 
patches_sub="/home/akebli/test5/patches/"
