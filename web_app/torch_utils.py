import torch 
import torch.nn as nn
from torchvision import transforms

import io
from PIL import  Image
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = ['1', '2', '3', '4', '5', '6', '7']

#initialize model
resnet34_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
resnet34_model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
resnet34_model.fc = nn.Linear(in_features= 512, out_features=7, bias=True)

#load pickel model
with open('./models/resnet34_model.pkl', 'rb') as model_file:
    resnet34_model = pickle.load(model_file)

# model to eval mode
resnet34_model.eval()


#pre processing image 
def transform_image(image_byte) : 
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)), 
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485], std=[0.225])
    ])

    image = Image.open(io.BytesIO(image_byte))
    return transform(image).unsqueeze(0)

#predict image
def prediction(image_tensor) : 
    with torch.no_grad() :
        image_tensor = image_tensor.to(device) 
        outputs = resnet34_model(image_tensor)
        _, predicted = torch.max(outputs, 1) 
        predicted_class_name = classes[predicted]

    return predicted_class_name