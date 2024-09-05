import torch
import cv2
from torchvision.transforms import Compose,Normalize,ToTensor
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
model = torch.jit.load('facial_landmarks.pt')
model.to(torch.device(device=device))


dir = "test_images"
output_dir = "test_images_output"
os.mkdir(output_dir)


imgs = os.listdir(dir)

for img in imgs:
    image = cv2.imread(os.path.join(dir,img))
    h,w,_, = image.shape
    face = cv2.resize(image,(128,128))[:,:,::-1]
    face_tensor = transform(face.copy())
    lmarks = model(face_tensor.unsqueeze(0).to(torch.device(device=device)))
    for j in range(5):
            image = cv2.circle(image,(int(lmarks[0,2*j].item()*w/128) ,int(lmarks[0,2*j + 1].item()*h/128) ),5,(0,0,255),-1)
    cv2.imwrite(os.path.join(output_dir,img),image)