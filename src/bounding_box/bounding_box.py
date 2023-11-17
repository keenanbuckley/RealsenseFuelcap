import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights

class BBoxModel():
    transform = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.transforms()

    def __init__(self, model_path) -> None:
        self.model = torch.load(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if str(self.device) == 'cuda':
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()
    
    def return_prediction(self, prediction):
        bbox = prediction['boxes'][0].cpu()
        score = prediction['scores'][0].cpu()
        return bbox, score

    def find_bbox(self, image):
        image = self.transform(image)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model([image])[0]
        return self.return_prediction(prediction)

    def find_bbox_many(self, images):
        for image in images:
            image = self.transform(image)
        images = [tensor.to(self.device) for tensor in images]
        with torch.no_grad():
            predictions = self.model(image)
        for pred in predictions:
            pred = self.return_prediction(pred)
        return predictions