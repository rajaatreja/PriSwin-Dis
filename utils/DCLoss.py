import torch
import torch.nn as nn
import torchvision.transforms as transforms


class DCLoss(nn.Module):
    def __init__(self, dis_model, device='cuda', reduction='mean'):
        """The discriminator loss which is intended to be used to realism in the images that are deformed
         during the anonymization process.

        :param dis_model: nn.Module
            A pre-trained discriminator (ResNet-18).
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        """
        super().__init__()
        self.dis_model = dis_model.to(device)
        self.device = device
        self.dis_model.eval()  # Start in eval, toggle to train if updating

        # Turn on gradient computation
        for param in self.dis_model.parameters():
            param.requires_grad = True

        self.reduction = reduction
        self.bce_loss = nn.BCELoss(reduction=self.reduction).cuda()
        
        # Properly configured transforms for single-channel images
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Update these values if different for your data
        ])

    def forward(self, real_image, deformed_image):
        real_labels = torch.ones(real_image.size(0), 1).cuda()
        fake_labels = torch.zeros(deformed_image.size(0), 1).cuda()

        # Convert and process images
        real_image = self.transform(real_image)
        deformed_image = self.transform(deformed_image)
        
        # Compute the discriminator output
        dis_r = self.dis_model(real_image)
        dis_f = self.dis_model(deformed_image)
        loss_r = self.bce_loss(dis_r, real_labels)
        loss_f = self.bce_loss(dis_f, fake_labels)

        return loss_r + loss_f
