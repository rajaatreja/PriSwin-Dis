import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.Discriminator import Discriminator
from networks.UNet_PrivacyNet import Unet2D_encoder

class AgentPretrain:
    def __init__(self, config):
        """This is the agent that provides code for pre-training the flow field generator of the anonymization model.

        :param config: dict
            A dictionary that stores the hyper-parameter configuration and some other important variables.
        """

        self.config = config

        # Set path used to save experiment-related files and results
        self.SAVINGS_PATH = './archive/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']

        # Reproducibility
        utils.seed_all(42)

        # Set all the important variables
        self.generator_type = self.config['generator_type']
        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']
        self.mu = self.config['mu']

        self.num_workers = 8
        self.pin_memory = True

        self.show_every_n_epochs = self.config['show_every_n_epochs']
        self.show_every_n_iterations = self.config['show_every_n_iterations']

        self.writer = SummaryWriter(self.SAVINGS_PATH + 'runs/')

        if self.generator_type == 'flow_field':
            # Define the identity grid
            d = torch.linspace(-1, 1, self.image_size)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            # Define the Gauss filter which is used for smoothing the resulting flow field
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
            # Define the generator
            self.generator = UNet(1, 2, 32).cuda()
        elif self.generator_type == 'privacy_net':
            # Set identity grid and gauss filter to None
            self.grid_identity = None
            self.gauss_filter = None
            # Define the generator
            self.generator = Unet2D_encoder(1, 1, 16).cuda()
        else:
            raise Exception('Invalid argument: ' + self.generator_type)
        
        self.discriminator = Discriminator().cuda()

        self.start_epoch = 0
        self.best_loss_g = 10000
        self.best_loss_d = 10000

        self.loss_dict = {
            'training': {
                'gen': [],
                'dis': []
            },
            'validation': {
                'gen': [],
                'dis': []
            }
        }

        # Define the loss function
        self.generator_loss = nn.MSELoss().cuda()
        self.discriminator_loss = nn.BCELoss().cuda()

        # Set the optimizer function
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        # Initialize data loaders
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='pretrain', 
                                                     image_size=self.image_size, n_channels=1, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                     image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='pretrain', 
                                                       image_size=self.image_size, n_channels=1, 
                                                       batch_size=self.batch_size, shuffle=False,
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                       image_path=self.IMAGE_PATH)

    def training_validation(self):
        # Training and validation loop
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            # Pre-train the model
            train_losses = utils.pretrain(self.generator, self.discriminator, self.training_loader, self.gauss_filter, self.grid_identity,
                                        self.mu, self.generator_loss, self.discriminator_loss, self.optimizer_g, self.optimizer_d, epoch, self.max_epochs,
                                        self.show_every_n_epochs, self.show_every_n_iterations, self.SAVINGS_PATH)

            # Pre-validate the model
            val_losses = utils.preval(self.generator, self.discriminator, self.validation_loader, self.gauss_filter, self.grid_identity,
                                    self.mu, self.generator_loss, self.discriminator_loss, epoch, self.max_epochs, self.show_every_n_epochs,
                                    self.show_every_n_iterations, self.SAVINGS_PATH)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            # Append losses to dict
            self.loss_dict['training']['gen'].append(train_losses[0])
            self.loss_dict['training']['dis'].append(train_losses[1])
            self.loss_dict['validation']['gen'].append(val_losses[0])
            self.loss_dict['validation']['dis'].append(val_losses[1])

            # Plot loss curves
            for phase in ['training', 'validation']:
                plt.figure()
                plt.plot(range(1, len(self.loss_dict[phase]['gen']) + 1), self.loss_dict[phase]['gen'], label='gen_loss')
                plt.plot(range(1, len(self.loss_dict[phase]['dis']) + 1), self.loss_dict[phase]['dis'], label='dis_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                if phase == 'training':
                    plt.title('Training loss curves')
                elif phase == 'validation':
                    plt.title('Validation loss curves')
                plt.legend()
                plt.savefig(self.SAVINGS_PATH + 'loss_curves_' + phase + '.png')
                plt.close()

            # Save loss dict
            utils.save_loss_dict(self.loss_dict, self.SAVINGS_PATH)

            # Save checkpoint
            if val_losses[0] < self.best_loss_g:
                self.best_loss_g = val_losses[0]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_best.pth')
            if val_losses[1] < self.best_loss_d:
                self.best_loss_d = val_losses[1]
                torch.save(self.discriminator.state_dict(), self.SAVINGS_PATH + 'discriminator_best.pth')

        print('Finished Training!')

    def run(self):
        # Call training/validation loop
        self.training_validation()
