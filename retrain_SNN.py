import os
import json
import argparse
from utils import utils
import shutil
from agents.AgentSiameseNetwork import AgentSiameseNetwork


if __name__ == "__main__":
    print('------------------------------------')
    print('------- Patient Verification -------')
    print('------------------------------------' + '\n')

    # Define an argument parser
    parser = argparse.ArgumentParser('Patient Verification')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_retrainSNN.json')
    parser.add_argument('--images_path', default='/images/')
    args = parser.parse_args()
    #print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config + '\n--images_path: ' + args.images_path + '\n')

    # Read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # Parse config
    config = json.loads(config)
    config['image_path'] = args.images_path

    # Create folder to save experiment-related files
    '''dir = './archive/' + config['experiment_description']
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    # os.mkdir('./archive/' + config['experiment_description'])
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
    utils.make_zip(SAVINGS_PATH + config['experiment_description'] + '.zip', './', args.config)'''

    os.chdir('/home/hpc/iwi5/iwi5156h/Discriminator/')

    # Call agent and run experiment
    experiment = AgentSiameseNetwork(config)
    experiment.run()
