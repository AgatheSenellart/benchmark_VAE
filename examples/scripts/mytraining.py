import numpy as np
import torch
from pythae.pipelines import TrainingPipeline
from pythae.models import VAE_LinNF as VAE
from pythae.models import VAE_LinNF_Config as VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.samplers import NormalSampler
from pythae.data.preprocessors import DataProcessor
from torch.utils.data import DataLoader
from vis import plot_embeddings_colorbars

data_path = 'data/circles/'
train, eval = np.load(data_path + 'train_data.npz'), np.load(data_path + 'eval_data.npz')


my_training_config = BaseTrainerConfig(
    output_dir = 'my_model',
    num_epochs = 50,
    learning_rate=1e-3,
    batch_size=256,
    steps_saving=None
)

my_vae_config = model_config = VAEConfig(input_dim = (1,32,32)
                                         ,latent_dim=2)
my_vae_model = VAE(model_config=my_vae_config)

pipeline = TrainingPipeline(training_config=my_training_config,
                            model=my_vae_model)
# pipeline(train_data = train['data'],eval_data= eval['data'])
dir = 'my_model/VAE_LinNF_training_2022-05-24_11-18-35/'
my_trained_vae = VAE.load_from_folder(dir + 'final_model')


# Post training analysis
my_sampler = NormalSampler(model=my_trained_vae)
gen_data = my_sampler.sample(
    num_samples=20,
    batch_size=10,
    # output_dir=dir + 'samples',
    output_dir=None,
    return_gen=True
)
print(gen_data.shape)
data_processor = DataProcessor()
eval_process = data_processor.process_data(eval['data'])
eval = data_processor.to_dataset(eval_process, labels = eval['rayon'])
eval_loader = DataLoader(eval,batch_size = 256, shuffle=True)
for i, batch in enumerate(eval_loader):
    batch['data'] = batch['data'].cuda()
    print(torch.unique(batch['labels']))
    latents = my_trained_vae.forward(batch).__getitem__('z').cpu().detach().numpy()
    if i ==0:
        plot_embeddings_colorbars(latents,batch['labels'], dir + 'latents.png')

