# Make a paired circles and squares toy dataset for multimodal encoding
import os

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

dataset_size = 1000
size_image = 32
min_rayon, max_rayon = 0.3, 0.9
circle_thickness = 0.25
n_repeat = 10
output_path = 'data'
if not os.path.exists(output_path):
    os.mkdir(output_path)

rayons = np.linspace(min_rayon,max_rayon,dataset_size)
x = np.linspace(-1,1,size_image)

def circle(X,Y,r):
    return (X**2 + Y**2 <= (r + circle_thickness/2)**2)*(X**2 + Y**2 >= (r - circle_thickness/2)**2)

def square_line(X,Y,r):
    return (np.abs(X) + np.abs(Y) <= (r + circle_thickness/2))*(np.abs(X) + np.abs(Y) >= r - circle_thickness/2)

squares, r_squares = [], []
circles,r_circles = [],[]
labels = []

for i, r_disc in enumerate(rayons):
    for _ in range(n_repeat):
        X,Y = np.meshgrid(x,x)
        r_circles.extend([np.random.uniform(min_rayon,max_rayon) for _ in range(2)])
        r_squares.extend([np.random.uniform(min_rayon, max_rayon) for _ in range(2)])
        # Associate a random-sized disc to a random-sized full square
        img_full_disc = X**2 + Y**2 <= r_circles[-2]**2
        img_full_square = np.abs(X) + np.abs(Y) <= r_squares[-2]
        # And a random-sized ring to a random sized line-square
        img_empty_disc = circle(X,Y,r_circles[-1])
        img_empty_square = square_line(X,Y,r_squares[-1])

        squares.extend([img_full_square, img_empty_square])
        circles.extend([img_full_disc, img_empty_disc])
        labels.extend([1,0])


# Visualize some examples
output_examples = output_path + '/examples'
if not os.path.exists(output_examples):
    os.mkdir(output_examples)

for i in np.linspace(0,dataset_size*n_repeat-1, 100):
    i = int(i)
    img = Image.fromarray(np.concatenate([squares[i], circles[i]]))
    img.save(output_examples + f'/example_{i}.png')



# Save in pytorch format
squares = np.expand_dims(np.array(squares), 1)
circles = np.expand_dims(np.array(circles), 1)
labels, r_squares, r_circles = torch.tensor(labels),torch.tensor(r_squares), torch.tensor(r_circles)
# Select some for training and testing
s_train, s_test, c_train, c_test, idx_train, idx_test = train_test_split(squares,circles, np.arange(len(labels)), test_size=0.3)

if not os.path.exists(output_path+'/squares'):
    os.mkdir(output_path + '/squares')
if not os.path.exists(output_path+'/circles'):
    os.mkdir(output_path + '/circles')

np.savez(output_path + '/squares/train_data', data = s_train, rayon = r_squares[idx_train], labels=labels[idx_train])
np.savez(output_path + '/squares/eval_data', data = s_test, rayon = r_squares[idx_test], labels=labels[idx_test])
np.savez(output_path + '/circles/train_data', data = c_train, rayon = r_circles[idx_train], labels=labels[idx_train] )
np.savez(output_path + '/circles/eval_data', data = c_test, rayon = r_circles[idx_test], labels = labels[idx_test])


print(c_train.shape, c_test.shape)




