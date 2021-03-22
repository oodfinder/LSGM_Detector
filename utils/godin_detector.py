import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def godin_score(model, images, noise_magnitude, score_func = 'h'):
    model.eval()

    # process
    images = Variable(images, requires_grad = True)
    logits, h, g = model(images)

    if score_func == 'h':
        scores = h
    elif score_func == 'g':
        scores = g
    elif score_func == 'logit':
        scores = logits

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of the numerator w.r.t. input

    max_scores, _ = torch.max(scores, dim = 1)
    max_scores.backward(torch.ones(len(max_scores)).cuda())

    # Normalizing the gradient to binary in {-1, 1}
    if images.grad is not None:
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
        gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
        gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)
    
        # Now calculate score
        logits, h, g = model(tempInputs)

        if score_func == 'h':
            scores = h
        elif score_func == 'g':
            scores = g
        elif score_func == 'logit':
            scores = logits

    results = torch.max(scores, dim=1)[0].data.cpu().numpy()
    return results
