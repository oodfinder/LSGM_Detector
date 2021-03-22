import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

def odin_score(model, std, data, magnitude, temperature):
    model.eval()
    scores = []
    
    if True:
        data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (std[0]))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (std[1]))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (std[2]))

        tempInputs = torch.add(data.data,  -magnitude * gradient)
        with torch.no_grad():
            outputs = model(Variable(tempInputs))
            outputs = outputs / temperature
            soft_out = F.softmax(outputs, dim=1)
            soft_out, _ = torch.max(soft_out.data, dim=1)
        
        for i in range(data.size(0)):
            scores.append(soft_out[i].item())
    
    return np.array(scores)