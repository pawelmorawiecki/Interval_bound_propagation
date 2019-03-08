import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1) 

def epoch(loader, model, device, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp,_ = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def bound_propagation(model, initial_bound, how_many_layers):
    l, u = initial_bound
    bounds = []
    bounds.append(initial_bound)
    list_of_layers = list(model.children())
    
    for i in range(how_many_layers):
        layer = list_of_layers[i]
        
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)

        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() 
                  + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() 
                  + layer.bias[:,None]).t()
            
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None,:,None,None])
            
            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) + 
                  layer.bias[None,:,None,None])
            
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            
        bounds.append((l_, u_))
        l,u = l_, u_
    return bounds


def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    cW = c.t() @ model.last_linear.weight
    cb = c.t() @ model.last_linear.bias
    
    l,u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:,None]).t()


def epoch_robust_bound(loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, opt=None):
    robust_err = 0
    total_robust_loss = 0
    total_combined_loss = 0
    
    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0,:] += 1
    
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        ###### fit loss calculation ######
        yp = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp,y)
    
        ###### robust loss calculation ######
        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])
        bounds = bound_propagation(model, initial_bound)
        robust_loss = 0
        for y0 in range(10):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0]) / X.shape[0]
                
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning       
        
        total_robust_loss += robust_loss.item() * X.shape[0]  
    
        ###### combined losss ######
        combined_loss = kappa_schedule[batch_counter]*fit_loss + (1-kappa_schedule[batch_counter])*robust_loss
        total_combined_loss += combined_loss.item()
        
        batch_counter +=1
        
      
        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()
            
    return robust_err / len(loader.dataset), total_combined_loss / len(loader.dataset)



def new_epoch_robust_bound(loader, model, epsilon, device, opt=None):
    
    robust_err = 0
    total_robust_loss = 0
    total_fit_loss = 0
    
    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0,:] += 1

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        ###### fit loss calculation ######
        yp,_ = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp,y)
    
        ###### robust loss calculation ######
        initial_bound = (X - epsilon, X + epsilon)
        bounds = bound_propagation(model, initial_bound, how_many_layers=14)
        robust_loss = 0
        for y0 in range(10):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0]) / X.shape[0]        
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning       
        
        total_robust_loss += robust_loss.item() * X.shape[0]
        total_fit_loss += fit_loss.item() * X.shape[0]
        
                ###### combined losss ######
        combined_loss = fit_loss + robust_loss
      
        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()
            
    return total_fit_loss / len(loader.dataset), total_robust_loss / len(loader.dataset)
        



def epoch_calculate_robust_err (loader, model, epsilon, how_many_layers, device):
    robust_err = 0.0
    
    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0,:] += 1


    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        initial_bound = (X - epsilon, X + epsilon)
        bounds = bound_propagation(model, initial_bound, how_many_layers)

        for y0 in range(10):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)                
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning       
        
    return robust_err / len(loader.dataset)
        
        


def generate_kappa_schedule_MNIST():

    kappa_schedule = 2000*[1] # warm-up phase
    kappa_value = 1.0
    step = 0.5/58000
    
    for i in range(58000):
        kappa_value = kappa_value - step
        kappa_schedule.append(kappa_value)
    
    return kappa_schedule

def generate_epsilon_schedule_MNIST(epsilon_train):
    
    epsilon_schedule = []
    step = epsilon_train/10000
            
    for i in range(10000):
        epsilon_schedule.append(i*step) #ramp-up phase
    
    for i in range(50000):
        epsilon_schedule.append(epsilon_train)
        
    return epsilon_schedule


def generate_kappa_schedule_CIFAR():

    kappa_schedule = 10000*[1] # warm-up phase
    kappa_value = 1.0
    step = 0.5/340000
    
    for i in range(340000):
        kappa_value = kappa_value - step
        kappa_schedule.append(kappa_value)
    
    return kappa_schedule

def generate_epsilon_schedule_CIFAR(epsilon_train):
    
    epsilon_schedule = []
    step = epsilon_train/150000
            
    for i in range(150000):
        epsilon_schedule.append(i*step) #ramp-up phase
    
    for i in range(200000):
        epsilon_schedule.append(epsilon_train)
        
    return epsilon_schedule 