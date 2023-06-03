import torch

from model import Flow1d, LogitTransform, FlowComposable1d
from data import load_data, load_data_classwise

def loss_function(target_distribution, z, log_dz_by_dx):
    z[z > 0.5] = z[z > 0.5] - 1e-4
    z[z < 0.5] = z[z < 0.5] + 1e-4
    print(z.min(), z.max())
    log_likelihood = target_distribution.log_prob(z) + log_dz_by_dx
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(epochs, lr, train_loader, test_loader, target_distribution):
    flow_models_list = [Flow1d(2951)]
    flow = FlowComposable1d(flow_models_list)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        train(flow, train_loader, optimizer, target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
        print("train loss : ", train_losses[-1])
        print("test loss : ", test_losses[-1])
        
    return flow, train_losses, test_losses