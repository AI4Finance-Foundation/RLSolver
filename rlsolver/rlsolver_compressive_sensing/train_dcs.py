import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from file_utils import *
from copy import deepcopy
from tqdm import tqdm
import sys
from torch.autograd import Variable, grad
class Generator(nn.Module):
    def __init__(self, in_dim=100, out_dim=28**2, mid_dim = 512):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim),nn.Tanh()
        )


    def forward(self, input):
        return self.net(input)

class Metric(nn.Module):
    def __init__(self, in_dim=28**2, out_dim=25, mid_dim=500, bs=64, device=torch.device("cuda:0")):
        super(Metric, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.bs = bs
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.measure = torch.randn(self.bs, self.out_dim, self.in_dim, device=self.device)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim)
        )

    def forward(self, input):
        #return torch.bmm(self.measure, input.unsqueeze(dim=-1))
        return self.net(input)


def load_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    return train_loader


def train_dcs(latent_dim=100, batch_size=64, num_training_epoch=100000, lr=1e-4, initial_step_size=0.01, num_grad_iters=4, device=torch.device("cuda:0")):
    file_exporter = FileExporter('./image', )
    training_data = load_data(batch_size)

    gen = Generator().to(device)
    measurement = Metric().to(device)
    step_size = torch.ones(1).to(device) * np.log(initial_step_size)
    optimizer = torch.optim.Adam(list(gen.parameters()) + list(measurement.parameters()) + list( step_size), lr=lr)
    #optimizer_m = torch.optim.Adam(measurement.parameters(), lr=lr)
    MSELoss = nn.MSELoss()
    #optimizer_s = torch.optim.Adam([step_size], lr=lr)
    step_size.requires_grad_()
    pbar = tqdm(range(num_training_epoch))
    n_batch = 0
    z_0 = torch.randn(batch_size, latent_dim, device=device, requires_grad=False)
    for epoch in pbar:
        for i, (images, labels) in enumerate(training_data):
            if (images.shape[0] == 32):
                continue
            original_data = images.reshape(batch_size, -1).to(device)
            #$assert 0
            original_data = (original_data) * 2 - 1
            z_initial = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
            z_initial = z_initial / z_initial.norm(keepdim=True, dim=-1)
            measurement_original_data = measurement(original_data)
            #z = [z_initial for _ in range(num_grad_iters)]
            #z_initial.requires_grad_()
            z = torch.clone(z_initial)
            for itr in range(1, num_grad_iters):
                #z_[itr-1].requires_grad_()
                #t = measurement(gen(z_[itr-1]))
                t = measurement(gen(z))
                l = (t - measurement_original_data).square().sum(dim=-1)
                #l.backward()
                #z__.append(grad(l, z_[itr-1]))
                #print(l.shape)
                g = grad(outputs=l, inputs=z,grad_outputs=torch.ones(64, device=device))[0]
                #print(g)
                #print(g)
                #print(g.shape, z.shape)
                z = z - step_size.exp() * g
                z = z / z.norm(dim=-1, keepdim=True)
                #z[itr] = z[itr - 1] - 0.02 * z_[itr-1].grad

                #z[itr] = z[itr - 1] - step_size.exp().detach() * z_[itr-1].grad

                #z[itr] = (z[itr] / z[itr].norm(keepdim=True, dim=-1))
                #z_[itr] = z_[itr-1] - 0.02 * z_[itr-1].grad
                #z_[itr] = z_[itr-1] - step_size.exp().detach() * z_[itr-1].grad
                #z_[itr] = (z_[itr] / z_[itr].norm(keepdim=True, dim=-1)).detach()
            z_optimized = z
            #z_initial.requires_grad_()
            #z_optimized = z_initial - step_size.exp() * z__[0]
            #z_optimized = z_optimized / z_optimized.norm(dim=-1, keepdim=True)
            #z_optimized = z_initial - step_size.exp() * z__[1]
            #z_optimized = z_optimized / z_optimized.norm(dim=-1, keepdim=True)
            #z_optimized = z_initial - step_size.exp() * z__[2]
            #z_optimized = z_optimized / z_optimized.norm(dim=-1, keepdim=True)

            #z_[-1]#z[-1]#z_initial - 0.02 * (z__[0] + z__[1] + z__[2])
            #z_optimized = z_initial - step_size.exp() * (z__[0] + z__[1] + z__[2])
            #z_initial.requires_grad_()
            generated_data_initial = gen(z_initial)
            generated_data_optimized = gen(z_optimized)
            #measurement_original_data = measurement(original_data)
            measurement_generated_data_initial = measurement(generated_data_initial)
            measurement_generated_data_optimized = measurement(generated_data_optimized)
            generated_loss = (measurement_generated_data_optimized- measurement_original_data).square().sum(dim=-1).mean()
            RIP_loss = ((measurement_generated_data_initial - measurement_original_data).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_initial - original_data).norm(dim=-1)).square() + \
                ((measurement_generated_data_optimized - measurement_original_data).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_optimized - original_data).norm(dim=-1)).square() + \
                 ((measurement_generated_data_optimized - measurement_generated_data_initial).reshape(-1, 25).norm(dim=-1)- \
                                (generated_data_optimized - generated_data_initial).norm(dim=-1)).square()#.mean()
            loss = generated_loss + RIP_loss.mean() / 3
            if  i % 50 ==  0:
                RECON_LOSS = (generated_data_optimized-original_data).norm(dim=-1).mean()
                desc = f"nbatch: {n_batch} | RIP_LOSS: {(RIP_loss.mean() / 3):.2f} | GEN_LOSS: {generated_loss:.2f} | RECON_LOSS: {RECON_LOSS} | step_size: {step_size.exp().item()}"
                pbar.set_description(desc)

                #print(loss.item())
                #print((generated_data_optimized[0] - generated_data_optimized[1] ) / 2 * 255)
                #file_exporter.save((original_data.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 1) / 2, 'origin2')
                file_exporter.save((generated_data_optimized.detach().reshape(batch_size, 28, 28, 1).cpu().numpy() + 1) / 2, f'reconstruction{sys.argv[1]}')

            optimizer.zero_grad()
            #optimizer_m.zero_grad()
            #optimizer_s.zero_grad()
            loss.backward()
            optimizer.step()
            #optimizer_s.step()
            #optimizer_m.step()

            #optimizer_m.zero_grad()
            #(RIP_loss.mean() / 3).backward()
            #optimizer_m.step()

            n_batch += 1

if __name__ == "__main__":
    train_dcs()
