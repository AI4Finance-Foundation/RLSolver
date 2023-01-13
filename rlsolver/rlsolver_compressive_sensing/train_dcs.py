import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self, in_dim=500, out_dim=28**2, mid_dim = 512):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim),nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),nn.ReLU(),
            nn.Linear(self.mid_dim, self.out_dim),nn.Tanh()
        )


    def forward(self, input):
        return self.net(input)

class Metric(nn.Module):
    def __init__(self, in_dim=28**2, out_dim=25, mid_dim=500, bs=32, device=torch.device("cuda:0")):
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
        return torch.bmm(self.measure, input.unsqueeze(dim=-1))
        #return self.net(input)


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


def train_dcs(latent_dim=500, batch_size=32, num_training_epoch=100, lr=5e-5, initial_step_size=0.01, num_grad_iters=3, device=torch.device("cuda:0")):
    training_data = load_data(batch_size)
    gen = Generator().to(device)
    measurement = Metric().to(device)
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    MSELoss = nn.MSELoss()
    step_size = nn.Parameter(torch.as_tensor([initial_step_size])).to(device)
    for epoch in range(num_training_epoch):
        for i, (images, labels) in enumerate(training_data):
            images = images.reshape(batch_size, -1)
            original_data = images.to(device)
            z_initial = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
            measurement_original_data = measurement(original_data)
            z = [z_initial for _ in range(num_grad_iters)]
            for itr in range(1, num_grad_iters):
                z[itr - 1].requires_grad_()
                t = measurement(gen(z[itr - 1]))
                MSELoss(t, measurement_original_data).backward()
                z[itr] = z[itr - 1] - step_size * z[itr - 1].grad
                z[itr] = (z[itr] / z[itr].norm(keepdim=True)).detach()
            z_optimized = z[-1]
            generated_data_initial = gen(z_optimized)
            generated_data_optimized = gen(z_optimized)

            measurement_generated_data_initial = measurement(generated_data_initial)
            measurement_generated_data_optimized = measurement(generated_data_optimized)
            generated_loss = MSELoss(measurement_generated_data_optimized, measurement_original_data)
            RIP_loss = MSELoss((measurement_generated_data_initial - measurement_original_data).norm(), \
                                (generated_data_initial - original_data).norm())
            RIP_loss += MSELoss((measurement_generated_data_optimized - measurement_original_data).norm(), \
                                (generated_data_optimized - original_data).norm())
            RIP_loss += MSELoss((measurement_generated_data_optimized - measurement_generated_data_initial).norm(), \
                                (generated_data_optimized - generated_data_initial).norm())
            loss = generated_loss + RIP_loss / 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    print(train_dcs)
    train_dcs()
