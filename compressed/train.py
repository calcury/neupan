import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import numpy as np
import os
import argparse
import time

from neupan import neupan, configuration as neupan_config
from neupan.configuration import tensor_dtype
from compressed.nrmp_net import NRMPNet
from compressed.dataset import DataGenerator, AugmentedNRMPDataset


def train(args):
    device = torch.device(args.device)
    neupan_config.device = device

    print(f"Initializing planner from {args.planner_yaml}")
    planner = neupan.init_from_yaml(args.planner_yaml)
    nrmp = planner.pan.nrmp_layer

    neupan_config.time_print = False
    T = nrmp.T
    max_num = nrmp.max_num
    edge_dim = nrmp.G.shape[0]
    state_dim = 3
    control_dim = 2

    print(f"T={T}, max_num={max_num}, edge_dim={edge_dim}")

    generator = DataGenerator(args.planner_yaml, device=args.device)
    print(f"Generating {args.num_samples} training samples...")
    dataset = generator.generate_dataset(args.num_samples, save_path=args.cache_path)

    aug_dataset = AugmentedNRMPDataset(
        dataset,
        noise_scale=args.noise_scale,
        dropout_prob=args.dropout_prob,
        T=T, max_num=max_num, edge_dim=edge_dim, state_dim=state_dim, control_dim=control_dim, point_dim=2,
    )

    train_size = int(len(aug_dataset) * 0.9)
    val_size = len(aug_dataset) - train_size
    train_dataset, val_dataset = random_split(aug_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = NRMPNet(
        T=T, max_num=max_num, edge_dim=edge_dim,
        state_dim=state_dim, control_dim=control_dim, point_dim=2,
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_freq, gamma=args.lr_decay)
    loss_fn = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Starting training for {args.epochs} epochs")
    best_val_loss = float('inf')

    for epoch in range(args.epochs + 1):
        model.train()
        train_loss = 0.0
        for inp, out in train_loader:
            inp, out = inp.to(device), out.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, out in val_loader:
                inp, out = inp.to(device), out.to(device)
                pred = model(inp)
                loss = loss_fn(pred, out)
                val_loss += loss.item() * inp.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        if epoch % args.print_freq == 0:
            state_mae = 0.0
            vel_mae = 0.0
            with torch.no_grad():
                for inp, out in val_loader:
                    inp, out = inp.to(device), out.to(device)
                    pred = model(inp)
                    sd = state_dim * (T + 1)
                    cd = control_dim * T
                    state_mae += nn.L1Loss()(pred[:, :sd], out[:, :sd]).item() * inp.size(0)
                    vel_mae += nn.L1Loss()(pred[:, sd:sd+cd], out[:, sd:sd+cd]).item() * inp.size(0)
            state_mae /= len(val_loader.dataset)
            vel_mae /= len(val_loader.dataset)
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.6e} | val_loss: {val_loss:.6e} | state_mae: {state_mae:.6e} | vel_mae: {vel_mae:.6e} | lr: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    print(f"Training complete. Best val loss: {best_val_loss:.6e}. Models saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner_yaml', type=str, default='example/corridor/diff/planner.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--cache_path', type=str, default='compressed/dataset_cache.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--decay_freq', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--noise_scale', type=float, default=0.005)
    parser.add_argument('--dropout_prob', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='compressed/models')
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=50)
    args = parser.parse_args()

    train(args)
