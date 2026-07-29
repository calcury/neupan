import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from neupan import neupan, configuration as neupan_config
from neupan.configuration import np_to_tensor, tensor_dtype
from neupan.blocks import NRMP
from neupan.util import downsample_decimation
from typing import Optional, List
import pickle
import os


class NRMPDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class DataGenerator:
    def __init__(self, planner_yaml: str = None, device=None):
        if planner_yaml is not None:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            neupan_config.device = torch.device(device)
            self.planner = neupan.init_from_yaml(planner_yaml)
            neupan_config.time_print = False
            self.edge_dim = self.planner.pan.nrmp_layer.G.shape[0]
            self.nrmp = self.planner.pan.nrmp_layer
            self.dune = self.planner.pan.dune_layer
            self.robot = self.planner.robot
            self.T = self.planner.T
            self.ref_speed = self.planner.ref_speed

    def generate_random_scenario(self):
        state = np.random.uniform(-15, 15, size=(3, 1)).astype(np.float32)
        state[2, 0] = np.random.uniform(-np.pi, np.pi)
        num_obs = np.random.randint(3, 15)
        obs_points = np.random.uniform(-20, 20, size=(2, num_obs)).astype(np.float32)
        velocities = np.random.uniform(-1, 1, size=(2, num_obs)).astype(np.float32) * 0.1
        return state, obs_points, velocities

    def generate_random_block_obstacle(self):
        cx, cy = np.random.uniform(-15, 15, 2)
        w, h_val = np.random.uniform(2, 8, 2)
        theta = np.random.uniform(0, 2 * np.pi)
        corners = np.array([
            [-w/2, -h_val/2], [w/2, -h_val/2], [w/2, h_val/2], [-w/2, h_val/2]
        ]).T
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        corners = R @ corners + np.array([[cx], [cy]])
        pts_per_edge = np.random.randint(3, 8)
        all_points = []
        n = corners.shape[1]
        for i in range(n):
            p1 = corners[:, i:i+1]
            p2 = corners[:, (i+1) % n]
            for t in np.linspace(0, 1, pts_per_edge, endpoint=False):
                all_points.append(p1 * (1 - t) + p2 * t)
        return np.hstack(all_points).astype(np.float32)

    def generate_scenario_with_blocks(self):
        state = np.random.uniform(-15, 15, size=(3, 1)).astype(np.float32)
        state[2, 0] = np.random.uniform(-np.pi, np.pi)
        num_blocks = np.random.randint(1, 5)
        all_points = []
        for _ in range(num_blocks):
            all_points.append(self.generate_random_block_obstacle())
        obs_points = np.hstack(all_points) if all_points else np.zeros((2, 1), dtype=np.float32)
        if obs_points.shape[1] > 200:
            obs_points = downsample_decimation(obs_points, 200)
        return state, obs_points

    def _trunc_to_max_num(self, tensor_list, max_num):
        result = []
        for t in tensor_list:
            n = min(t.shape[1], max_num)
            truncated = t[:, :n]
            if n < max_num:
                pad = t[:, :1].repeat(1, max_num - n)
                truncated = torch.cat([truncated, pad], dim=1)
            result.append(truncated)
        return result

    def generate_sample(self, state, obs_points):
        goal = np.random.uniform(-15, 15, (3, 1)).astype(np.float32)
        goal[2, 0] = np.random.uniform(-np.pi, np.pi)
        self.planner.update_initial_path_from_goal(state, goal)
        self.planner.reset()
        cur_vel_array = np.zeros((2, self.T))
        nom_input_np = self.planner.ipath.generate_nom_ref_state(state, cur_vel_array, self.ref_speed)
        nom_input_tensor = [np_to_tensor(n) for n in nom_input_np]
        obs_points_tensor = np_to_tensor(obs_points)

        with torch.no_grad():
            point_flow_list, R_list, obs_points_list = self.planner.pan.generate_point_flow(
                nom_input_tensor[0], obs_points_tensor, None
            )
            mu_list, lam_list, sort_point_list = self.dune(point_flow_list, R_list, obs_points_list)
            assert len(mu_list) == self.T + 1

            max_num = self.nrmp.max_num
            mu_list = self._trunc_to_max_num(mu_list, max_num)
            lam_list = self._trunc_to_max_num(lam_list, max_num)
            sort_point_list = self._trunc_to_max_num(sort_point_list, max_num)

            opt_state, opt_vel, opt_distance = self.nrmp(
                nom_input_tensor[0], nom_input_tensor[1],
                nom_input_tensor[2], nom_input_tensor[3],
                mu_list, lam_list, sort_point_list,
            )

        inp = torch.cat(
            [nom_input_tensor[0].flatten(), nom_input_tensor[1].flatten(),
             nom_input_tensor[2].flatten(), nom_input_tensor[3].flatten()]
            + [m.flatten() for m in mu_list]
            + [l.flatten() for l in lam_list]
            + [p.flatten() for p in sort_point_list]
        )

        out_parts = [opt_state.flatten(), opt_vel.flatten()]
        if opt_distance is not None:
            out_parts.append(opt_distance.flatten())
        else:
            out_parts.append(torch.zeros(self.T, dtype=tensor_dtype, device=opt_state.device))
        out = torch.cat(out_parts)

        return inp.cpu(), out.cpu()

    def generate_dataset(self, num_samples=10000, save_path=None, batch_print=500):
        inputs, outputs = [], []
        for i in range(num_samples):
            try:
                state, obs_points = self.generate_scenario_with_blocks()
                inp, out = self.generate_sample(state, obs_points)
                inputs.append(inp)
                outputs.append(out)
                if (i + 1) % batch_print == 0:
                    print(f"Generated {i+1}/{num_samples} samples")
            except Exception as e:
                print(f"Sample {i} failed: {e}")
                continue

        dataset = NRMPDataset(inputs, outputs)
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            torch.save({'inputs': torch.stack(inputs), 'outputs': torch.stack(outputs)}, save_path)
            print(f"Dataset saved to {save_path}")

        return dataset

    def generate_cached_dataset(self, num_samples=10000, cache_path='compressed/dataset_cache.pt'):
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            return NRMPDataset(data['inputs'], data['outputs'])
        return self.generate_dataset(num_samples, cache_path)


def augment_sample(inp, out, noise_scale=0.01, dropout_prob=0.1, T=10, max_num=10, edge_dim=4, state_dim=3, control_dim=2, point_dim=2):
    inp_aug = inp.clone()
    noise = torch.randn_like(inp_aug) * noise_scale
    inp_aug = inp_aug + noise

    mu_start = state_dim * (T + 1) + control_dim * T + state_dim * (T + 1) + T
    mu_dim = (T + 1) * edge_dim * max_num
    lam_start = mu_start + mu_dim
    lam_dim = (T + 1) * point_dim * max_num
    pt_start = lam_start + lam_dim

    if torch.rand(1).item() < dropout_prob:
        num_drop = int(max_num * 0.3)
        for t in range(T + 1):
            s = mu_start + t * edge_dim * max_num
            inp_aug[s + num_drop * edge_dim:s + max_num * edge_dim] = 0
            s2 = lam_start + t * point_dim * max_num
            inp_aug[s2 + num_drop * point_dim:s2 + max_num * point_dim] = 0
            s3 = pt_start + t * point_dim * max_num
            inp_aug[s3 + num_drop * point_dim:s3 + max_num * point_dim] = 0

    return inp_aug, out


class AugmentedNRMPDataset(Dataset):
    def __init__(self, base_dataset, noise_scale=0.005, dropout_prob=0.05, T=10, max_num=10, edge_dim=4, state_dim=3, control_dim=2, point_dim=2):
        self.base = base_dataset
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob
        self.T = T
        self.max_num = max_num
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.point_dim = point_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inp, out = self.base[idx]
        if self.noise_scale > 0 or self.dropout_prob > 0:
            inp, out = augment_sample(
                inp, out, self.noise_scale, self.dropout_prob,
                self.T, self.max_num, self.edge_dim, self.state_dim, self.control_dim, self.point_dim,
            )
        return inp, out
