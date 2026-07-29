import torch
import torch.nn as nn
from neupan.configuration import tensor_dtype
from typing import Optional, List


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class NRMPNet(nn.Module):
    def __init__(self, T=10, max_num=10, edge_dim=4, state_dim=3, control_dim=2, point_dim=2, hidden_dim=1024, num_blocks=4):
        super().__init__()
        self.T = T
        self.max_num = max_num
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.point_dim = point_dim

        input_dim = self._compute_input_dim()
        output_dim = self._compute_output_dim()

        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        self.net = nn.Sequential(*layers)

    def _compute_input_dim(self):
        s = self.state_dim * (self.T + 1)
        c = self.control_dim * self.T
        mu_dim = (self.T + 1) * self.edge_dim * self.max_num
        lam_dim = (self.T + 1) * self.point_dim * self.max_num
        pt_dim = (self.T + 1) * self.point_dim * self.max_num
        return s + c + s + self.T + mu_dim + lam_dim + pt_dim

    def _compute_output_dim(self):
        return self.state_dim * (self.T + 1) + self.control_dim * self.T + 1 * self.T

    def forward(self, x):
        return self.net(x)

    def flatten_inputs(self, nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list):
        tensors = [
            nom_s.flatten(), nom_u.flatten(), ref_s.flatten(), ref_us.flatten(),
        ]
        for t in range(self.T + 1):
            tensors.append(mu_list[t].flatten() if t < len(mu_list) else torch.zeros(self.edge_dim, self.max_num, dtype=tensor_dtype, device=nom_s.device).flatten())
        for t in range(self.T + 1):
            tensors.append(lam_list[t].flatten() if t < len(lam_list) else torch.zeros(self.point_dim, self.max_num, dtype=tensor_dtype, device=nom_s.device).flatten())
        for t in range(self.T + 1):
            tensors.append(point_list[t].flatten() if t < len(point_list) else torch.zeros(self.point_dim, self.max_num, dtype=tensor_dtype, device=nom_s.device).flatten())
        return torch.cat(tensors)

    def unflatten_outputs(self, x):
        offset = 0
        sd = self.state_dim * (self.T + 1)
        opt_state = x[offset:offset+sd].reshape(self.state_dim, self.T + 1)
        offset += sd
        cd = self.control_dim * self.T
        opt_vel = x[offset:offset+cd].reshape(self.control_dim, self.T)
        offset += cd
        opt_distance = x[offset:offset+self.T].reshape(1, self.T)
        return opt_state, opt_vel, opt_distance


class NRMPCompressed(nn.Module):
    def __init__(self, nrmp_layer: nn.Module, hidden_dim=1024, num_blocks=4):
        super().__init__()
        self.T = nrmp_layer.T
        self.max_num = nrmp_layer.max_num
        self.edge_dim = nrmp_layer.G.shape[0]
        self.no_obs = nrmp_layer.no_obs

        net = NRMPNet(
            T=self.T, max_num=self.max_num, edge_dim=self.edge_dim,
            state_dim=3, control_dim=2, point_dim=2,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
        )
        self.net = net
        self.obstacle_points = None

    @property
    def points(self):
        return self.obstacle_points

    def forward(
        self,
        nom_s: torch.Tensor,
        nom_u: torch.Tensor,
        ref_s: torch.Tensor,
        ref_us: torch.Tensor,
        mu_list: Optional[List[torch.Tensor]] = None,
        lam_list: Optional[List[torch.Tensor]] = None,
        point_list: Optional[List[torch.Tensor]] = None,
    ):
        if point_list:
            self.obstacle_points = point_list[0][:, :self.max_num]

        inp = self.net.flatten_inputs(nom_s, nom_u, ref_s, ref_us, mu_list or [], lam_list or [], point_list or [])
        out = self.net(inp)
        opt_state, opt_vel, opt_distance = self.net.unflatten_outputs(out)

        if self.no_obs:
            opt_distance = None

        return opt_state, opt_vel, opt_distance
