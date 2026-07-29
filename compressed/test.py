import torch
import torch.nn as nn
import numpy as np
import time
import os
import argparse

from neupan import neupan, configuration as neupan_config
from neupan.configuration import np_to_tensor, tensor_dtype
from compressed.nrmp_net import NRMPCompressed


def _prepare_input(planner, state, obs_points):
    goal = np.random.uniform(-15, 15, (3, 1)).astype(np.float32)
    goal[2, 0] = np.random.uniform(-np.pi, np.pi)
    planner.update_initial_path_from_goal(state, goal)
    planner.reset()
    cur_vel = np.zeros((2, planner.T))
    nom_t = [np_to_tensor(n) for n in planner.ipath.generate_nom_ref_state(state, cur_vel, planner.ref_speed)]
    obs_t = np_to_tensor(obs_points)
    pf, R, op = planner.pan.generate_point_flow(nom_t[0], obs_t, None)
    mu, lam, sp = planner.pan.dune_layer(pf, R, op)

    max_num = planner.pan.nrmp_layer.max_num
    mu = [m[:, :min(m.shape[1], max_num)] for m in mu]
    lam = [l[:, :min(l.shape[1], max_num)] for l in lam]
    sp = [s[:, :min(s.shape[1], max_num)] for s in sp]

    return nom_t[0], nom_t[1], nom_t[2], nom_t[3], mu, lam, sp


@torch.no_grad()
def compare_outputs(planner_yaml, model_path, num_samples=200, device='cpu'):
    print(f"\n{'='*60}")
    print("Test 1: Output Similarity Comparison")
    print(f"{'='*60}")

    neupan_config.device = torch.device(device)
    planner = neupan.init_from_yaml(planner_yaml)
    nrmp = planner.pan.nrmp_layer

    nrmp_comp = NRMPCompressed(nrmp)
    nrmp_comp.net.load_state_dict(torch.load(model_path, map_location=device))
    nrmp_comp = nrmp_comp.to(device)
    nrmp_comp.eval()

    from compressed.dataset import DataGenerator
    gen = DataGenerator()
    gen.planner = planner
    gen.nrmp = nrmp
    gen.dune = planner.pan.dune_layer
    gen.robot = planner.robot
    gen.T = planner.T
    gen.ref_speed = planner.ref_speed

    T = nrmp.T
    state_errors, vel_errors, dist_errors = [], [], []
    state_cos, vel_cos = [], []
    t_nrmp_list, t_comp_list = [], []

    for i in range(num_samples):
        state, obs_points = gen.generate_scenario_with_blocks()
        nom_s, nom_u, ref_s, ref_us, mu, lam, sp = _prepare_input(planner, state, obs_points)

        t0 = time.perf_counter()
        gt_state, gt_vel, gt_dist = nrmp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)
        t_nrmp_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred_state, pred_vel, pred_dist = nrmp_comp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)
        t_comp_list.append(time.perf_counter() - t0)

        gn_s = torch.norm(gt_state); gn_v = torch.norm(gt_vel)
        state_errors.append(torch.norm(pred_state - gt_state).item() / gn_s.item() if gn_s > 0 else 0)
        vel_errors.append(torch.norm(pred_vel - gt_vel).item() / gn_v.item() if gn_v > 0 else 0)
        if gt_dist is not None and pred_dist is not None:
            gn_d = torch.norm(gt_dist)
            dist_errors.append(torch.norm(pred_dist - gt_dist).item() / (gn_d.item() + 1e-8))

        state_cos.append(nn.functional.cosine_similarity(pred_state.flatten().unsqueeze(0), gt_state.flatten().unsqueeze(0)).item())
        vel_cos.append(nn.functional.cosine_similarity(pred_vel.flatten().unsqueeze(0), gt_vel.flatten().unsqueeze(0)).item())

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    print(f"\nResults ({num_samples} samples):")
    print(f"  State rel error:   mean={np.mean(state_errors):.4f}, median={np.median(state_errors):.4f}")
    print(f"  Vel rel error:     mean={np.mean(vel_errors):.4f}, median={np.median(vel_errors):.4f}")
    if dist_errors:
        print(f"  Distance rel err:  mean={np.mean(dist_errors):.4f}")
    print(f"  State cosine sim:  mean={np.mean(state_cos):.4f}, min={np.min(state_cos):.4f}")
    print(f"  Vel cosine sim:    mean={np.mean(vel_cos):.4f}, min={np.min(vel_cos):.4f}")
    print(f"  Avg NRMP time:     {np.mean(t_nrmp_list)*1000:.4f} ms")
    print(f"  Avg Comp time:     {np.mean(t_comp_list)*1000:.4f} ms")


@torch.no_grad()
def benchmark_inference(planner_yaml, model_path, num_runs=500, device='cpu'):
    print(f"\n{'='*60}")
    print("Test 2: Inference Time Benchmark")
    print(f"{'='*60}")

    neupan_config.device = torch.device(device)
    planner = neupan.init_from_yaml(planner_yaml)
    nrmp = planner.pan.nrmp_layer

    nrmp_comp = NRMPCompressed(nrmp)
    nrmp_comp.net.load_state_dict(torch.load(model_path, map_location=device))
    nrmp_comp = nrmp_comp.to(device)
    nrmp_comp.eval()

    from compressed.dataset import DataGenerator
    gen = DataGenerator()
    gen.planner = planner; gen.nrmp = nrmp
    gen.dune = planner.pan.dune_layer; gen.robot = planner.robot
    gen.T = planner.T; gen.ref_speed = planner.ref_speed

    state, obs = gen.generate_scenario_with_blocks()
    nom_s, nom_u, ref_s, ref_us, mu, lam, sp = _prepare_input(planner, state, obs)

    for _ in range(10):
        nrmp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)
        nrmp_comp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)

    t_nrmp, t_comp = [], []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        nrmp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)
        t_nrmp.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        nrmp_comp(nom_s, nom_u, ref_s, ref_us, mu, lam, sp)
        t_comp.append(time.perf_counter() - t0)

    mn = np.mean(t_nrmp) * 1000; sn = np.std(t_nrmp) * 1000
    mc = np.mean(t_comp) * 1000; sc = np.std(t_comp) * 1000
    print(f"NRMP (CVXPY):        {mn:.4f} ms +/- {sn:.4f} ms")
    print(f"NRMPCompressed (NN): {mc:.4f} ms +/- {sc:.4f} ms")
    print(f"Speedup:             {mn / mc:.2f}x")


@torch.no_grad()
def test_navigation(planner_yaml, model_path, env_yaml=None, max_steps=500, device='cpu'):
    print(f"\n{'='*60}")
    print("Test 3: Navigation Success Rate")
    print(f"{'='*60}")

    neupan_config.device = torch.device(device)
    planner_orig = neupan.init_from_yaml(planner_yaml)
    planner_comp = neupan.init_from_yaml(planner_yaml)

    nrmp_comp = NRMPCompressed(planner_orig.pan.nrmp_layer)
    nrmp_comp.net.load_state_dict(torch.load(model_path, map_location=device))
    nrmp_comp = nrmp_comp.to(device)
    nrmp_comp.eval()
    planner_comp.pan.nrmp_layer = nrmp_comp

    if env_yaml and os.path.exists(env_yaml):
        try:
            import irsim
            _run_nav_sim(planner_orig, planner_comp, env_yaml, max_steps)
            return
        except ImportError:
            pass
    _run_nav_standalone(planner_orig, planner_comp, device)


def _run_nav_standalone(planner_orig, planner_comp, device, num_scenarios=50):
    from compressed.dataset import DataGenerator
    gen = DataGenerator()
    gen.planner = planner_orig; gen.nrmp = planner_orig.pan.nrmp_layer
    gen.dune = planner_orig.pan.dune_layer; gen.robot = planner_orig.robot
    gen.T = planner_orig.T; gen.ref_speed = planner_orig.ref_speed

    succ_orig, succ_comp = 0, 0
    for i in range(num_scenarios):
        state, obs = gen.generate_scenario_with_blocks()

        def run(planner):
            planner.reset()
            s = state.copy()
            for _ in range(50):
                if planner.info.get("arrive", False) or planner.info.get("stop", False):
                    break
                action, info = planner(s, obs)
                s[0, 0] += action[0, 0] * 0.1 * np.cos(s[2, 0])
                s[1, 0] += action[0, 0] * 0.1 * np.sin(s[2, 0])
                s[2, 0] += action[1, 0] * 0.1
            return not planner.info.get("stop", False)

        if run(planner_orig): succ_orig += 1
        if run(planner_comp): succ_comp += 1

    print(f"Original NRMP:  {succ_orig}/{num_scenarios} ({100*succ_orig/num_scenarios:.1f}%)")
    print(f"Compressed NRMP: {succ_comp}/{num_scenarios} ({100*succ_comp/num_scenarios:.1f}%)")


def _run_nav_sim(planner_orig, planner_comp, env_yaml, max_steps):
    import irsim
    for name, planner in [("Original", planner_orig), ("Compressed", planner_comp)]:
        env = irsim.make(env_yaml, save_ani=False, display=False, full=False)
        planner.reset()
        arrived = collision = False
        for i in range(max_steps):
            rs = env.get_robot_state()
            ls = env.get_lidar_scan()
            pts = planner.scan_to_point(rs, ls)
            action, info = planner(rs, pts)
            if info["arrive"]: arrived = True; break
            if info["stop"] or info["collision"]: collision = True; break
            env.step(action)
        env.end(0)
        status = "ARRIVED" if arrived else ("COLLISION" if collision else "MAX_STEPS")
        print(f"  {name}: {status} at step {i+1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner_yaml', default='example/corridor/diff/planner.yaml')
    parser.add_argument('--model_path', default='compressed/models/best_model.pth')
    parser.add_argument('--env_yaml', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--benchmark_runs', type=int, default=500)
    parser.add_argument('--nav_scenarios', type=int, default=50)
    args = parser.parse_args()

    compare_outputs(args.planner_yaml, args.model_path, args.num_samples, args.device)
    benchmark_inference(args.planner_yaml, args.model_path, args.benchmark_runs, args.device)
    test_navigation(args.planner_yaml, args.model_path, args.env_yaml, device=args.device)


if __name__ == "__main__":
    main()
