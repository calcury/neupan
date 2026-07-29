import argparse
import os
import sys
import warnings
import time
from pathlib import Path

# add project root to path (parent of script/)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import numpy as np
import torch
from tqdm import tqdm
from neupan import neupan, configuration as neupan_config
from compressed.nrmp_net import NRMPCompressed
import irsim

from shuffle import shuffle_env_file


class SuppressStdout:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._devnull = None

    def __enter__(self):
        if self.enabled:
            self._original_stdout = sys.stdout
            self._devnull = open(os.devnull, 'w')
            sys.stdout = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            sys.stdout = self._original_stdout


def run_episode(env_file, planner, max_steps=1000, render=False):
    with SuppressStdout(enabled=True):
        env = irsim.make(env_file, save_ani=False, full=False, display=render)

    success = False
    steps = 0
    velocities = []
    inference_times = []

    with SuppressStdout(enabled=True):
        for i in range(max_steps):
            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()

            if isinstance(robot_state, np.ndarray):
                vel = np.linalg.norm(robot_state[3:5]) if robot_state.shape[0] > 3 else 0.0
            else:
                vel = getattr(robot_state, 'v', 0.0)
            velocities.append(vel)

            points = planner.scan_to_point(robot_state, lidar_scan)

            t0 = time.perf_counter()
            action, info = planner(robot_state, points, None)
            inference_times.append(time.perf_counter() - t0)

            if info.get("arrive", False):
                success = True
                steps = i + 1
                break

            if render:
                env.render()

            if env.done():
                steps = i + 1
                break

            env.step(action)

    nav_time = steps * 0.1 if success else 0.0
    avg_speed = np.mean(velocities) if velocities else 0.0
    avg_inference = np.mean(inference_times) * 1000 if inference_times else 0.0

    return success, nav_time, avg_speed, avg_inference


def run_display(env_file, planner_file, model_path, seed, max_steps):
    if not os.path.isabs(env_file):
        env_file = str(_project_root / env_file)
    if not os.path.isabs(planner_file):
        planner_file = str(_project_root / planner_file)
    if not os.path.isabs(model_path):
        model_path = str(_project_root / model_path)

    # use shuffle so obstacles vary, but override start/goal to stay in safe zone
    temp_env = shuffle_env_file(env_file, seed=seed)
    start_pt = np.array([-1.0, 25.0, 0.0]).reshape(3, 1)
    goal_pt = np.array([50.0, 25.0, 0.0]).reshape(3, 1)

    for name, is_comp in [("--- Original NRMP ---", False), ("--- Compressed NRMP ---", True)]:
        print(f"\n{name}")
        planner = neupan.init_from_yaml(planner_file)
        neupan_config.time_print = False
        planner.set_reference_speed(4.0)
        planner.update_initial_path_from_waypoints([start_pt, goal_pt])
        if is_comp:
            nrmp_comp = NRMPCompressed(planner.pan.nrmp_layer)
            if os.path.exists(model_path):
                nrmp_comp.net.load_state_dict(torch.load(model_path, map_location='cpu'))
            nrmp_comp.eval()
            planner.pan.nrmp_layer = nrmp_comp

        env = irsim.make(str(temp_env), save_ani=False, full=False, display=True)
        planner.reset()
        arrived = collision = False
        for i in range(max_steps):
            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()
            points = planner.scan_to_point(robot_state, lidar_scan)
            action, info = planner(robot_state, points, None)
            if is_comp:
                print(f"  step {i}: vel=[{action[0,0]:.3f}, {action[1,0]:.3f}]  arrive={info.get('arrive',False)} stop={info.get('stop',False)}")
            if info.get("arrive", False): arrived = True; break
            if info.get("stop", False): collision = True; break
            env.step(action)
            env.render()
        env.end(0)
        status = "ARRIVED" if arrived else ("STOP/COLLISION" if collision else "MAX_STEPS")
        print(f"  Result: {status} at step {i+1}")

    if os.path.exists(temp_env):
        os.remove(temp_env)


def main():
    parser = argparse.ArgumentParser(description="Compressed NRMP vs Original NRMP 回测对比")
    parser.add_argument("-e", "--example", type=str, default="convex_obs", choices=["convex_obs", "non_obs", "dyna_obs", "corridor"])
    parser.add_argument("-d", "--kinematics", type=str, default="diff", choices=["diff", "acker", "omni"])
    parser.add_argument("--env_yaml", type=str, default=None)
    parser.add_argument("--planner_yaml", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="compressed/models/best_model.pth")
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("-m", "--max_steps", type=int, default=1000)
    parser.add_argument("--display", action="store_true", help="可视化单次运行，对比原始和压缩网络的输出")
    parser.add_argument("--render", action="store_true", help="批量测试时每轮显示可视化（每轮会打开一个窗口）")
    parser.add_argument("--seed", type=int, default=42, help="--display 时使用的随机种子")
    args = parser.parse_args()

    neupan_config.time_print = False

    script_dir = Path(__file__).resolve().parent

    if args.env_yaml and args.planner_yaml:
        env_file = args.env_yaml
        planner_file = args.planner_yaml
    else:
        env_file = str(_project_root / f"example/{args.example}/{args.kinematics}/env.yaml")
        planner_file = str(_project_root / f"example/{args.example}/{args.kinematics}/planner.yaml")
        if not os.path.exists(env_file):
            print(f"Error: env file not found at {env_file}")
            print("Use --env_yaml and --planner_yaml to specify custom paths.")
            sys.exit(1)

    model_path = str((_project_root / args.model_path).resolve())

    print("=" * 60)
    print("Compressed NRMP vs Original NRMP 回测对比")
    print(f"Env:     {env_file}")
    print(f"Planner: {planner_file}")
    print(f"Model:   {model_path}")
    print(f"Trials:  {args.num_samples}")
    print("=" * 60)

    if args.display:
        return run_display(env_file, planner_file, model_path, args.seed, args.max_steps)

    neupan_config.time_print = False
    results = {"orig": {"success": 0, "nav_times": [], "speeds": [], "infer_times": []},
               "comp": {"success": 0, "nav_times": [], "speeds": [], "infer_times": []}}

    for seed in tqdm(range(args.num_samples), desc="Benchmark", unit="ep"):
        temp_env = shuffle_env_file(env_file, seed=seed)

        for key, is_comp in [("orig", False), ("comp", True)]:
            planner = neupan.init_from_yaml(planner_file)
            if is_comp:
                nrmp_comp = NRMPCompressed(planner.pan.nrmp_layer)
                if os.path.exists(model_path):
                    nrmp_comp.net.load_state_dict(torch.load(model_path, map_location='cpu'))
                nrmp_comp.eval()
                planner.pan.nrmp_layer = nrmp_comp

            success, nav_time, avg_speed, avg_infer = run_episode(str(temp_env), planner, args.max_steps, args.render)
            r = results[key]
            if success:
                r["success"] += 1
                r["nav_times"].append(nav_time)
            r["speeds"].append(avg_speed)
            r["infer_times"].append(avg_infer)

        if os.path.exists(temp_env):
            os.remove(temp_env)

    print("\n" + "#" * 60)
    print("  对比报告")
    print("#" * 60)

    for label, key in [("Original NRMP", "orig"), ("Compressed NRMP", "comp")]:
        r = results[key]
        sr = r["success"] / args.num_samples * 100
        ant = np.mean(r["nav_times"]) if r["nav_times"] else float('inf')
        ast = np.mean(r["speeds"]) if r["speeds"] else 0.0
        ait = np.mean(r["infer_times"]) if r["infer_times"] else 0.0
        print(f"\n  [{label}]")
        print(f"    Success Rate:       {sr:6.2f}% ({r['success']}/{args.num_samples})")
        print(f"    Avg Nav Time:       {ant:6.2f} s")
        print(f"    Avg Speed:          {ast:6.2f} m/s")
        print(f"    Avg Inference:      {ait:6.3f} ms")

    r_orig = results["orig"]
    r_comp = results["comp"]
    sr_orig = r_orig["success"] / args.num_samples * 100
    sr_comp = r_comp["success"] / args.num_samples * 100
    ant_orig = np.mean(r_orig["nav_times"]) if r_orig["nav_times"] else float('inf')
    ant_comp = np.mean(r_comp["nav_times"]) if r_comp["nav_times"] else float('inf')
    ait_orig = np.mean(r_orig["infer_times"]) if r_orig["infer_times"] else 0.0
    ait_comp = np.mean(r_comp["infer_times"]) if r_comp["infer_times"] else 0.0

    print(f"\n  {'='*40}")
    print(f"  Delta Summary")
    print(f"  {'='*40}")
    print(f"    Success Rate Delta:  {sr_comp - sr_orig:+.2f}%")
    print(f"    Nav Time Delta:      {ant_comp - ant_orig:+.2f} s")
    print(f"    Inference Speedup:   {ait_orig / ait_comp:.2f}x" if ait_comp > 0 else "    Inference Speedup:   N/A")
    print(f"  {'='*40}\n")


if __name__ == "__main__":
    main()
