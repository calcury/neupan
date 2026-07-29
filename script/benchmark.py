import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import numpy as np
import yaml
from tqdm import tqdm
from neupan import neupan
import irsim

# 从 shuffle.py 中导入核心的场景打乱函数
from shuffle import shuffle_env_file

class SuppressStdout:
    """标准输出静默器（修复了 loguru 文件关闭冲突漏洞）"""
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

def run_single_episode(env_file, planner_file, max_steps=1000, display=False):
    silence = not display 

    with SuppressStdout(enabled=silence):
        env = irsim.make(env_file, save_ani=False, full=False, display=display)
        neupan_planner = neupan.init_from_yaml(planner_file)
        
        neupan_planner.set_reference_speed(4.0)
        start_pt = np.array([-1.0, 25.0, 0.0]).reshape(3, 1)
        goal_pt = np.array([50.0, 25.0, 0.0]).reshape(3, 1)
        neupan_planner.update_initial_path_from_waypoints([start_pt, goal_pt])
    
    success = False
    steps = 0
    velocities = []
    dt = 1

    with SuppressStdout(enabled=silence):
        for i in range(max_steps):
            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()

            if isinstance(robot_state, np.ndarray):
                vel = np.linalg.norm(robot_state[3:5]) if robot_state.shape[0] > 3 else 0.0
            else:
                vel = getattr(robot_state, 'v', 0.0) 
            velocities.append(vel)

            points = neupan_planner.scan_to_point(robot_state, lidar_scan)
            action, info = neupan_planner(robot_state, points, None)

            if info.get("arrive", False):
                success = True
                steps = i + 1
                break

            if env.done():
                steps = i + 1
                break

            env.step(action)
            
            if display:
                # 绘制优化轨迹和参考轨迹，便于可视化观察
                env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
                env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
                env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
                env.draw_trajectory(neupan_planner.ref_trajectory, "b", refresh=True)
                env.render()

    nav_time = steps * dt if success else 0.0
    avg_speed = np.mean(velocities) if velocities else 0.0

    return success, nav_time, avg_speed


def main():
    parser = argparse.ArgumentParser(description="NeuPAN Baseline 自动化回测脚本")
    parser.add_argument("-e", "--example", type=str, default="convex_obs", choices=["convex_obs", "non_obs", "dyna_obs"])
    parser.add_argument("-d", "--kinematics", type=str, default="diff", choices=["diff", "acker", "omni"])
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("-m", "--max_steps", type=int, default=1000)
    
    # 控制图形界面的开启
    parser.add_argument("--display", action="store_true", help="开启图形界面可视化（会降低回测速度）")
    
    args = parser.parse_args()

    base_env_file = f"config/{args.example}/{args.kinematics}/env.yaml"
    planner_file = f"config/{args.example}/{args.kinematics}/planner.yaml"

    print("=" * 50)
    print(f"开始进行 NeuPAN Baseline 测试")
    print(f"场景 (Scenario): {args.example} | 运动学 (Kinematics): {args.kinematics}")
    print(f"显示图形界面 (Display): {args.display}")
    print("=" * 50)

    success_count = 0
    total_nav_time = 0.0
    all_speeds = []

    # 如果开启了 display，tqdm 进度条可能会和 irsim 自身的刷新有些许冲突，但仍能正常工作
    for seed in tqdm(range(args.num_samples), desc="回测进度", unit="ep"):
        temp_env_path = f"temp_random_env_{seed}.yaml"
        
        # 该接口会自动保护机器人起点/终点，并保留障碍物的高维物理参数
        shuffle_env_file(base_env_file, seed=seed, output_path=temp_env_path)

        success, nav_time, avg_speed = run_single_episode(temp_env_path, planner_file, args.max_steps, display=args.display)
        
        if success:
            success_count += 1
            total_nav_time += nav_time
        all_speeds.append(avg_speed)
        
        # 测试完后及时清理临时文件
        if os.path.exists(temp_env_path):
            os.remove(temp_env_path)

    success_rate = (success_count / args.num_samples) * 100
    avg_nav_time = (total_nav_time / success_count) if success_count > 0 else float('inf')
    avg_speed_total = np.mean(all_speeds) if all_speeds else 0.0

    print("\n" + "#" * 50)
    print(" 回测测试结果报告 (Benchmark Report)")
    print("#" * 50)
    print(f" * Success Rate (成功率):          {success_rate:.2f} %")
    print(f" * Average Navigation Time (平均导航时间): {avg_nav_time:.2f} s")
    print(f" * Average Speed (平均速度):              {avg_speed_total:.2f} m/s")
    print("#" * 50 + "\n")

if __name__ == "__main__":
    main()