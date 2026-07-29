from __future__ import annotations

import argparse
import copy
import math
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


def dump_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    with open(file_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(data, file_handle, sort_keys=False, allow_unicode=True)


def _world_bounds(env_config: dict[str, Any]) -> tuple[float, float, float, float]:
    world = env_config.get("world", {})
    offset = world.get("offset", [0.0, 0.0])
    width = float(world.get("width", 0.0))
    height = float(world.get("height", 0.0))
    x_min = float(offset[0])
    y_min = float(offset[1])
    return x_min, y_min, x_min + width, y_min + height


def _point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _shape_margin(shape: Any) -> float:
    if isinstance(shape, list) and shape:
        shape = shape[0]

    if not isinstance(shape, dict):
        return 1.0

    shape_name = str(shape.get("name", "")).lower()
    if shape_name == "circle":
        return float(shape.get("radius", 1.0)) + 0.5
    if shape_name == "rectangle":
        length = float(shape.get("length", 1.0))
        width = float(shape.get("width", 1.0))
        return 0.5 * math.hypot(length, width) + 0.5
    if shape_name == "polygon":
        vertices = shape.get("vertices") or []
        if vertices:
            return max(math.hypot(float(x), float(y)) for x, y in vertices) + 0.5
        if shape.get("random_shape"):
            radius_range = shape.get("avg_radius_range", [0.5, 1.0])
            return float(radius_range[-1]) + 0.5
        return 1.5
    return 1.0


def _sample_state(
    rng: random.Random,
    bounds: tuple[float, float, float, float],
    margin: float,
    forbidden_points: list[tuple[float, float, float]],
    existing_points: list[tuple[tuple[float, float], float]],
    state_template: list[float],
) -> list[float]:
    x_min, y_min, x_max, y_max = bounds
    x_low = x_min + margin
    x_high = x_max - margin
    y_low = y_min + margin
    y_high = y_max - margin

    if x_low >= x_high:
        x_low, x_high = x_min, x_max
    if y_low >= y_high:
        y_low, y_high = y_min, y_max

    # --- 核心修改：计算正态分布的均值(地图中心)和标准差 ---
    # 均值设定为可用区域的中心
    x_mu = (x_low + x_high) / 2.0
    y_mu = (y_low + y_high) / 2.0
    
    # 标准差控制密集程度：根据正态分布的 3-sigma 原则，
    # 将标准差设为单边距离的 1/3 到 1/4，可以让大约 95%~99% 的点落在边界内。
    # 这里取 1/3.5 作为一个折中值，既保证向中心靠拢，又避免高频触发边界截断。
    x_sigma = max((x_high - x_low) / 3.5, 0.1)
    y_sigma = max((y_high - y_low) / 3.5, 0.1)

    new_state = copy.deepcopy(state_template)

    for _ in range(500):
        # 使用高斯（正态）分布进行采样
        x_pos = rng.gauss(x_mu, x_sigma)
        y_pos = rng.gauss(y_mu, y_sigma)

        # 截断超出边界的采样值，确保安全
        if not (x_low <= x_pos <= x_high) or not (y_low <= y_pos <= y_high):
            continue
            
        candidate = (x_pos, y_pos)

        if any(_point_distance(candidate, point) < clearance for point, clearance in existing_points):
            continue
        if any(_point_distance(candidate, (point_x, point_y)) < clearance for point_x, point_y, clearance in forbidden_points):
            continue

        new_state[0] = x_pos
        new_state[1] = y_pos
        return new_state

    # 兜底强制赋值：如果正态分布多次碰撞失败，退回到中心点附近均匀采样
    new_state[0] = rng.uniform(x_low, x_high)
    new_state[1] = rng.uniform(y_low, y_high)
    return new_state


def shuffle_env_config(env_config: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    shuffled_config = copy.deepcopy(env_config)
    rng = random.Random(seed)

    bounds = _world_bounds(shuffled_config)
    occupied_points: list[tuple[tuple[float, float], float]] = []

    for robot in shuffled_config.get("robot", []) or []:
        state = robot.get("state", [])
        goal = robot.get("goal", [])
        if isinstance(state, list) and len(state) >= 2:
            occupied_points.append(((float(state[0]), float(state[1])), 3.0))
        if isinstance(goal, list) and len(goal) >= 2:
            occupied_points.append(((float(goal[0]), float(goal[1])), 3.0))

    for obstacle_group in shuffled_config.get("obstacle", []) or shuffled_config.get("obstacles", []) or []:
        distribution = obstacle_group.get("distribution", {}) or {}
        if str(distribution.get("name", "")).lower() != "manual":
            continue

        shapes = obstacle_group.get("shape", []) or []
        states = obstacle_group.get("state", []) or []
        obstacle_count = int(obstacle_group.get("number", len(states) or len(shapes) or 0))

        for index in range(obstacle_count):
            if index >= len(states):
                continue
            
            shape = shapes[index] if index < len(shapes) else (shapes[-1] if shapes else {})
            margin = _shape_margin(shape)
            state_template = states[index]
            
            if not isinstance(state_template, list) or len(state_template) < 2:
                continue

            updated_state = _sample_state(
                rng,
                bounds,
                margin,
                [],
                occupied_points,
                state_template,
            )
            states[index] = updated_state
            occupied_points.append(((updated_state[0], updated_state[1]), margin))

    return shuffled_config


def shuffle_env_file(env_file: str | Path, seed: int | None = None, output_path: str | Path | None = None) -> Path:
    env_path = Path(env_file)
    shuffled_config = shuffle_env_config(load_yaml(env_path), seed=seed)

    if output_path is None:
        with tempfile.NamedTemporaryFile(prefix=f"{env_path.stem}_{seed}_", suffix=".yaml", delete=False) as temp_file:
            output_file = Path(temp_file.name)
    else:
        output_file = Path(output_path)

    dump_yaml(shuffled_config, output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle a NeuPAN IR-SIM env configuration")
    parser.add_argument("env_file", type=str, help="Path to the base env.yaml file")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed used for shuffling")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional output path for the shuffled env file")
    args = parser.parse_args()

    output_file = shuffle_env_file(args.env_file, seed=args.seed, output_path=args.output)
    print(output_file)


if __name__ == "__main__":
    main()