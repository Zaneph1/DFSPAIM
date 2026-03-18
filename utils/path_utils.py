# coding: utf-8
"""
path utility module
provides unified path management for cross-platform open source projects
"""

import os
import json
from pathlib import Path


# project root (two levels up from current file)
BASE_DIR = Path(__file__).resolve().parent.parent

# config file path
CONFIG_PATH = BASE_DIR / "config.json"


def load_config():
    """
    load config file

    Returns:
        dict: config dictionary
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config file not found:{CONFIG_PATH}\n"
            "please copy config.example.json to config.json and update path configuration"
        )

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_path(key, as_str=False):
    """
    get absolute path by config key

    Args:
        key (str): config key name, e.g. 'root_folder', 'resume_path' etc.
        as_str (bool): whether to return string format, default returns Path object

    Returns:
        Path | str: absolute path
    """
    config = load_config()

    if key not in config:
        raise KeyError(f"config item '{key}' not found, please add to config.json")

    # 将配置中的相对路径转换为absolute path
    path = BASE_DIR / config[key]

    return str(path) if as_str else path


def ensure_dir(path):
    """
    ensure directory exists, create if not

    Args:
        path (Path | str): directory path
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


# common path shortcuts
def get_root_folder():
    return get_path("root_folder", as_str=True)


def get_resume_path():
    return get_path("resume_path", as_str=True)


def get_single_img_path():
    return get_path("single_img_path", as_str=True)


def get_save_path():
    return get_path("save_path", as_str=True)
