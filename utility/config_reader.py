"""Utility to load application configuration."""

import configparser
from pathlib import Path


def load_config(path: str | None = None) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    if path:
        cfg_path = Path(path)
    else:
        # Default to the repository's data/config.cfg when no path is provided
        cfg_path = Path(__file__).resolve().parents[1] / "data" / "config.cfg"
    parser.read(cfg_path)
    return parser


# Load default configuration at import time
CONFIG = load_config()
