"""Utility to load application configuration."""

import configparser
from pathlib import Path


def load_config(path: str | None = None) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    if path:
        cfg_path = Path(path)
    else:
        # Look for config.cfg in the data directory relative to the project root
        project_root = Path(__file__).parent.parent
        cfg_path = project_root / "data" / "config.cfg"
    
    parser.read(cfg_path)
    return parser


# Load default configuration at import time
CONFIG = load_config()
