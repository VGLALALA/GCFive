"""Utility to load application configuration."""
import configparser
from pathlib import Path

def load_config(path: str | None = None) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    cfg_path = Path(path) if path else Path(__file__).with_name("config.cfg")
    parser.read(cfg_path)
    return parser

# Load default configuration at import time
CONFIG = load_config()
