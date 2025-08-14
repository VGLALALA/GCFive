import utility.config_reader as config_reader


def test_load_config_defaults():
    cfg = config_reader.load_config()
    assert cfg.getint("Camera", "roi_w") == 640
    assert cfg.get("YOLO", "model_path") == "data/model/golfballv4.pt"
    assert not cfg.getboolean("Calibration", "recalibrate_hitting_zone")
    assert cfg.getint("Spin", "coarse_x_inc") == 6
