import os
import logging
import logging.config
from pathlib import Path
import json
import datetime

DIR_BASE = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood")
CONFIG_PATH = DIR_BASE / "src/logger/config.json"


def setup_logging(CONFIG_PATH, name=None):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    file_name = Path(config["handlers"]["file"]["filename"])
    if name is not None:
        config["handlers"]["file"]["filename"] = file_name.with_stem(name)
    else:
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        config["handlers"]["file"]["filename"] = file_name.with_stem(today_date)
    logging.config.dictConfig(config)


def create_logger(name=None):
    setup_logging(CONFIG_PATH, name)
    return logging.getLogger(name)
