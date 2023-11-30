from dotenv import load_dotenv

from modules.config import BaseConfig, configure_logger
from modules.misc import create_folder


create_folder("logs")
create_folder("models")

load_dotenv()

config_options = BaseConfig()
configure_logger("osinter")
