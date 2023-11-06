from dotenv import load_dotenv

from modules.config import BaseConfig, configure_logger
from modules.misc import create_folder

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

create_folder("logs")
create_folder("models")

load_dotenv()

config_options = BaseConfig()
configure_logger("osinter")
