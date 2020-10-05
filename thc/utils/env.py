import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from thc.utils.constants import REPOSITORY_NAME


def load_env_variable(name: str) -> str:
    load_dotenv(find_dotenv())

    try:
        return os.environ[name]
    except KeyError as err:
        raise EnvironmentError(
            f"Variable {name} is missing. "
            "Declare this variable inside the .env file."
        ).with_traceback(err.__traceback__)


def check_repository_path() -> Path:
    repository_dir = Path(load_env_variable("REPOSITORY_PATH")).resolve()

    is_correct = repository_dir.name == REPOSITORY_NAME
    exists = repository_dir.is_dir()

    if not (is_correct and exists):
        raise EnvironmentError(
            "Specified repository path in the .env file is incompatible."
        )

    return repository_dir
