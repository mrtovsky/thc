import codecs
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch.utils.data import Dataset


class TweetsDataset(Dataset):
    def __init__(
        self,
        text_file: Union[str, Path],
        tags_file: Union[str, Path],
        text_open_params: Optional[Dict[str, Any]] = None,
        tags_open_params: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable[[str], str]] = None,
    ) -> None:
        if text_open_params is None:
            text_open_params = {"mode": "r", "encoding": "utf-8"}
        if tags_open_params is None:
            tags_open_params = {"mode": "r"}

        with codecs.open(str(text_file), **text_open_params) as file:
            self.texts = file.read().splitlines()
        with codecs.open(str(tags_file), **tags_open_params) as file:
            self.tags = [int(tag) for tag in file]

        if len(self.texts) != len(self.tags):
            raise TypeError("Files length mismatch")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        sample = self.texts[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.tags[index]
