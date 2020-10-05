from typing import Callable, Iterable, Set, Union

import emoji


class Compose(object):
    def __init__(self, transforms: Iterable[Callable[[str], str]]) -> None:
        self.transforms = transforms

    def __call__(self, text: str) -> str:
        for transform in self.transforms:
            text = transform(text)

        return text


class Lowercase(object):
    def __call__(self, text: str) -> str:
        return text.lower()


class WordRemove(object):
    def __init__(self, words_to_remove: Union[str, Iterable[str]]) -> None:
        self._words_to_remove = None

        self.words_to_remove = words_to_remove

    def __call__(self, text: str) -> str:
        words = [
            word for word in text.split() if word not in self.words_to_remove
        ]

        return " ".join(words)

    @property
    def words_to_remove(self) -> Set[str]:
        return self._words_to_remove

    @words_to_remove.setter
    def words_to_remove(self, obj: Union[str, Iterable[str]]) -> None:
        if isinstance(obj, str):
            self._words_to_remove = {obj}
        else:
            self._words_to_remove = set(obj)


class Demojize(object):
    def __init__(self, use_aliases: bool = False) -> None:
        self.use_aliases = use_aliases

    def __call__(self, text: str) -> str:
        return emoji.demojize(text, use_aliases=self.use_aliases)
