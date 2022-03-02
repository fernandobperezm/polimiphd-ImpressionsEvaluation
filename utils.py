import functools
from typing import Any, Callable, TypeVar

RT = TypeVar("RT", bound=Callable[..., Any])


def typed_cache(
    user_function: Callable[..., RT]
) -> Callable[..., RT]:
    return functools.cache(user_function)
