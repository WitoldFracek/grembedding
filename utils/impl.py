from typing import NoReturn


def panic(msg: str = "") -> NoReturn:
    raise Exception(msg)


def todo() -> NoReturn:
    import inspect
    caller_name = inspect.currentframe().f_back.f_code.co_name
    raise Exception(f"\"{caller_name}\" not yet implemented")