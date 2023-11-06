from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logging

from typing import Callable, TypeVar

from modules.objects import FullArticle

logger = logging.getLogger("osinter")


class Counter:
    def __init__(self) -> None:
        self.lock = multiprocessing.Manager().Lock()
        self.count = 0

    def get_count(self):
        with self.lock:
            self.count += 1
            return self.count


Input = TypeVar("Input")
Output = TypeVar("Output")


def process_threaded(
    inputs: list[Input], processor: Callable[[Input], Output], max_workers: int = 6
) -> list[Output]:
    counter = Counter()

    def process(input: Input) -> Output:
        process_count = counter.get_count()
        logger.debug(f"Starting processing of process nr {process_count}")
        result = processor(input)
        logger.debug(f"Finished processing of process nr {process_count}")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process, inputs))
