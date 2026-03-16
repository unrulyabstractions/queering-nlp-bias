"""Rate-limited async executor for API calls with retry logic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

# Rate limit exceptions from both providers
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError as OpenAIRateLimitError

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ExecutorConfig:
    """Configuration for rate-limited executor."""

    max_concurrent: int = 3
    max_retries: int = 5
    base_retry_delay: float = 15.0


@dataclass
class TaskResult:
    """Result from an executed task."""

    key: Any  # Identifier for the task
    result: Any | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class RateLimitedExecutor:
    """Executes async tasks with concurrency control and retry logic.

    Handles rate limiting transparently - tasks are queued and executed
    with controlled concurrency. Rate limit errors trigger automatic
    retries with exponential backoff.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error from any provider."""
        return isinstance(error, (AnthropicRateLimitError, OpenAIRateLimitError))

    async def execute(
        self,
        key: Any,
        coro_func: Callable[..., Awaitable[R]],
        *args: Any,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a single task with rate limiting and retries.

        Args:
            key: Identifier for tracking this task
            coro_func: Async function to call
            *args, **kwargs: Arguments to pass to coro_func

        Returns:
            TaskResult with the result or error
        """
        async with self._semaphore:
            last_error = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    result = await coro_func(*args, **kwargs)
                    return TaskResult(key=key, result=result)
                except Exception as e:
                    last_error = e
                    if self._is_rate_limit_error(e) and attempt < self.config.max_retries:
                        delay = self.config.base_retry_delay * (2 ** attempt)
                        print(f"⚠️ Rate limit hit, retrying in {delay:.0f}s (attempt {attempt + 1}/{self.config.max_retries})")
                        await asyncio.sleep(delay)
                    elif not self._is_rate_limit_error(e):
                        # Non-rate-limit errors should not retry
                        return TaskResult(key=key, error=e)

            return TaskResult(key=key, error=last_error)

    async def execute_all(
        self,
        tasks: list[tuple[Any, Callable[..., Awaitable[R]], tuple, dict]],
    ) -> AsyncIterator[TaskResult]:
        """Execute multiple tasks, yielding results as they complete.

        Args:
            tasks: List of (key, coro_func, args, kwargs) tuples

        Yields:
            TaskResult for each completed task, in completion order
        """
        if not tasks:
            return

        # Create all task coroutines
        async_tasks = [
            asyncio.create_task(self.execute(key, func, *args, **kwargs))
            for key, func, args, kwargs in tasks
        ]

        # Yield results as they complete
        for completed in asyncio.as_completed(async_tasks):
            yield await completed

    async def map_unordered(
        self,
        func: Callable[[T], Awaitable[R]],
        items: list[T],
        key_func: Callable[[T], Any] | None = None,
    ) -> AsyncIterator[TaskResult]:
        """Apply func to each item, yielding results as they complete.

        Args:
            func: Async function to apply to each item
            items: Items to process
            key_func: Optional function to extract key from item (default: item itself)

        Yields:
            TaskResult for each completed task, in completion order
        """
        if key_func is None:
            key_func = lambda x: x

        tasks = [(key_func(item), func, (item,), {}) for item in items]
        async for result in self.execute_all(tasks):
            yield result
