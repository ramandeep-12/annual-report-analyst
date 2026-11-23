"""Simple in-memory limits so a public demo can't run up the OpenAI bill.

Both are off unless set (see .env.example). They reset when the server restarts,
so also set a monthly spending cap in your OpenAI account.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_hour: int, per_day_total: int):
        self.per_hour = per_hour  # questions per visitor per hour (0 = no limit)
        self.per_day_total = per_day_total  # questions per day across everyone (0 = no limit)
        self._visitors: dict[str, deque[float]] = defaultdict(deque)
        self._day: tuple[int, int] = (0, 0)  # (day number, questions that day)
        self._lock = threading.Lock()

    def check(self, visitor: str) -> str | None:
        """Record one question. Returns a message if it's over a limit, else None."""
        now = time.time()
        with self._lock:
            day = int(now // 86400)
            day_count = self._day[1] if self._day[0] == day else 0
            if self.per_day_total and day_count >= self.per_day_total:
                return "The demo has reached its daily question limit. Please try again tomorrow."

            if self.per_hour:
                recent = self._visitors[visitor]
                while recent and now - recent[0] > 3600:
                    recent.popleft()
                if len(recent) >= self.per_hour:
                    wait = int((3600 - (now - recent[0])) // 60) + 1
                    return f"You've hit the limit of {self.per_hour} questions per hour. Try again in {wait} min."
                recent.append(now)

            self._day = (day, day_count + 1)
            return None
