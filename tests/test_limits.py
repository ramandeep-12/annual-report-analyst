from rag.limits import RateLimiter


def test_no_limits_by_default():
    limiter = RateLimiter(0, 0)
    assert all(limiter.check("1.2.3.4") is None for _ in range(100))


def test_per_visitor_hourly_limit():
    limiter = RateLimiter(per_hour=2, per_day_total=0)
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert "per hour" in limiter.check("a")
    assert limiter.check("b") is None  # other visitors unaffected


def test_daily_total_limit():
    limiter = RateLimiter(per_hour=0, per_day_total=2)
    limiter.check("a")
    limiter.check("b")
    assert "daily" in limiter.check("c")
