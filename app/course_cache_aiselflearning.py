import time

course_cache = {}
CACHE_TTL_SECONDS = 60 * 30  # 30 นาที


def get_cached_course(course_no: int):
    item = course_cache.get(course_no)
    if not item:
        return None

    loaded_at = item.get("loaded_at", 0)
    if time.time() - loaded_at > CACHE_TTL_SECONDS:
        course_cache.pop(course_no, None)
        return None

    return item.get("data")


def set_cached_course(course_no: int, data):
    course_cache[course_no] = {
        "data": data,
        "loaded_at": time.time(),
    }


def clear_cached_course(course_no: int):
    course_cache.pop(course_no, None)