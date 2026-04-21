def get_course_data_by_nos(conn, course_nos: list[str]):
    if not course_nos:
        return []

    clean_course_nos = [str(x).strip() for x in course_nos if str(x).strip()]
    if not clean_course_nos:
        return []

    placeholders = ",".join(["%s"] * len(clean_course_nos))

    cur = conn.cursor()
    sql = f"""
        SELECT
            OCourse_no,
            course,
            script
        FROM ai_data_sl
        WHERE OCourse_no IN ({placeholders})
    """
    cur.execute(sql, tuple(clean_course_nos))
    rows = cur.fetchall()
    cur.close()

    return rows