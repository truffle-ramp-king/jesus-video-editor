from trailer_editor.render import timestamp_seconds


def test_timestamp_seconds() -> None:
    assert timestamp_seconds("01:01:01.250") == 3661.25
