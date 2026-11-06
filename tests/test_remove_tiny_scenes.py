from trailer_editor.remove_tiny_scenes import build_concat_filter


class Timecode:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def get_seconds(self) -> float:
        return self.seconds


def test_concat_filter_decodes_once_and_resets_timestamps() -> None:
    graph = build_concat_filter([
        (Timecode(0.0), Timecode(1.25)),
        (Timecode(1.5), Timecode(3.0)),
    ])
    assert "trim=start=0.000000:end=1.250000" in graph
    assert "trim=start=1.500000:end=3.000000" in graph
    assert graph.count(",setpts=PTS-STARTPTS") == 2
    assert "concat=n=2:v=1:a=1" in graph
