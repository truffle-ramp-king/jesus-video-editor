from trailer_editor.cli import build_parser


def test_duration_flag_accepts_seconds() -> None:
    args = build_parser().parse_args(["source.mp4", "--duration", "60"])
    assert args.duration == 60.0


def test_deduplication_is_opt_in():
    parser = build_parser()
    assert not parser.parse_args(["source.mp4"]).deduplicate
    assert parser.parse_args(["source.mp4", "--deduplicate"]).deduplicate
    assert not parser.parse_args(["source.mp4", "--no-deduplicate"]).deduplicate
