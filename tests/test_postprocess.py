from trailer_editor.postprocess import (
    build_filtergraph,
    make_even,
    scaled_main_size,
)


def test_even_geometry() -> None:
    assert make_even(101) == 100
    assert scaled_main_size(721, 1281) == (612, 1088)


def test_filtergraph_uses_native_subtle_noise() -> None:
    graph = build_filtergraph(720, 1280, 720, 1280, fps=25, duration=30)
    assert "scale=612:1088" in graph
    assert "noise=alls=100:allf=t+u" in graph
    assert "blend=all_mode=overlay:all_opacity=0.01000" in graph
    assert "color=c=gray:s=720x1280:r=25.000" in graph
    assert "[2:v]fps=25.000" in graph
    assert "scale='trunc(iw*1.5/2)*2':'trunc(ih*1.5/2)*2'" in graph
    assert "colorchannelmixer=aa=0.05000" in graph
    assert "rotate=PI/4" in graph
    assert "x='-w+(W+w)*t/30.000000'" in graph
    assert "y='-h+(H+h)*t/30.000000'" in graph
    assert "[final]" in graph
    assert "stream_loop" not in graph
