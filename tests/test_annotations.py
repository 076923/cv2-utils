import cv2
import numpy as np
import pytest

from cv2u.core.annotations import (
    __all__,
    boxesToPolygons,
    boxesToYolo,
    colorToLabelMap,
    drawPolygons,
    drawSegmentation,
    flipBoxes,
    flipPolygons,
    instanceMasksToLabelMap,
    labelMapToColor,
    labelMapToMasks,
    maskIoU,
    maskToPolygons,
    polygonArea,
    polygonIoU,
    polygonsToBoxes,
    polygonsToMask,
    readYoloLabels,
    rleDecode,
    rleEncode,
    writeYoloLabels,
    yoloToBoxes,
)

SIZE = (160, 120)  # width, height


@pytest.fixture
def mask():
    # A square on the left, a circle on the right.
    img = np.zeros((120, 160), np.uint8)
    cv2.rectangle(img, (20, 20), (69, 69), 255, cv2.FILLED)
    cv2.circle(img, (115, 60), 25, 255, cv2.FILLED)
    return img


def test_names_do_not_exist_in_cv2():
    assert not [name for name in __all__ if hasattr(cv2, name)]


# --- masks and polygons ----------------------------------------------------------


def test_maskToPolygons_shapes_and_order(mask):
    polygons = maskToPolygons(mask)
    assert len(polygons) == 2
    assert all(p.ndim == 2 and p.shape[1] == 2 and p.dtype == np.int32 for p in polygons)
    assert cv2.contourArea(polygons[0]) >= cv2.contourArea(polygons[1])  # square (2500) first, then circle


def test_maskToPolygons_epsilon_simplifies_and_minArea_filters(mask):
    mask[5, 5] = 255  # a one-pixel speck
    polygons = maskToPolygons(mask, epsilon=0.02)
    assert len(polygons) == 2  # the speck has fewer than 3 points
    square = min(polygons, key=len)
    assert len(square) == 4
    assert len(maskToPolygons(mask, minArea=2000)) == 1


def test_maskToPolygons_bool_and_color_input(mask):
    assert len(maskToPolygons(mask.astype(bool))) == 2
    assert len(maskToPolygons(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))) == 2
    assert maskToPolygons(np.zeros((10, 10), np.uint8)) == []


def test_maskToPolygons_validation(mask):
    with pytest.raises(ValueError, match="epsilon"):
        maskToPolygons(mask, epsilon=-1)
    with pytest.raises(ValueError, match="minArea"):
        maskToPolygons(mask, minArea=float("nan"))
    with pytest.raises(ValueError, match="mask"):
        maskToPolygons(np.zeros((0, 5), np.uint8))


def test_polygonsToMask_round_trip(mask):
    out = polygonsToMask(maskToPolygons(mask), SIZE)
    assert out.shape == (120, 160) and out.dtype == np.uint8
    assert maskIoU(out, mask) > 0.97


def test_polygonsToMask_single_polygon_value_and_empty():
    out = polygonsToMask(np.array([[0, 0], [9.6, 0], [9.6, 9.6], [0, 9.6]]), (20, 20), value=7)
    assert set(np.unique(out)) == {0, 7}
    assert out[0, 0] == 7 and out[15, 15] == 0
    assert polygonsToMask([], (4, 3)).shape == (3, 4)
    assert not polygonsToMask([], (4, 3)).any()
    assert polygonsToMask([[(1, 1), (3, 1), (3, 3)]], (5, 5)).any()


def test_polygonsToMask_validation():
    with pytest.raises(ValueError, match="size"):
        polygonsToMask([], (0, 5))
    with pytest.raises(ValueError, match="value"):
        polygonsToMask([], (5, 5), value=0)
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask([np.zeros((0, 2))], (5, 5))
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask(np.zeros((0, 2)), (5, 5))
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask([[(1, 2, 3)]], (5, 5))
    with pytest.raises(ValueError, match="finite"):
        polygonsToMask([[(1, np.nan), (2, 2), (3, 3)]], (5, 5))
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask(np.zeros((2, 3)), (5, 5))
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask([[[[[1, 2]]]]], (5, 5))  # 5-D: not a polygon list
    with pytest.raises(ValueError, match="polygons"):
        polygonsToMask([np.zeros((2, 2, 2))], (5, 5))  # (N, 2, 2) is neither (N, 2) nor (N, 1, 2)
    assert polygonsToMask(np.zeros((3, 1, 2), np.int32), (5, 5)).shape == (5, 5)  # one (N, 1, 2) contour


def test_boxesToPolygons_and_back():
    boxes = np.array([[10, 20, 30, 40], [0.5, 1.5, 2, 3]])
    polygons = boxesToPolygons(boxes)
    assert len(polygons) == 2 and polygons[0].shape == (4, 2)
    np.testing.assert_array_equal(polygons[0], [[10, 20], [40, 20], [40, 60], [10, 60]])
    np.testing.assert_allclose(polygonsToBoxes(polygons), boxes)
    assert boxesToPolygons([]) == []
    assert polygonsToBoxes([]).shape == (0, 4)
    assert polygonsToBoxes(np.array([[1, 1], [4, 1], [4, 5]])).tolist() == [[1, 1, 3, 4]]
    with pytest.raises(ValueError, match="boxes"):
        boxesToPolygons([1, 2, 3])


def test_polygonArea():
    assert polygonArea([(0, 0), (4, 0), (4, 3), (0, 3)]) == 12.0
    assert polygonArea([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]) == pytest.approx(0.25)  # cv.contourArea gives 0
    assert polygonArea([(0, 0), (1, 1)]) == 0.0
    with pytest.raises(ValueError, match="polygon"):
        polygonArea([])


def test_polygonIoU_precision():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(5, 0), (15, 0), (15, 10), (5, 10)]
    assert polygonIoU(a, b) == pytest.approx(1 / 3, abs=0.01)
    assert polygonIoU(a, a) == pytest.approx(1.0, abs=0.005)
    assert polygonIoU(a, [(20, 20), (30, 20), (30, 30)]) == 0.0
    # Concave shape (an L) against its bounding square: 3/4 of the square.
    ell = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    assert polygonIoU(ell, a) == pytest.approx(0.75, abs=0.01)
    # Normalized coordinates work too (the raster is scaled to the resolution).
    tiny_a = [(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]
    tiny_b = [(0.05, 0), (0.15, 0), (0.15, 0.1), (0.05, 0.1)]
    assert polygonIoU(tiny_a, tiny_b) == pytest.approx(1 / 3, abs=0.01)


def test_polygonIoU_degenerate_and_validation():
    assert polygonIoU([(0, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)]) == 0.0
    assert polygonIoU([(1, 1), (1, 1), (1, 1)], [(1, 1), (1, 1), (1, 1)]) == 0.0
    with pytest.raises(ValueError, match="resolution"):
        polygonIoU([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)], resolution=0)
    with pytest.raises(ValueError, match="polygonB"):
        polygonIoU([(0, 0), (1, 0), (1, 1)], [])


def test_maskIoU(mask):
    assert maskIoU(mask, mask) == 1.0
    assert maskIoU(mask, np.zeros_like(mask)) == 0.0
    assert maskIoU(np.zeros_like(mask), np.zeros_like(mask)) == 0.0
    half = mask.copy()
    half[:, 80:] = 0
    inter = np.count_nonzero(half)
    assert maskIoU(mask, half.astype(bool)) == pytest.approx(inter / np.count_nonzero(mask))
    with pytest.raises(ValueError, match="maskB"):
        maskIoU(mask, mask[:50])


# --- RLE -------------------------------------------------------------------------


def test_rle_round_trip(mask):
    rle = rleEncode(mask)
    assert rle["size"] == [120, 160]
    assert all(isinstance(c, int) for c in rle["counts"])
    assert sum(rle["counts"]) == 120 * 160
    np.testing.assert_array_equal(rleDecode(rle), mask)


def test_rleEncode_column_major_like_pycocotools():
    m = np.array([[1, 0, 1], [1, 1, 0]], np.uint8)
    # Fortran order: column 0 = (1, 1), column 1 = (0, 1), column 2 = (1, 0) -> 1 1 0 1 1 0
    assert rleEncode(m) == {"counts": [0, 2, 1, 2, 1], "size": [2, 3]}
    assert rleEncode(np.zeros((2, 2), bool)) == {"counts": [4], "size": [2, 2]}
    assert rleEncode(np.ones((2, 2), bool)) == {"counts": [0, 4], "size": [2, 2]}
    np.testing.assert_array_equal(rleDecode(rleEncode(m)), m * 255)


def test_rleDecode_validation():
    with pytest.raises(ValueError, match="size"):
        rleDecode({"counts": [], "size": [0, 3]})
    with pytest.raises(ValueError, match="add up"):
        rleDecode({"counts": [1, 2], "size": [2, 2]})
    with pytest.raises(ValueError, match="negative"):
        rleDecode({"counts": [-1, 5], "size": [2, 2]})
    with pytest.raises(ValueError, match="compressed"):
        rleDecode({"counts": "abc", "size": [2, 2]})
    with pytest.raises(ValueError, match="mask"):
        rleEncode(np.zeros((0, 3), np.uint8))


# --- YOLO ------------------------------------------------------------------------


def test_boxesToYolo_and_back():
    boxes = np.array([[10, 20, 40, 60], [0, 0, 160, 120]])
    rows = boxesToYolo(boxes, SIZE)
    assert rows.shape == (2, 4)
    np.testing.assert_allclose(rows[0], [30 / 160, 50 / 120, 0.25, 0.5])
    with_ids = boxesToYolo(boxes, SIZE, classIds=[3, 0])
    assert with_ids.shape == (2, 5) and with_ids[0, 0] == 3
    back, ids = yoloToBoxes(with_ids, SIZE)
    np.testing.assert_allclose(back, boxes)
    assert ids.dtype == np.int64 and ids.tolist() == [3, 0]
    back2, ids2 = yoloToBoxes(rows, SIZE)
    np.testing.assert_allclose(back2, boxes)
    assert ids2 is None
    assert boxesToYolo([], SIZE).shape == (0, 4)
    single, _ = yoloToBoxes([0, 0.5, 0.5, 1, 1], (10, 10))
    np.testing.assert_allclose(single, [[0, 0, 10, 10]])


def test_yolo_extra_columns_and_empty():
    boxes, ids = yoloToBoxes([[1, 0.5, 0.5, 0.5, 0.5, 0.9]], (10, 10))
    np.testing.assert_allclose(boxes, [[2.5, 2.5, 5, 5]])
    assert ids.tolist() == [1]
    boxes, ids = yoloToBoxes([], (10, 10))
    assert boxes.shape == (0, 4) and ids.shape == (0,)


def test_yolo_validation():
    with pytest.raises(ValueError, match="size"):
        boxesToYolo([[0, 0, 1, 1]], (0, 10))
    with pytest.raises(ValueError, match="classIds"):
        boxesToYolo([[0, 0, 1, 1]], SIZE, classIds=[1, 2])
    with pytest.raises(ValueError, match="boxes"):
        boxesToYolo([[0, 0, 1]], SIZE)
    with pytest.raises(ValueError, match="rows"):
        yoloToBoxes([[0, 0.5, 0.5]], SIZE)
    with pytest.raises(ValueError, match="finite"):
        yoloToBoxes([[0, np.nan, 0.5, 0.5, 0.5]], SIZE)
    with pytest.raises(ValueError, match="size"):
        yoloToBoxes([[0, 0.5, 0.5, 0.5, 0.5]], (5, -1))


def test_yolo_label_files_round_trip(tmp_path):
    path = tmp_path / "한글" / "labels" / "img_001.txt"
    boxes = np.array([[10, 20, 40, 60], [100, 50, 30, 30]], dtype=np.float64)
    writeYoloLabels(path, boxes, [2, 7], SIZE, makeDirs=True)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("2 0.187500 0.416667 0.250000 0.500000\n")
    assert text.count("\n") == 2
    read_boxes, ids = readYoloLabels(path, SIZE)
    np.testing.assert_allclose(read_boxes, boxes, atol=1e-3)
    assert ids.tolist() == [2, 7]


def test_yolo_label_files_empty_missing_and_blank_lines(tmp_path):
    path = tmp_path / "empty.txt"
    writeYoloLabels(path, [], [], SIZE)
    assert path.read_text(encoding="utf-8") == ""
    boxes, ids = readYoloLabels(path, SIZE)
    assert boxes.shape == (0, 4) and ids.shape == (0,)
    boxes, ids = readYoloLabels(tmp_path / "missing.txt", SIZE)
    assert boxes.shape == (0, 4)
    path.write_text("\n0 0.5 0.5 0.5 0.5 0.99\n\n", encoding="utf-8")
    boxes, ids = readYoloLabels(path, (10, 10))
    np.testing.assert_allclose(boxes, [[2.5, 2.5, 5, 5]])


def test_yolo_label_files_validation(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        readYoloLabels(path, SIZE)
    with pytest.raises(ValueError, match="size"):
        readYoloLabels(path, (0, 0))
    with pytest.raises(FileNotFoundError):
        writeYoloLabels(tmp_path / "no" / "dir.txt", [[0, 0, 1, 1]], [0], SIZE)


# --- label maps ------------------------------------------------------------------


def test_labelMapToColor_default_palette_and_inverse():
    labels = np.array([[0, 1, 2], [2, 1, 0]], np.int32)
    color = labelMapToColor(labels)
    assert color.shape == (2, 3, 3) and color.dtype == np.uint8
    assert color[0, 0].tolist() == [0, 0, 0]
    assert color[0, 1].tolist() == [0, 255, 0]  # distinctColors(3)[1]
    assert color[0, 2].tolist() == [255, 0, 0]
    from cv2u.imgproc.color import distinctColors

    np.testing.assert_array_equal(colorToLabelMap(color, distinctColors(3)), labels)


def test_labelMapToColor_custom_palette_and_background():
    labels = np.array([[0, 1], [1, 2]], np.uint8)
    palette = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    color = labelMapToColor(labels, palette)
    assert color[0, 0].tolist() == [0, 0, 0]  # background 0 is black
    assert color[1, 1].tolist() == [70, 80, 90]
    color = labelMapToColor(labels, palette, background=-1)
    assert color[0, 0].tolist() == [10, 20, 30]
    color = labelMapToColor(labels.astype(bool), palette, background=5)
    assert color[0, 1].tolist() == [40, 50, 60]
    back = colorToLabelMap(np.dstack([color, np.full(labels.shape, 255, np.uint8)]), palette)
    assert back.dtype == np.int32
    np.testing.assert_array_equal(back, labels.astype(bool))


def test_colorToLabelMap_unknown_colors_and_duplicates():
    img = np.array([[[1, 2, 3], [9, 9, 9], [1, 2, 3]]], np.uint8)
    palette = [(1, 2, 3), (4, 5, 6), (1, 2, 3)]
    np.testing.assert_array_equal(colorToLabelMap(img, palette, background=7), [[0, 7, 0]])
    np.testing.assert_array_equal(colorToLabelMap(img, [(200, 200, 200)]), [[0, 0, 0]])


def test_label_map_validation():
    with pytest.raises(ValueError, match="labels"):
        labelMapToColor(np.zeros((2, 2, 3), np.int32))
    with pytest.raises(ValueError, match="labels"):
        labelMapToColor(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="negative"):
        labelMapToColor(np.array([[-1, 0]]))
    with pytest.raises(ValueError, match="labels"):
        labelMapToColor(np.zeros((0, 2), np.int32))
    with pytest.raises(ValueError, match="palette"):
        labelMapToColor(np.array([[0, 3]]), [(0, 0, 0), (1, 1, 1)])
    with pytest.raises(ValueError, match="palette"):
        labelMapToColor(np.array([[0, 1]]), [(0, 0, 0), (256, 1, 1)])
    with pytest.raises(ValueError, match="image"):
        colorToLabelMap(np.zeros((2, 2), np.uint8), [(0, 0, 0)])
    with pytest.raises(ValueError, match="image"):
        colorToLabelMap(np.zeros((0, 2, 3), np.uint8), [(0, 0, 0)])
    with pytest.raises(ValueError, match="palette"):
        colorToLabelMap(np.zeros((2, 2, 3), np.uint8), [1, 2])
    with pytest.raises(ValueError, match="palette"):
        colorToLabelMap(np.zeros((2, 2, 3), np.uint8), [(np.inf, 0, 0)])


def test_instanceMasksToLabelMap_and_back():
    a = np.zeros((4, 4), bool)
    a[:2, :2] = True
    b = np.zeros((4, 4), np.uint8)
    b[1:3, 1:3] = 255
    labels = instanceMasksToLabelMap([a, b])
    assert labels.dtype == np.int32 and labels.shape == (4, 4)
    assert labels[0, 0] == 1 and labels[1, 1] == 2 and labels[3, 3] == 0  # later masks on top
    masks = labelMapToMasks(labels)
    assert len(masks) == 2 and all(m.dtype == np.uint8 for m in masks)
    np.testing.assert_array_equal(masks[1], b)
    np.testing.assert_array_equal(masks[0] != 0, a & ~b.astype(bool))
    # Unused ids give empty masks, so the index stays aligned with the id.
    sparse = labelMapToMasks(np.array([[0, 3]]))
    assert len(sparse) == 3 and not sparse[0].any() and sparse[2].tolist() == [[0, 255]]
    assert len(labelMapToMasks(np.array([[0, 3]]), background=2)) == 2
    assert labelMapToMasks(np.zeros((2, 2), np.int32)) == []


def test_instanceMasksToLabelMap_validation():
    with pytest.raises(ValueError, match="masks"):
        instanceMasksToLabelMap([])
    with pytest.raises(ValueError, match="masks"):
        instanceMasksToLabelMap([np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8)])
    with pytest.raises(ValueError, match="labels"):
        labelMapToMasks(np.zeros((2, 2), np.float64))


# --- drawing ---------------------------------------------------------------------


@pytest.fixture
def canvas():
    return np.zeros((60, 80, 3), np.uint8)


def test_drawPolygons_in_place_distinct_colors(canvas):
    tri = [(5, 5), (30, 5), (30, 30)]
    quad = np.array([[40, 10], [70, 10], [70, 40], [40, 40]], np.float32)
    out = drawPolygons(canvas, [tri, quad])
    assert out is canvas
    assert canvas[5, 10].tolist() == [0, 0, 255]  # first distinct color
    assert canvas[10, 55].tolist() == [0, 255, 0]
    assert canvas[25, 55].tolist() == [0, 0, 0]  # outline only


def test_drawPolygons_fill_alpha_labels_and_single_color(canvas):
    quad = [(10, 10), (50, 10), (50, 50), (10, 50)]
    drawPolygons(canvas, quad, color=(0, 0, 200), fill=True, alpha=0.5)
    assert canvas[30, 30].tolist() == [0, 0, 100]
    drawPolygons(canvas, [quad], color=[(255, 255, 255)], labels=["cat"])
    assert canvas.max() == 255
    assert (canvas[:10, 10:30] != 0).any()  # the label tag above the polygon
    gray = np.zeros((60, 80), np.uint8)
    drawPolygons(gray, [quad], color=200, thickness=1, lineType=cv2.LINE_8)
    assert gray[10, 30] == 200
    assert drawPolygons(gray, [], alpha=0.0) is gray
    drawPolygons(gray, [quad], color=255, labels=["x"], alpha=0.0)  # alpha 0 draws nothing
    assert gray[10, 30] == 200


def test_drawPolygons_label_at_top_edge(canvas):
    drawPolygons(canvas, [[(0, 0), (20, 0), (20, 20)]], color=(0, 0, 255), labels=["top"])
    assert canvas[0, 1].tolist() != [0, 0, 0]


def test_drawPolygons_validation(canvas):
    quad = [(10, 10), (50, 10), (50, 50), (10, 50)]
    with pytest.raises(ValueError, match="src"):
        drawPolygons(canvas.astype(np.float32), [quad])
    with pytest.raises(ValueError, match="src"):
        drawPolygons(np.zeros((0, 4, 3), np.uint8), [quad])
    with pytest.raises(ValueError, match="alpha"):
        drawPolygons(canvas, [quad], alpha=1.5)
    with pytest.raises(ValueError, match="thickness"):
        drawPolygons(canvas, [quad], thickness=0)
    with pytest.raises(ValueError, match="labels"):
        drawPolygons(canvas, [quad], labels=["a", "b"])
    with pytest.raises(ValueError, match="color"):
        drawPolygons(canvas, [quad], color=[(0, 0, 0), (1, 1, 1)])
    with pytest.raises(ValueError, match="color"):
        drawPolygons(canvas, [quad], color=(0, 0, 300))
    with pytest.raises(ValueError, match="polygons"):
        drawPolygons(canvas, [[(1, 2, 3)]])


def test_drawSegmentation_label_map_and_mask(mask):
    src = np.full((120, 160, 3), 100, np.uint8)
    labels = np.where(mask != 0, 2, 0).astype(np.int32)
    labels[:10, :10] = 1
    palette = [(0, 0, 0), (0, 0, 200), (200, 0, 0)]
    out = drawSegmentation(src, labels, alpha=0.5, palette=palette)
    assert out.shape == src.shape and out.dtype == np.uint8 and out is not src
    assert src[0, 0].tolist() == [100, 100, 100]  # not modified
    assert out[100, 100].tolist() == [100, 100, 100]  # background untouched
    assert out[5, 5].tolist() == [50, 50, 150]
    assert out[45, 45].tolist() == [150, 50, 50]
    # Binary masks (uint8 0/255 or bool) are one class in the first palette color; alpha 1 paints it solid.
    masked = drawSegmentation(src, mask, alpha=1.0, palette=palette)
    assert masked[45, 45].tolist() == [0, 0, 200]
    assert masked[100, 100].tolist() == [100, 100, 100]
    np.testing.assert_array_equal(drawSegmentation(src, mask.astype(bool), 1.0, palette), masked)
    default = drawSegmentation(src, mask)
    assert default[45, 45].tolist() == [50, 178, 50]  # distinctColors(2)[1] = (0, 255, 0), blended at 0.5
    gray = drawSegmentation(np.full((120, 160), 100, np.uint8), labels, palette=palette)
    assert gray.shape == (120, 160, 3)
    bgra = drawSegmentation(np.full((120, 160, 4), 100, np.uint8), labels, palette=palette)
    assert bgra.shape == (120, 160, 3)


def test_drawSegmentation_validation(mask):
    src = np.zeros((120, 160, 3), np.uint8)
    with pytest.raises(ValueError, match="src"):
        drawSegmentation(src.astype(np.float32), mask)
    with pytest.raises(ValueError, match="src"):
        drawSegmentation(np.zeros((0, 160, 3), np.uint8), mask)
    with pytest.raises(ValueError, match="alpha"):
        drawSegmentation(src, mask, alpha=-0.1)
    with pytest.raises(ValueError, match="labels"):
        drawSegmentation(src, mask[:50])


# --- flips -----------------------------------------------------------------------


def test_flipBoxes_matches_cv_flip():
    img = np.zeros((120, 160), np.uint8)
    box = (10, 20, 40, 30)
    img[20:50, 10:50] = 255
    for code in (0, 1, -1):
        flipped = cv2.flip(img, code)
        x, y, w, h = flipBoxes(box, SIZE, code)
        assert (x, y, w, h) == tuple(cv2.boundingRect(flipped))
    assert flipBoxes([box, box], SIZE, 1).shape == (2, 4)
    assert flipBoxes(box, SIZE, 1).shape == (4,)
    assert flipBoxes([], SIZE, 1).shape == (0, 4)
    np.testing.assert_array_equal(flipBoxes(flipBoxes(box, SIZE, -1), SIZE, -1), box)


def test_flipPolygons():
    pts = np.array([[0, 0], [10, 5]], np.float64)
    np.testing.assert_array_equal(flipPolygons([pts], SIZE, 1)[0], [[160, 0], [150, 5]])
    np.testing.assert_array_equal(flipPolygons([pts], SIZE, 0)[0], [[0, 120], [10, 115]])
    np.testing.assert_array_equal(flipPolygons([[(0, 0)]], SIZE, -1)[0], [[160, 120]])
    out = flipPolygons([pts, [(1, 1), (2, 2), (3, 1)]], SIZE, 1)
    assert len(out) == 2 and out[1].shape == (3, 2) and out[1].dtype == np.float64
    assert flipPolygons([], SIZE, 0) == []
    # The flipped mask of a polygon is the polygon of the flipped mask.
    tri = [(20, 10), (120, 30), (40, 100)]
    a = cv2.flip(polygonsToMask([tri], SIZE), 1)
    b = polygonsToMask(flipPolygons([tri], SIZE, 1), SIZE)
    assert maskIoU(a, b) > 0.95


def test_flip_validation():
    with pytest.raises(ValueError, match="flipCode"):
        flipBoxes((0, 0, 1, 1), SIZE, 2)
    with pytest.raises(ValueError, match="size"):
        flipBoxes((0, 0, 1, 1), (0, 1), 1)
    with pytest.raises(ValueError, match="boxes"):
        flipBoxes((0, 0, 1), SIZE, 1)
    with pytest.raises(ValueError, match="flipCode"):
        flipPolygons([[(0, 0)]], SIZE, 3)
    with pytest.raises(ValueError):
        flipPolygons([[]], SIZE, 1)
    with pytest.raises(ValueError, match="size"):
        flipPolygons([], (1, 0), 1)
    with pytest.raises(ValueError, match="flipCode"):
        flipPolygons([], SIZE, 5)


# --- regression: review findings ----------------------------------------------------------------------------------


def test_drawPolygons_label_text_is_white_on_dark_boxes(canvas):
    # A scalar 255 text color is expanded by OpenCV to (255, 0, 0): blue text on a BGR image.
    quad = [(10, 20), (70, 20), (70, 50), (10, 50)]
    drawPolygons(canvas, [quad], color=(0, 0, 0), labels=["dark"], thickness=1, lineType=cv2.LINE_8)
    tag = canvas[:20, 10:50].reshape(-1, 3)
    text = tag[tag.any(axis=1)]
    assert len(text) > 0
    assert np.all(text[:, 0] == text[:, 1]) and np.all(text[:, 1] == text[:, 2])  # gray levels, not blue
    assert text.max() == 255
    bgra = np.zeros((60, 80, 4), np.uint8)
    drawPolygons(bgra, [quad], color=(0, 0, 0, 255), labels=["dark"], thickness=1, lineType=cv2.LINE_8)
    text = bgra[:20, 10:50].reshape(-1, 4)
    text = text[text[:, :3].any(axis=1)]
    assert np.all(text[:, 0] == text[:, 2]) and np.all(text[:, 0] == text[:, 3]) and text.max() == 255


def test_polygons_accepts_equal_length_contours_and_mixed_forms():
    square = np.array([[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], np.int32)  # (N, 1, 2) like cv.findContours
    other = square + 3
    mask = polygonsToMask([square, other], (10, 10))  # equal lengths: np.asarray would give a 4-D array
    assert mask.shape == (10, 10) and mask[2, 2] == 255 and mask[6, 6] == 255 and mask[0, 0] == 0
    mixed = [square, square.reshape(-1, 2) + 3, [(0, 8), (2, 8), (2, 9)]]  # contour, (N, 2), point list
    assert len(polygonsToBoxes(mixed)) == 3
    ragged = [square, [(0, 8), (2, 8), (2, 9), (1, 9), (0, 9)]]
    assert len(polygonsToBoxes(ragged)) == 2
    assert polygonsToBoxes([square])[0].tolist() == [1, 1, 3, 3]


def test_drawPolygons_numpy_array_is_one_color(canvas):
    quad = [(10, 10), (50, 10), (50, 50), (10, 50)]
    color = np.array([0, 0, 255], np.uint8)  # e.g. a row of Clusters.centers
    drawPolygons(canvas, [quad, quad], color=color, fill=True)
    assert canvas[30, 30].tolist() == [0, 0, 255]
    gray = np.zeros((60, 80), np.uint8)
    drawPolygons(gray, [quad], color=np.uint8(200), fill=True)
    assert gray[30, 30] == 200
    centers = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    drawPolygons(canvas, [quad, [(60, 10), (70, 10), (70, 20)]], color=centers, fill=True)
    assert canvas[30, 30].tolist() == [255, 0, 0]
