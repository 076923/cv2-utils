import cv2
import numpy as np
import pytest

import cv2u.dnn as dnn
from cv2u.dnn.inference import (
    Detections,
    PreprocessInfo,
    drawDetections,
    inferImage,
    loadModel,
    postprocessClassification,
    postprocessDetections,
    preprocessImage,
    readClassNames,
    sigmoid,
    softmax,
)

HAS_DNN = hasattr(cv2, "dnn")


# --- package: passthrough to cv2.dnn ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_package_passes_through_cv2_dnn():
    assert dnn.readNet is cv2.dnn.readNet
    assert dnn.NMSBoxes is cv2.dnn.NMSBoxes
    assert dnn.DNN_BACKEND_OPENCV == cv2.dnn.DNN_BACKEND_OPENCV
    names = dir(dnn)
    assert "readNet" in names and "preprocessImage" in names
    assert dnn.preprocessImage is preprocessImage


def test_package_unknown_attribute():
    with pytest.raises(AttributeError, match=r"cv2u.dnn"):
        _ = dnn.noSuchThing


def test_public_names_not_in_cv2():
    for name in dnn.__all__:
        assert not hasattr(cv2, name), name
        if HAS_DNN:
            assert not hasattr(cv2.dnn, name), name


# --- softmax / sigmoid -------------------------------------------------------------------------------------------


def test_softmax_sums_to_one_and_is_stable():
    logits = np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), [1, 1], rtol=1e-6)
    np.testing.assert_allclose(probs[1], [1 / 3] * 3, rtol=1e-6)
    assert np.argmax(probs[0]) == 2
    np.testing.assert_allclose(softmax(logits, axis=0).sum(axis=0), [1, 1, 1], rtol=1e-6)
    assert softmax(np.zeros((0, 3))).shape == (0, 3)


def test_sigmoid_values():
    out = sigmoid([0.0, -1000.0, 1000.0])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.5, 0.0, 1.0], atol=1e-6)


# --- preprocessImage ---------------------------------------------------------------------------------------------


@pytest.fixture
def img():
    return np.random.default_rng(1).integers(0, 256, (60, 120, 3), dtype=np.uint8)  # h=60, w=120


def test_preprocessImage_letterbox_blob_and_info(img):
    blob, info = preprocessImage(img, (64, 64))
    assert blob.shape == (1, 3, 64, 64) and blob.dtype == np.float32
    assert isinstance(info, PreprocessInfo)
    assert info.inputSize == (64, 64) and info.originalSize == (120, 60)
    scale = 64 / 120
    assert info.scale == (scale, scale)
    assert info.pad == (0, (64 - 32) // 2)
    # Padding rows are the padding color (114 / 255), scaled.
    np.testing.assert_allclose(blob[0, :, 0, :], 114 / 255, atol=1e-6)
    assert blob.max() <= 1.0 and blob.min() >= 0.0


def test_preprocessImage_swapRB_mean_scale(img):
    blob, _ = preprocessImage(img, (120, 60), swapRB=True, scale=1.0, mean=(0, 0, 0))
    np.testing.assert_array_equal(blob[0, 0], img[:, :, 2])
    np.testing.assert_array_equal(blob[0, 2], img[:, :, 0])
    blob, _ = preprocessImage(img, (120, 60), swapRB=False, scale=2.0, mean=(10, 20, 30))
    np.testing.assert_allclose(blob[0, 1], (img[:, :, 1].astype(np.float32) - 20) * 2)


def test_preprocessImage_stretch(img):
    blob, info = preprocessImage(img, (30, 30), letterboxPad=False)
    assert blob.shape == (1, 3, 30, 30)
    assert info.scale == (0.25, 0.5) and info.pad == (0, 0)


@pytest.mark.parametrize("shape", [(60, 120), (60, 120, 4), (60, 120, 1)])
def test_preprocessImage_gray_bgra_single_channel(shape):
    src = np.full(shape, 200, np.uint8)
    blob, _ = preprocessImage(src, (32, 32), dtype=np.float16)
    assert blob.shape == (1, 3, 32, 32) and blob.dtype == np.float16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (0, 32)},
        {"size": (32, -1)},
        {"size": (32, 32), "scale": float("nan")},
        {"size": (32, 32), "mean": (1, float("inf"), 3)},
        {"size": (32, 32), "mean": (1, 2)},
    ],
)
def test_preprocessImage_validation(img, kwargs):
    with pytest.raises(ValueError):
        preprocessImage(img, **kwargs)


def test_preprocessImage_empty():
    with pytest.raises(ValueError, match="src"):
        preprocessImage(np.zeros((0, 10, 3), np.uint8), (32, 32))


# --- postprocessDetections ---------------------------------------------------------------------------------------


def _identity_info(width=200, height=100):
    return PreprocessInfo((1.0, 1.0), (0, 0), (width, height), (width, height))


def _rows(*rows):
    # rows of (cx, cy, w, h, score_class0, score_class1, ...)
    return np.array(rows, dtype=np.float32)


def test_postprocessDetections_basic_cxcywh():
    rows = _rows(
        [50, 50, 20, 10, 0.9, 0.1],  # class 0, kept
        [52, 50, 20, 10, 0.8, 0.1],  # overlaps the first, same class -> suppressed
        [150, 50, 20, 10, 0.1, 0.7],  # class 1
        [150, 50, 20, 10, 0.05, 0.1],  # below threshold
    )
    det = postprocessDetections(rows[None], _identity_info(), transpose=False)
    assert isinstance(det, Detections) and det.size == 2
    assert det.boxes.dtype == np.float32 and det.scores.dtype == np.float32 and det.classIds.dtype == np.int32
    np.testing.assert_allclose(det.boxes, [[40, 45, 20, 10], [140, 45, 20, 10]])
    np.testing.assert_allclose(det.scores, [0.9, 0.7])
    np.testing.assert_array_equal(det.classIds, [0, 1])


def test_postprocessDetections_per_class_vs_agnostic():
    rows = _rows([50, 50, 20, 10, 0.9, 0.1], [50, 50, 20, 10, 0.1, 0.8])
    assert postprocessDetections(rows[None], _identity_info(), transpose=False).size == 2
    assert postprocessDetections(rows[None], _identity_info(), transpose=False, classAgnostic=True).size == 1


def test_postprocessDetections_channel_first_layout_auto_and_forced():
    rows = _rows([50, 50, 20, 10, 0.9, 0.1], [150, 50, 20, 10, 0.1, 0.7])
    channel_first = rows.T[None]  # (1, 6, 2)
    forced = postprocessDetections(channel_first, _identity_info(), transpose=True)
    assert forced.size == 2
    # Auto-detect on a realistic size: (1, 6, 300) with 300 candidates.
    many = np.zeros((300, 6), np.float32)
    many[:2] = rows
    auto = postprocessDetections(many.T[None], _identity_info())
    np.testing.assert_allclose(auto.boxes, forced.boxes)
    auto_rows = postprocessDetections(many[None], _identity_info())
    np.testing.assert_allclose(auto_rows.boxes, forced.boxes)


def test_postprocessDetections_objectness_and_formats():
    rows = np.array([[10, 10, 30, 20, 0.5, 0.2, 0.8]], np.float32)  # xyxy with objectness
    det = postprocessDetections(rows[None], _identity_info(), transpose=False, hasObjectness=True, boxFormat="xyxy")
    np.testing.assert_allclose(det.boxes, [[10, 10, 20, 10]])
    np.testing.assert_allclose(det.scores, [0.4])
    assert det.classIds[0] == 1
    det = postprocessDetections(rows[None], _identity_info(), transpose=False, hasObjectness=True, boxFormat="xywh")
    np.testing.assert_allclose(det.boxes, [[10, 10, 30, 20]])


def test_postprocessDetections_normalized_and_letterbox_mapping(img):
    _, info = preprocessImage(img, (64, 64))  # w=120 -> scale 64/120, pad top 16
    # A box covering the whole image inside the letterbox: cx=0.5, cy=0.5, w=1, h=32/64.
    rows = np.array([[0.5, 0.5, 1.0, 0.5, 0.9]], np.float32)
    det = postprocessDetections(rows[None], info, transpose=False, normalized=True)
    np.testing.assert_allclose(det.boxes, [[0, 0, 120, 60]], atol=1e-4)
    # Boxes partly outside the image are clipped to it.
    rows = np.array([[0, 32, 40, 40, 0.9]], np.float32)
    det = postprocessDetections(rows[None], info, transpose=False)
    assert det.boxes[0, 0] == 0 and det.boxes[0, 1] == 0
    assert det.boxes[0, 2] == pytest.approx(20 / (64 / 120))


def test_postprocessDetections_maxCount_and_empty():
    rows = _rows([50, 50, 20, 10, 0.9], [150, 50, 20, 10, 0.8], [100, 80, 20, 10, 0.7])
    assert postprocessDetections(rows[None], _identity_info(), transpose=False, maxCount=2).size == 2
    assert postprocessDetections(rows[None], _identity_info(), transpose=False, scoreThreshold=0.95).size == 0
    det = postprocessDetections(np.zeros((1, 0, 5), np.float32), _identity_info(), transpose=False)
    assert det.size == 0 and det.boxes.shape == (0, 4)


@pytest.mark.parametrize(
    ("output", "kwargs"),
    [
        (np.zeros((1, 2, 4), np.float32), {"transpose": False}),  # no class columns
        (np.zeros((1, 2, 5), np.float32), {"transpose": False, "hasObjectness": True}),
        (np.zeros((2, 2, 6), np.float32), {}),  # batch of 2
        (np.zeros((1, 2, 6), np.float32), {"boxFormat": "ltrb"}),
        (np.zeros((1, 2, 6), np.float32), {"scoreThreshold": float("nan")}),
        (np.zeros((1, 2, 6), np.float32), {"iouThreshold": -1}),
    ],
)
def test_postprocessDetections_validation(output, kwargs):
    with pytest.raises(ValueError):
        postprocessDetections(output, _identity_info(), **kwargs)


# --- postprocessClassification -----------------------------------------------------------------------------------


def test_postprocessClassification_topk():
    logits = np.array([[1.0, 3.0, 2.0, 0.0]], np.float32)
    top = postprocessClassification(logits, topK=2)
    assert [i for i, _ in top] == [1, 2]
    assert sum(p for _, p in postprocessClassification(logits, topK=4)) == pytest.approx(1.0)
    raw = postprocessClassification([0.1, 0.7, 0.2], topK=5, softmax=False)
    assert raw == [(1, pytest.approx(0.7)), (2, pytest.approx(0.2)), (0, pytest.approx(0.1))]
    assert all(isinstance(i, int) and isinstance(p, float) for i, p in raw)


@pytest.mark.parametrize(("output", "topK"), [([1, 2, 3], 0), (np.zeros((2, 3)), 1), ([], 1)])
def test_postprocessClassification_validation(output, topK):
    with pytest.raises(ValueError):
        postprocessClassification(output, topK=topK)


# --- loadModel / inferImage / readClassNames ----------------------------------------------------------------------


def test_loadModel_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="model"):
        loadModel(tmp_path / "없는모델.onnx")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="config"):
        loadModel(model, config=tmp_path / "missing.pbtxt")


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_loadModel_reads_onnx_from_buffer_with_unicode_path(tmp_path, monkeypatch):
    captured = {}

    class FakeNet:
        def setPreferableBackend(self, backend):
            captured["backend"] = backend

        def setPreferableTarget(self, target):
            captured["target"] = target

    def fake_read(buffer):
        captured["buffer"] = bytes(buffer)
        return FakeNet()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", fake_read)
    model = tmp_path / "모델" / "yolo.ONNX"
    model.parent.mkdir()
    model.write_bytes(b"onnx-bytes")
    net = loadModel(model, backend=cv2.dnn.DNN_BACKEND_OPENCV, target=cv2.dnn.DNN_TARGET_CPU)
    assert isinstance(net, FakeNet)
    assert captured == {
        "buffer": b"onnx-bytes",
        "backend": cv2.dnn.DNN_BACKEND_OPENCV,
        "target": cv2.dnn.DNN_TARGET_CPU,
    }


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_loadModel_preferGpu_without_cuda_keeps_default(tmp_path, monkeypatch):
    class FakeNet:
        def setPreferableBackend(self, backend):
            raise AssertionError("backend must not be set without CUDA")

        def setPreferableTarget(self, target):
            raise AssertionError("target must not be set without CUDA")

    monkeypatch.setattr(cv2.dnn, "readNetFromTFLite", lambda buffer: FakeNet())
    monkeypatch.setattr(cv2.dnn, "getAvailableTargets", lambda backend: [])
    model = tmp_path / "m.tflite"
    model.write_bytes(b"x")
    assert isinstance(loadModel(model, preferGpu=True), FakeNet)


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_loadModel_other_formats_use_readNet(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cv2.dnn, "readNet", lambda *args: calls.append(args) or object())
    model, config = tmp_path / "m.weights", tmp_path / "m.cfg"
    model.write_bytes(b"x")
    config.write_text("x")
    loadModel(model)
    loadModel(model, config)
    assert calls == [(str(model),), (str(model), str(config))]


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_loadModel_real_invalid_model_raises_opencv_error(tmp_path):
    model = tmp_path / "bad.onnx"
    model.write_bytes(b"not an onnx file")
    with pytest.raises(cv2.error):
        loadModel(model)


@pytest.mark.skip(reason="needs a real model file; run manually with a YOLO .onnx")
def test_loadModel_real_model():  # pragma: no cover
    pass


class FakeNet:
    def __init__(self, outputs):
        self.outputs = outputs
        self.blob = None
        self.names = None

    def setInput(self, blob):
        self.blob = blob

    def forward(self, outputNames):
        self.names = list(outputNames)
        return self.outputs if len(outputNames) > 1 else self.outputs[0]

    def getUnconnectedOutLayersNames(self):
        return ["out0", "out1"]


def test_inferImage_with_blob_and_image(img):
    net = FakeNet([np.ones((1, 2)), np.zeros((1, 3))])
    blob, _ = preprocessImage(img, (32, 32))
    outputs = inferImage(net, blob)
    assert len(outputs) == 2 and net.names == ["out0", "out1"] and net.blob is blob
    outputs = inferImage(net, img, outputNames=["out0"], size=(32, 32))
    assert len(outputs) == 1 and net.names == ["out0"] and net.blob.shape == (1, 3, 32, 32)
    np.testing.assert_array_equal(outputs[0], np.ones((1, 2)))


def test_inferImage_validation(img):
    net = FakeNet([np.ones((1, 2))])
    with pytest.raises(ValueError, match="size"):
        inferImage(net, img)
    with pytest.raises(ValueError, match="src"):
        inferImage(net, np.zeros(5))


def test_readClassNames(tmp_path):
    path = tmp_path / "클래스.names"
    path.write_text("﻿person\n\n  고양이  \ncar\n", encoding="utf-8")
    assert readClassNames(path) == ["person", "고양이", "car"]
    with pytest.raises(FileNotFoundError):
        readClassNames(tmp_path / "missing.names")


# --- drawDetections ----------------------------------------------------------------------------------------------


def _detections():
    return Detections(
        np.array([[10, 10, 40, 30], [100, 20, 50, 50]], np.float32),
        np.array([0.9, 0.5], np.float32),
        np.array([0, 2], np.int32),
    )


def test_drawDetections_draws_in_place_and_returns_image():
    canvas = np.zeros((100, 200, 3), np.uint8)
    out = drawDetections(canvas, _detections(), classNames=["person", "bike", "car"])
    assert out is canvas and canvas.any()
    # Each box gets its class color (red for class 0 from distinctColors).
    assert tuple(canvas[10, 30]) == (0, 0, 255)


def test_drawDetections_colors_and_missing_names():
    canvas = np.zeros((100, 200, 3), np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    drawDetections(canvas, _detections(), classNames=["only-one"], colors=colors, showScore=False, thickness=1)
    assert tuple(canvas[10, 30]) == (255, 0, 0)
    assert tuple(canvas[20, 120]) == (0, 0, 255)
    gray = np.zeros((100, 200), np.uint8)
    assert drawDetections(gray, _detections(), colors=[200]).any()


def test_drawDetections_empty_and_validation():
    canvas = np.zeros((10, 10, 3), np.uint8)
    empty = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))
    assert drawDetections(canvas, empty) is canvas and not canvas.any()
    with pytest.raises(ValueError, match="src"):
        drawDetections(np.zeros((0, 10, 3), np.uint8), _detections())


# --- remaining loadModel branches --------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DNN, reason="OpenCV build without cv2.dnn")
def test_loadModel_tensorflow_buffers_and_cuda(tmp_path, monkeypatch):
    captured = {}

    class FakeNet:
        def setPreferableBackend(self, backend):
            captured["backend"] = backend

        def setPreferableTarget(self, target):
            captured["target"] = target

    def fake_read(model, config):
        captured["model"], captured["config"] = bytes(model), bytes(config)
        return FakeNet()

    monkeypatch.setattr(cv2.dnn, "readNetFromTensorflow", fake_read)
    monkeypatch.setattr(cv2.dnn, "getAvailableTargets", lambda backend: [cv2.dnn.DNN_TARGET_CUDA])
    model, config = tmp_path / "m.pb", tmp_path / "m.pbtxt"
    model.write_bytes(b"pb")
    config.write_text("cfg")
    loadModel(model, preferGpu=True)
    assert captured == {
        "model": b"pb",
        "config": b"",
        "backend": cv2.dnn.DNN_BACKEND_CUDA,
        "target": cv2.dnn.DNN_TARGET_CUDA,
    }
    loadModel(model, config, preferGpu=True, target=cv2.dnn.DNN_TARGET_CUDA_FP16)
    assert captured["config"] == b"cfg" and captured["target"] == cv2.dnn.DNN_TARGET_CUDA_FP16

    def failing(backend):
        raise cv2.error("no cuda")

    monkeypatch.setattr(cv2.dnn, "getAvailableTargets", failing)
    captured.clear()
    loadModel(model, preferGpu=True)
    assert "backend" not in captured


def test_loadModel_without_dnn_module(tmp_path, monkeypatch):
    monkeypatch.delattr(cv2, "dnn", raising=False)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"cv2.dnn"):
        loadModel(model)
