"""Hallucination guards of the semantic stage, with a scripted fake VLM (no model download)."""

import numpy as np

from src.semantics import label_one, parse_label

CHANGE = {
    "id": 0,
    "action": "removed",
    "box": {"center": [0.1, 0.2, 0.3], "extents": [0.4, 0.4, 0.6], "yaw_rad": 0.0, "corners": []},
}
IMGS = [np.zeros((8, 8, 3), np.uint8)] * 2


class FakeVLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, images, prompt):
        self.calls += 1
        # no measurements may reach the model (image numbering "image 1/2" is fine)
        import re

        assert not re.search(r"\d+\.\d+", prompt), "prompt must not contain measurements"
        return self.replies.pop(0)


def test_parse_accepts_fenced_json_and_rejects_bad_schema():
    ok = parse_label(
        '```json\n{"object": "chair", "action": "removed", "confidence": 0.7, "spatial_summary": "by the table"}\n```'
    )
    assert ok is not None and ok.object == "chair"
    assert (
        parse_label('{"object": "chair", "action": "flew", "confidence": 0.7, "spatial_summary": "x"}')
        is None
    )
    assert (
        parse_label('{"object": "chair", "action": "removed", "confidence": 3, "spatial_summary": "x"}')
        is None
    )
    assert parse_label("no json here") is None


def test_retries_then_label_unavailable():
    vlm = FakeVLM(["garbage", "{not json}", "still bad"])
    res = label_one(vlm, IMGS, CHANGE, "m", max_retries=2)
    assert vlm.calls == 3
    assert res["object"] == "label_unavailable" and res["confidence"] == 0.0


def test_second_try_succeeds():
    vlm = FakeVLM(
        [
            "oops",
            '{"object": "stool", "action": "removed", "confidence": 0.6, "spatial_summary": "near the wall"}',
        ]
    )
    res = label_one(vlm, IMGS, CHANGE, "m", max_retries=2)
    assert vlm.calls == 2 and res["object"] == "stool" and res["spatial_summary"] == "near the wall"


def test_numbers_from_model_are_discarded_and_geometry_numbers_used():
    vlm = FakeVLM(
        [
            '{"object": "stool", "action": "removed", "confidence": 0.6, '
            '"spatial_summary": "it moved 3 m to the left"}'
        ]
    )
    res = label_one(vlm, IMGS, CHANGE, "m", max_retries=0)
    assert res["spatial_summary"] == ""
    assert "x=0.10" in res["description"] and "0.40 x 0.40 x 0.60" in res["description"]


def test_model_action_disagreement_is_recorded_but_geometry_wins():
    vlm = FakeVLM(['{"object": "stool", "action": "added", "confidence": 0.6, "spatial_summary": "corner"}'])
    res = label_one(vlm, IMGS, CHANGE, "m", max_retries=0)
    assert res["vlm_action_agreed"] is False
    assert "removed" in res["description"]


def test_retry_prompt_carries_the_validation_error():
    # Real-pair failure: greedy decoding repeated "action": "placed" for identical prompts.
    prompts = []

    class Recorder(FakeVLM):
        def generate(self, images, prompt):
            prompts.append(prompt)
            return super().generate(images, prompt)

    vlm = Recorder(
        [
            '{"object": "cup", "action": "placed", "confidence": 0.8, "spatial_summary": "on the bed"}',
            '{"object": "cup", "action": "removed", "confidence": 0.8, "spatial_summary": "on the bed"}',
        ]
    )
    res = label_one(vlm, IMGS, CHANGE, "m", max_retries=2)
    assert vlm.calls == 2 and res["object"] == "cup"
    assert "invalid" not in prompts[0]
    assert "previous answer was invalid" in prompts[1] and "action" in prompts[1]


def test_label_verdict_flags_empty_change_and_contradicted_label():
    from src.semantics import label_verdict

    names = ["cup", "bottle", "chair"]
    # object view gains on the VLM label -> supported, best name reported first
    v = label_verdict(np.array([0.12, 0.004, 0.0]), names, "cup", 0.01)
    assert v["verdict"] == "label_supported" and v["vlm_rank"] == 1 and v["top"][0][0] == "cup"
    # VLM label fits the empty view better, while another name does gain -> contradicted
    v = label_verdict(np.array([-0.55, 0.13, 0.08]), ["wall", "couch", "bench"], "wall", 0.01)
    assert v["verdict"] == "label_contradicted" and v["vlm_rank"] == 3
    # nothing gains -> no visible object, whatever the VLM said
    v = label_verdict(np.array([0.0005, 0.005, 0.002]), ["desk", "sink", "bench"], "desk", 0.01)
    assert v["verdict"] == "no_visible_object"
    # no VLM label at all: still judged on the COCO names
    v = label_verdict(np.array([0.3, 0.0]), ["chair", "cup"], None, 0.01)
    assert v["verdict"] == "label_supported" and "vlm_rank" not in v
