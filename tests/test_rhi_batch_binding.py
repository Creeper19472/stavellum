"""Batched native ownership must remain correct when Python wrapping fails."""

from __future__ import annotations

import ctypes
import gc
import threading
from collections import OrderedDict

import pytest

from stavellum.rendering._rhi import Quad, RhiTarget, _OwnedFrame
from stavellum.rendering.gpu import GpuBackendError


class NativeBatch:
    def __init__(self, *, corrupt=None, fail=False):
        self.storage = {}
        self.released = []
        self.removed = []
        self.command_counts = []
        self.corrupt, self.fail = corrupt, fail

    def sprhi_submit_batch_owned(self, handle, items, count, outputs):
        assert handle == 1
        self.command_counts = [items[index].count for index in range(count)]
        for index in range(count):
            for offset in range(items[index].count):
                assert items[index].quads[offset].texture_id == 0
            owner = index + 101
            pixels = (ctypes.c_uint8 * 24)(*([index + 1] * 24))
            self.storage[owner] = pixels
            outputs[index] = _OwnedFrame(owner, ctypes.addressof(pixels),
                                         23 if self.corrupt == index else 24)
        return int(self.fail)

    def sprhi_release_frame(self, owner):
        assert owner in self.storage, "owned frame released twice"
        del self.storage[owner]
        self.released.append(owner)

    def sprhi_remove(self, handle, key):
        self.removed.append(key)
        return 0

    def sprhi_last_error(self, handle):
        return b"batch failed"


def target_with(dll):
    target = RhiTarget.__new__(RhiTarget)
    target.width, target.height = 3, 2
    target._thread = threading.get_ident()
    target._handle = 1
    target._dll = dll
    target._copy_readback = False
    target._textures = OrderedDict(((11, 24), (22, 24)))
    target._bytes, target.budget = 48, 24
    target.texture_cache_evictions = 0
    target.command_pack_seconds = 0.0
    return target


def quad():
    return Quad(0, 0, 0, 3, 2, 0, 0, 1, 1, 1, 1, 1, 1)


def test_batch_wraps_frames_in_order_without_copying_and_evicts_after_completion():
    dll = NativeBatch()
    target = target_with(dll)
    images = target.render_batch([[quad()], [], [quad(), quad()]])
    assert dll.command_counts == [1, 0, 2]
    assert dll.removed == [11] and target._bytes == 24
    assert [bytes(image.constBits())[0] for image in images] == [1, 2, 3]
    dll.storage[102][0] = 201
    assert bytes(images[1].constBits())[0] == 201
    assert dll.released == []
    del target, images
    gc.collect()
    assert sorted(dll.released) == [101, 102, 103] and not dll.storage


@pytest.mark.parametrize("index", [0, 1, 2])
def test_invalid_output_frees_wrapped_and_remaining_frames(index):
    dll = NativeBatch(corrupt=index)
    target = target_with(dll)
    with pytest.raises(GpuBackendError, match="Invalid owned RHI frame"):
        target.render_batch([[], [], []])
    gc.collect()
    assert sorted(dll.released) == [101, 102, 103] and not dll.storage
    assert dll.removed == []


def test_native_failure_releases_any_partially_returned_owners():
    dll = NativeBatch(fail=True)
    with pytest.raises(GpuBackendError, match="batch failed"):
        target_with(dll).render_batch([[], []])
    assert dll.released == [101, 102] and not dll.storage
    assert dll.removed == []


@pytest.mark.parametrize("count", [0, 9])
def test_batch_size_rejected_before_native_call(count):
    dll = NativeBatch()
    with pytest.raises(ValueError, match="between one and eight"):
        target_with(dll).render_batch([[] for _ in range(count)])
    assert dll.command_counts == [] and dll.storage == {}


def test_copy_diagnostic_preserves_single_frame_copy_path(monkeypatch):
    target = target_with(NativeBatch())
    target._copy_readback = True
    seen = []

    def render(commands, *, evict):
        assert evict is False
        assert dll.removed == []
        seen.append(commands)
        return len(commands)

    dll = target._dll
    monkeypatch.setattr(target, "render", render)
    assert target.render_batch([[], [quad()]]) == [0, 1]
    assert [len(commands) for commands in seen] == [0, 1]
    assert dll.removed == [11]
