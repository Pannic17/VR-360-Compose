"""Frame interpolation: a 30 fps render delivered at 60 fps (ROADMAP P16).

A 60 fps delivery removes the trailing on moving content in the headset, and rendering it
doubles the render, the tiles and the stitch. This module is the other way to get there:
every stitched frame is kept as it is and one synthesised frame goes between each pair, so
the delivery has twice the frames and lasts exactly as long. In the blind headset review
that closed P16 (2026-11-10) the interpolated 60 fps clip was judged equal to the rendered
one, and both far better than the 30 fps original.

What the measurements decided (`tools/interp_probe.py`, ROADMAP P16 step 1):

* **The interpolator is an external tool, like ffmpeg.** rife-ncnn-vulkan with its
  `rife-v4.6` model, run on a directory of PNGs. Inference is under half a second an 8K
  frame; the rest of its 1.1 s a pair is its own PNG reading and writing. It is found
  beside the program, never downloaded, and a job that asks for interpolation without it
  warns and delivers at the capture's own rate (the GPU rule: warn, do not block).
* **On the stitched equirect, not on the tiles.** Fifteen views interpolated on their own
  scored the same against the truth, took 4.5 times as long and no longer agreed with each
  other in the overlaps.
* **Padded.** The tool sees a flat picture, so the frame is continued past its edges the
  way the sphere continues: the left and right edges into each other (that halved the
  error at the 0/360 seam), and a few rows past each pole as the far side of the pole
  (no measurable effect, but it is what the reviewed clip was made with).
* **Kept and synthesised frames take the same road.** When the master is larger than the
  delivery it is resampled here, once, before the interpolator sees it, and the encoder
  is fed delivery-size frames for both kinds -- otherwise the two would differ by a
  resampling step and the difference would alternate at 30 Hz.
* **Its exit code is not believed.** Short of video memory the tool does not fail: with
  8 GiB of the card held by another process it returned 86 of 89 frames that were other
  pictures entirely, and exited with 0. So the card's free memory is read before the job
  starts, one pair is tried at the job's size, and every frame that comes back is held
  against the two it was made from before it is delivered (:func:`not_between`).
* **What it cannot do.** Content that moves too far between two frames -- the sea floor
  directly under a low camera, an object brushing past it -- comes out smeared whatever
  interpolates it. That is in the delivery, and the review accepted it.

The interpolator runs on a GPU through Vulkan and, with memory to spare, repeats itself on
one machine: two separate runs handed the encoder 180 identical frames. Nothing here
promises the same bytes on another machine: an interpolated delivery is outside the
CPU/GPU byte-identity the stitch keeps.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose import encode

U8 = npt.NDArray[np.uint8]

__all__ = [
    "BATCH",
    "CUT_LEVELS",
    "DEFAULT_INTERPOLATION",
    "INTERPOLATIONS",
    "InterpolationError",
    "Interpolator",
    "Rife",
    "Stopped",
    "collect",
    "crop_equirect",
    "find",
    "frame_difference",
    "free_video_memory",
    "not_between",
    "pad_equirect",
    "padding",
    "prepare",
    "resample",
    "stage",
    "video_memory_needed",
]

INTERPOLATIONS = ("off", "2x")
DEFAULT_INTERPOLATION = "off"
"""Off: a delivery is the frames that were rendered unless someone asks otherwise."""

MODEL = "rife-v4.6"
"""The model the measurements and the headset review were made with."""
TOOL_DIRECTORY = "rife-ncnn-vulkan"
"""The folder looked for beside the program, holding the executable and the model."""
TOOL_ENVIRONMENT = "VR_COMPOSE_RIFE"
"""Or this variable, naming that folder wherever it is."""
TOOL_JOBS = "2:2:4"
"""Its load:process:save threads. Its PNG encoder is the slow part: at 8K the default
`1:2:2` took 1.75-1.9 s a pair and this 1.11 s."""

BATCH = 30
"""Stitched frames handed to the tool per run.

One run is one process start (a few seconds of Vulkan and model set-up) and holds its
inputs and outputs on disk and the kept frames in memory, so the batch trades start-up per
frame against both: 30 frames at 8K is 2.7 GB of frames in memory and about 5 GB of
temporary PNGs, and puts the start-up near 0.1 s a frame."""

CUT_LEVELS = 45.0
"""Mean absolute difference between two stitched frames above which they are a cut.

Nothing sensible lies halfway between two shots, so across a cut the earlier frame is
repeated instead. Measured with :func:`frame_difference` on the 20 s of the review
sequence (ROADMAP P16 step 3): consecutive frames differ by 19.7 levels on average and
26.6 at most -- the camera skims the sea floor, so every frame moves a lot -- while frames
five seconds apart, unrelated pictures of the same scene, differ by 33 to 83 and 45 on
average. The two ranges nearly touch, so no threshold separates fast motion from a cut
inside one scene, and this one **errs towards interpolating**: a frame repeated in the
middle of fast motion is a visible stutter at the worst moment, a frame blended across a
cut is one sixtieth of a second. At 45 it never fires on motion (the fastest step of the
whole sequence scales to about 30) and catches a change of scene or of brightness; a cut
between two similar views goes through interpolated. It has not met a real cut: no
sequence with one was available."""


class InterpolationError(RuntimeError):
    """The interpolator is missing something or failed; the message says which."""


class Stopped(Exception):
    """The caller asked to stop while the tool was running; it has been killed.

    A request honoured rather than a failure, which is why it is not an `...Error`."""


def padding(width: int) -> tuple[int, int]:
    """`(wrap columns, pole rows)` for a frame this wide: 12 and 6 degrees, 256 and 128 at 8K."""
    wrap = max(8, round(width / 30 / 8) * 8)
    return wrap, wrap // 2


def pad_equirect(image: U8, wrap: int, pole: int) -> U8:
    """Continue an equirect past its edges the way the sphere does.

    Past the left edge is the right edge. Past a pole is the same meridian seen from the
    other side: the rows next to the pole, upside down and half a turn away.
    """
    width = image.shape[1]
    if pole:
        above = np.roll(image[:pole][::-1], width // 2, axis=1)
        below = np.roll(image[-pole:][::-1], width // 2, axis=1)
        image = np.concatenate([above, image, below], axis=0)
    if wrap:
        image = np.concatenate([image[:, -wrap:], image, image[:, :wrap]], axis=1)
    return image


def crop_equirect(image: U8, wrap: int, pole: int) -> U8:
    height, width = image.shape[0] - 2 * pole, image.shape[1] - 2 * wrap
    return image[pole : pole + height, wrap : wrap + width]


def resample(frame: U8, width: int, height: int) -> U8:
    """A master brought to the delivery size, or the frame itself when it already is.

    Lanczos, as the encoder's own downsample is. Done here because the interpolator works
    at the delivery size: at the master's it would need the video memory the warp plan is
    already using.
    """
    if frame.shape[:2] == (height, width):
        return frame
    resized = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def stage(frame: U8, path: pathlib.Path) -> None:
    """Write one delivery-size frame, padded, where the tool will read it.

    Three channels on purpose: the tool does not refuse a four-channel PNG, it writes grey
    noise the size of the input.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"a staged frame is height x width x 3, got {frame.shape}")
    padded = pad_equirect(frame, *padding(frame.shape[1]))
    Image.fromarray(padded).save(path, format="PNG", compress_level=1)


def prepare(frame: U8, path: pathlib.Path, width: int, height: int) -> U8:
    """:func:`resample` then :func:`stage`, returning the delivery-size frame.

    One call so a worker thread can take a stitched frame off the stitch loop's hands:
    the loop gets the next frame started while this one is brought down and written.
    """
    delivery = resample(frame, width, height)
    stage(delivery, path)
    return delivery


def collect(path: pathlib.Path, width: int, height: int) -> U8:
    """Read a synthesised frame back and take the padding off again."""
    with Image.open(path) as image:
        padded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    wrap, pole = padding(width)
    if padded.shape != (height + 2 * pole, width + 2 * wrap, 3):
        raise InterpolationError(
            f"{path.name} is {padded.shape[1]}x{padded.shape[0]}, not the padded "
            f"{width + 2 * wrap}x{height + 2 * pole} that was handed in"
        )
    return np.ascontiguousarray(crop_equirect(padded, wrap, pole))


def frame_difference(a: U8, b: U8) -> float:
    """How far apart two frames are, in levels: the mean absolute difference of every
    eighth pixel in both directions. Cheap, and all :data:`CUT_LEVELS` needs."""
    step = 8
    coarse_a = a[::step, ::step].astype(np.int16)
    coarse_b = b[::step, ::step].astype(np.int16)
    return float(np.abs(coarse_a - coarse_b).mean())


BETWEEN_MARGIN = 10.0
"""How much further from a neighbour than the neighbours are from each other a synthesised
frame may be before it is not a frame between them (levels, :func:`frame_difference`).

A real in-between frame is nearer to each neighbour than they are to each other, so the
margin only has to cover a still scene, where all three differences are noise. What it is
there to catch is not subtle: with the card's memory taken by another program the tool
returned 86 of 89 frames that were other pictures entirely -- 85 levels from the frames
they should have been -- and exited with 0 (ROADMAP P16 step 3)."""


def not_between(before: U8, made: U8, after: U8) -> str | None:
    """Why `made` is not a frame between `before` and `after`, or None when it is.

    The interpolator's exit code proves nothing (see :data:`BETWEEN_MARGIN`), so every
    frame it returns is held against the two it was made from before it is delivered.
    """
    apart = frame_difference(before, after)
    from_before = frame_difference(before, made)
    from_after = frame_difference(made, after)
    if max(from_before, from_after) <= apart + BETWEEN_MARGIN:
        return None
    return (
        f"it is {from_before:.0f} and {from_after:.0f} levels from the frames either side "
        f"of it, which are {apart:.0f} apart"
    )


MEMORY_BYTES_PER_PIXEL = 220
"""Video memory the tool takes per pixel of the padded frame it is given.

Measured at 8K: 6.5 to 7.0 GB on top of what the card already held for 8192x4096, which
is 195 to 210 bytes a pixel. A little above that, so a card that just fits is not refused."""


def video_memory_needed(width: int, height: int) -> int:
    """Bytes of video memory interpolating a `width` x `height` delivery takes."""
    wrap, pole = padding(width)
    return (width + 2 * wrap) * (height + 2 * pole) * MEMORY_BYTES_PER_PIXEL


def free_video_memory() -> int | None:
    """Bytes free on the first NVIDIA card right now, or None when that cannot be read.

    `nvidia-smi` rather than cupy on purpose: the question is what the *card* has left
    for another process, whatever this one's warp runs on, and it has to work in a build
    without cupy. None -- no NVIDIA card, no nvidia-smi -- means no opinion, never "none".
    """
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            creationflags=encode.child_creation_flags(),
        ).stdout
        return int(listing.strip().splitlines()[0]) * 2**20
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


OUT_OF_MEMORY = (
    "rife-ncnn-vulkan writes such frames, without reporting an error, when the video card "
    "has too little free memory: close other programs that use the GPU"
)
"""What a frame that fails :func:`not_between` has meant every time it was seen."""


class Interpolator(Protocol):
    """What the pipeline needs: the frame halfway between each consecutive pair."""

    @property
    def describe(self) -> str: ...

    def synthesise(
        self,
        staged: pathlib.Path,
        count: int,
        out: pathlib.Path,
        *,
        stop: Callable[[], bool],
    ) -> list[pathlib.Path]:
        """`staged` holds `000000.png` .. `count - 1`, all one size. Returns `count - 1`
        PNGs of that size in `out`: the frame between inputs `i` and `i + 1`, in order.
        Raises :class:`Stopped` if `stop()` turns true while it works."""
        ...


def _executable() -> str:
    return "rife-ncnn-vulkan.exe" if sys.platform == "win32" else "rife-ncnn-vulkan"


@dataclasses.dataclass(frozen=True, slots=True)
class Rife:
    """rife-ncnn-vulkan in `directory`, with its model beside the executable."""

    directory: pathlib.Path
    model: str = MODEL
    jobs: str = TOOL_JOBS

    @property
    def describe(self) -> str:
        return f"rife-ncnn-vulkan {self.model} ({self.directory})"

    def synthesise(
        self,
        staged: pathlib.Path,
        count: int,
        out: pathlib.Path,
        *,
        stop: Callable[[], bool],
    ) -> list[pathlib.Path]:
        # For N inputs the tool writes 2 N files numbered from 1: file 2 i + 1 is input i
        # again, file 2 i + 2 the frame halfway to input i + 1, the last two the last input.
        out.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.directory / _executable()),
            "-i", str(staged), "-o", str(out), "-m", str(self.directory / self.model),
            "-j", self.jobs,
        ]  # fmt: skip
        process = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=encode.child_creation_flags(),
        )
        # Its stderr is a few lines of device listing; draining it only at the end cannot
        # fill a pipe. Polling rather than waiting is what lets a cancel land in a fifth
        # of a second instead of at the end of a half-minute batch.
        while True:
            try:
                process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if stop():
                    process.kill()
                    process.wait()
                    raise Stopped from None
        errors = process.stderr.read().decode(errors="replace") if process.stderr else ""
        made = [out / f"{2 * i + 2:08d}.png" for i in range(count - 1)]
        missing = [path.name for path in made if not path.is_file()]
        if process.returncode != 0 or missing:
            tail = "\n".join(errors.strip().splitlines()[-6:])
            raise InterpolationError(
                f"rife-ncnn-vulkan exited with {process.returncode} and left "
                f"{len(missing)} of {len(made)} frame(s) unwritten:\n{tail}"
            )
        return made

    def check(self) -> None:
        """Run the tool once on two small frames. Raises :class:`InterpolationError`.

        Finding the executable proves nothing about the machine: no Vulkan device, a
        missing runtime library or a model the build cannot load all fail here, in two
        seconds, rather than after the first batch has been stitched.
        """
        with tempfile.TemporaryDirectory(prefix="vr-compose-rife-") as scratch:
            staged = pathlib.Path(scratch, "in")
            staged.mkdir()
            ramp = np.linspace(0, 255, 64, dtype=np.uint8)
            for index in range(2):
                frame = np.stack([np.roll(ramp, 4 * index)] * 64)[..., None].repeat(3, axis=2)
                Image.fromarray(frame).save(staged / f"{index:06d}.png", format="PNG")
            try:
                (made,) = self.synthesise(staged, 2, pathlib.Path(scratch, "out"), stop=bool)
            except OSError as error:
                raise InterpolationError(
                    f"rife-ncnn-vulkan could not be started: {error}"
                ) from None
            with Image.open(made) as image:
                if image.size != (64, 64):
                    raise InterpolationError(f"the check frame came back as {image.size}")

    def rehearse(self, width: int, height: int) -> None:
        """One pair at the job's own size, judged like every frame of the job will be.

        :meth:`check` passes on a card with no memory to spare, because two 64-pixel
        frames need none. This runs after the warp plan is on the card, at the size the
        job will use, and holds the result against its neighbours: a card that cannot
        fit the job is found here, before anything is stitched, instead of in a delivery
        whose every other frame is noise.
        """
        with tempfile.TemporaryDirectory(prefix="vr-compose-rife-") as scratch:
            staged = pathlib.Path(scratch, "in")
            staged.mkdir()
            # A smooth picture that drifts: what lies between two such frames is known
            # well enough to tell a frame from noise, and costs nothing to make.
            ramp = np.linspace(0.0, 4.0 * np.pi, width, dtype=np.float32)
            rows = np.linspace(0.0, np.pi, height, dtype=np.float32)[:, None]
            base = 128.0 + 60.0 * np.sin(ramp)[None, :] * np.sin(rows)
            first = np.repeat(base.astype(np.uint8)[..., None], 3, axis=2)
            second = np.roll(first, max(4, width // 256), axis=1)
            for index, frame in enumerate((first, second)):
                stage(frame, staged / f"{index:06d}.png")
            (made,) = self.synthesise(staged, 2, pathlib.Path(scratch, "out"), stop=bool)
            wrong = not_between(first, collect(made, width, height), second)
        if wrong is not None:
            raise InterpolationError(
                f"a trial frame at {width}x{height} came back wrong ({wrong}). {OUT_OF_MEMORY}"
            )


def find(explicit: pathlib.Path | None = None) -> Rife | None:
    """The tool, or None. `explicit`, then :data:`TOOL_ENVIRONMENT`, then beside the program.

    "Beside the program" is the directories ffmpeg is looked for in (:mod:`vr_compose.encode`),
    each with a :data:`TOOL_DIRECTORY` folder in it. A folder counts only when it holds
    both the executable and the model.
    """
    candidates: list[pathlib.Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if named := os.environ.get(TOOL_ENVIRONMENT):
        candidates.append(pathlib.Path(named))
    candidates += [directory / TOOL_DIRECTORY for directory in encode._search_dirs()]
    for directory in candidates:
        if (directory / _executable()).is_file() and (directory / MODEL).is_dir():
            return Rife(directory.resolve())
    return None


MISSING = (
    f"rife-ncnn-vulkan was not found: put its folder beside the program as "
    f"`{TOOL_DIRECTORY}` (the executable and its `{MODEL}` folder inside), or point "
    f"{TOOL_ENVIRONMENT} at it"
)
"""Why :func:`find` returned None, in the words a warning uses."""
