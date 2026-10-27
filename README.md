# topconverge

Find the **rim** of a football pitch in a fisheye-corrected broadcast camera frame by iterating a network on the
*top view*. This repo holds a dataset of annotated frames, the trained network, and a small standalone server to
browse the frames, annotate them, and let the network place the rim.

## The rim

Each camera lens (L or R) sees one half of the pitch. The rim is the four corners of that half, in image pixels, in
this order:

| point | corner |
|---|---|
| 1 | frontline / centerline (the near end of the halfway line) |
| 2 | frontline / goalline |
| 3 | goalline / sideline (the far corner flag) |
| 4 | sideline / centerline (the far end of the halfway line) |

The *frontline* is the near touchline and the *sideline* is the far one. The sides of the rim are straight lines
of the world. That makes them straight in **undistorted** camera coordinates and curved in the picture. Corners may
lie outside the picture: point 1 usually does, because the halfway line runs out under the camera.

## How the network works

Converging on the rim works like this:

1. Undistort the current four points with the lens model (OpenCV's 14 coefficients).
2. Map the points to a 100 × 100 square with a homography.
3. Render the picture from above through that homography: 1024 × 1024 px over the window −12.5 … 112.5, plus a
   mask channel that marks where no camera pixel lands. You can see this render in the annotator with
   *Network view*.
4. The network (FPN with a ResNet-34 encoder; 4 input channels: RGB + mask) outputs four heat maps. Their
   soft-argmax is where the true corners appear in this view.
5. Map those positions back through the homography and the lens. Take half a step (`alpha` 0.5) towards them.

Repeat until no corner moves more than 0.5 px, or more than 0.1 square units; Auto stops after 60 steps. At the fixed
point the rim *is* the square.

A right lens's rim turns the other way, so its view comes out mirrored and point 1 is always at the bottom right
for the network. Annotation and Auto start from the middle of each side of the picture: 1 bottom, 2 left, 3 top,
4 right, mirrored for R.

The weights (`weights/rim_iter4_sides.pt`, stored as float16) are model `iter4-sides`. It was trained for
16 000 steps (batch 3) on the starred rims that existed then, with recordings held out by
`crc32(recording uuid) % 5 == 0`.

## Dataset (`data/`)

- `data/images/<id>.jpg` holds the full 3840 × 2160 frame.
- `data/labels/<id>.json` holds its label.
- `<id>` is `<recording>_<view>_<frame number>`, for example `r017_L_0186472`. Recordings are numbered
  `r001 …`; the numbering is stable but anonymous.

```json
{
 "image": "images/r001_L_0129123.jpg",
 "width": 3840, "height": 2160,
 "recording": "r001",
 "view": "L",
 "lens": {"fx": 1928.1, "fy": 1929.9, "cx": 1802.4, "cy": 962.5, "dist": [14 OpenCV coefficients]},
 "rim": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
 "quality": 3,
 "split": "train",
 "auto_used": false,
 "source": "annotated"
}
```

| field | meaning |
|---|---|
| `rim` | The four points in image px, or `null` when the frame has no rim yet. |
| `quality` | `1`, `2` or `3` stars (3 is best). `"unusable"` means the frame shows no usable half pitch; `"needs_work"` means not finished. `null` means nobody has looked at it. The model was trained on the starred frames only. |
| `split` | See below. |
| `auto_used` | Whether the network's Auto / Step was used while the rim was annotated. |
| `source` | `annotated`, or `unannotated` for the 100 extra frames. |
| `edited` | When the frame was last saved by this server. Only present on frames saved here. |

`split` says what the model saw:

| value | meaning |
|---|---|
| `heldout` | The recording was held out of training. These are the honest test frames. |
| `train` | The annotation existed when training ended. A few came in during the run. |
| `after` | First saved after training ended. The model never saw these either. |
| `unannotated` | One of the extra frames, from recordings with no rim annotation at all. |

Contents:

- 270 annotated frames: 107 ★★★, 93 ★★, 52 ★, 14 unusable, 4 needs work.
- 100 unannotated frames, each from a different recording.
- 135 L and 135 R among the annotated frames, from 258 recordings.

## Running the server

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pick the torch build for your CUDA from pytorch.org, or use the CPU one
python server.py                       # http://127.0.0.1:8000/
python server.py --host 0.0.0.0 --port 8000 --device cpu
```

`--device auto` (the default) uses CUDA when it is available. A step takes about 15 ms on an RTX 3080 and a
few seconds on a CPU (2.7 s with 8 threads), and Auto usually needs 15–25 steps.

**The frame list** shows every frame with its rim. Filter by state, split or lens; click a frame to open it.

**The annotator:**

- Drag a point to move it. Drag anywhere else to pan. The wheel zooms about the pointer; two fingers pinch.
- **A** runs Auto (any click stops it). **Shift+A** takes one step. **R** resets the points to the start.
- **Ctrl+Z / Ctrl+Y** undo and redo.
- **0 / X / 1 / 2 / 3** set the state: needs work, unusable, or stars.
- **S** saves. **← / →** go to the previous or next frame of the list. **F** fits the picture to the window.
- **V** shows the exact network input for the current rim, with the corners it predicts.
- **H** shows or hides the help.

Saving rewrites `data/labels/<id>.json`, so your edits show up as a git diff. Untouched start points of an
unannotated frame are not stored as a rim.

## Evaluation

```sh
python -m topconverge.evaluate --split heldout            # starred held-out frames, Auto from the start, error per corner in px
python -m topconverge.evaluate --split after --json after.json
```

On the 46 starred held-out frames (60 half steps from the start; numbers from the shipped float16 weights on an RTX 3080):

| corner | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| median error, px | 27.5 | 3.1 | 2.2 | 1.3 |

- **Corners 2–4:** within 5 px on 39 % of the frames, within 10 px on 54 %.
- **All four corners:** within 10 px on 11 %, within 25 px on 33 %.
- **Settling:** the rim settled on 74 % of the frames, after a median of 15 steps.

Corner 1 usually lies far outside the picture (under the camera). There a pixel covers a long way on the ground, so
both the label and the network are loose. The failures are mostly frames where corner 4 or corner 1 runs off,
and those frames do not settle.

Compared with the original float32 service, the first step differs by 0.01–0.8 px, which comes from the float16
weights. Frames that settle end within a few pixels of it; frames that do not settle can wander further apart.

## Files

```
server.py                 the web server (Flask; waitress if installed)
topconverge/rimnet.py     lens, homography, top-view render, network, step / iterate
topconverge/evaluate.py   Auto from the start vs the annotated rims
static/                   the frame list and the annotator (plain ES modules, no build step)
weights/                  config.json + the float16 weights
data/                     images/ and labels/
```
