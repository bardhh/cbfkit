"""Compose measured warehouse camera frames; never synthesize robot motion.

Requires Pillow, NumPy, Matplotlib (fonts), and ffmpeg. The optional grid is
another actual recording with --all_frames, not copies of the hero trajectory.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LayoutDraw:
    """Draw layout coordinates at output resolution; fonts are already scaled."""

    def __init__(self, image, scale):
        self.draw = ImageDraw.Draw(image)
        self.scale = scale

    def xy(self, coordinates):
        return tuple(round(x * self.scale) for x in coordinates)

    def text(self, xy, *args, **kwargs):
        self.draw.text(self.xy(xy), *args, **kwargs)

    def line(self, xy, *, width=1, **kwargs):
        self.draw.line(self.xy(xy), width=width * self.scale, **kwargs)

    def rounded_rectangle(self, xy, *, radius=0, **kwargs):
        self.draw.rounded_rectangle(self.xy(xy), radius=radius * self.scale, **kwargs)


def load_run(path, camera_ids=(0,)):
    result = json.loads((path / "result.json").read_text())
    rows = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    expected_steps = list(range(5, result["steps"] + 1, 5))
    if result["steps"] % 5:
        expected_steps.append(result["steps"])
    if not expected_steps or [row["step"] for row in rows] != expected_steps:
        raise ValueError(f"Incomplete trajectory: {path}")
    count = result["steps"] // 5 + 1
    if not result["rendered"] or result["frame_count"] != count:
        raise ValueError(f"Incomplete recording: {path}")
    for camera_id in camera_ids:
        files = sorted((path / "frames").glob(f"env-{camera_id:02d}-*.png"))
        if [p.name for p in files] != [f"env-{camera_id:02d}-{i:04d}.png" for i in range(count)]:
            raise ValueError(f"Missing frames: {path}, camera {camera_id}")
        hashes = set()
        for p in files:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"))
            if rgb.shape != (result["height"], result["width"], 3) or rgb.std() < 1:
                raise ValueError(f"Invalid frame: {p}")
            hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
        if len(hashes) < 2:
            raise ValueError(f"Frozen recording: {path}")
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True, help="MP4 output")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--grid", type=Path, help="One recorded run with all 16 cameras")
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Layout scale: 1=1280x600, 2=2560x1200, 3=3840x1800; requires native camera detail",
    )
    args = parser.parse_args()
    scale = args.scale
    paths = {m: args.directory / f"seed-{args.seed}-{m}" for m in ("nominal", "filtered")}
    runs = {m: load_run(p) for m, p in paths.items()}
    cfg = runs["nominal"][0]
    for result, _ in runs.values():
        if result["width"] < 640 * scale or result["height"] < 360 * scale:
            raise ValueError(f"Rerender hero cameras at least {640 * scale}x{360 * scale}")
    for key in (
        "steps",
        "seed",
        "num_envs",
        "dt_s",
        "scenario_parameters",
        "initial_xy",
        "policy_path",
        "camera_eye",
        "camera_lookat",
    ):
        if cfg[key] != runs["filtered"][0][key]:
            raise ValueError(f"Unpaired recordings: {key}")
    if not runs["filtered"][0]["successful"][0]:
        raise ValueError(
            "Hero scenario did not complete without contact/fall; publish a failure analysis instead"
        )
    if args.grid:
        grid_cfg, _ = load_run(args.grid, range(16))
        if grid_cfg["num_envs"] != 16:
            raise ValueError("Grid requires 16 independently simulated environments")
        if grid_cfg["width"] < 220 * scale or grid_cfg["height"] < 124 * scale:
            raise ValueError(
                "Rerender grid cameras at a resolution sufficient for each output tile"
            )
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"

    def regular(size):
        return ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), size * scale)

    def bold(size):
        return ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), size * scale)

    bg, muted, white, teal, amber, red = (
        "#0E171F",
        "#AABBC7",
        "#F6F7F8",
        "#65E3BA",
        "#F0B765",
        "#F17E76",
    )
    width, height = 1280 * scale, 600 * scale
    count = cfg["frame_count"]
    rendered = []

    def project(points, result):
        eye = np.array(result["camera_eye"])
        forward = np.array(result["camera_lookat"]) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        delta = np.array(points) - eye
        depth = delta @ forward
        k = np.array(
            result.get("camera_intrinsics")
            or [
                [result["width"] * 24 / 25, 0, result["width"] / 2],
                [0, result["width"] * 24 / 25, result["height"] / 2],
                [0, 0, 1],
            ]
        )
        return np.column_stack(
            (k[0, 2] + k[0, 0] * (delta @ right) / depth, k[1, 2] - k[1, 1] * (delta @ up) / depth)
        )

    def decorated(mode, index, row):
        result = runs[mode][0]
        with Image.open(paths[mode] / "frames" / f"env-00-{index:04d}.png") as original:
            frame = original.convert("RGBA")
        layer = Image.new("RGBA", frame.size)
        d = ImageDraw.Draw(layer)
        camera_scale = result["width"] / 800
        xy = np.array(row["xy"][0])
        active = row["intervened"][0]
        angles = np.linspace(0, 2 * np.pi, 65)
        circle = np.column_stack(
            (xy[0] + 0.78 * np.cos(angles), xy[1] + 0.78 * np.sin(angles), np.full(65, 0.03))
        )
        polygon = [tuple(p) for p in project(circle, result)]
        d.line(
            polygon,
            fill=(101, 227, 186, 230) if active and mode == "filtered" else (220, 230, 237, 100),
            width=max(1, round(3 * camera_scale)),
        )
        for command, color, thickness in (
            (row["nominal"][0], (245, 245, 250, 180), 3),
            (row["applied"][0], (101, 227, 186, 255), 5),
        ):
            if mode == "nominal" and thickness == 5:
                continue
            u = np.array(command)
            if np.linalg.norm(u) < 0.03:
                continue
            p = project([[*xy, 0.68], [*(xy + 1.5 * u), 0.68]], result)
            a, b = p
            d.line([tuple(a), tuple(b)], fill=color, width=max(1, round(thickness * camera_scale)))
            direction = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            perp = np.array([-direction[1], direction[0]])
            d.polygon(
                [
                    tuple(b),
                    tuple(b - camera_scale * (13 * direction - 6 * perp)),
                    tuple(b - camera_scale * (13 * direction + 6 * perp)),
                ],
                fill=color,
            )
        return (
            Image.alpha_composite(frame, layer)
            .convert("RGB")
            .resize((640 * scale, 360 * scale), Image.Resampling.LANCZOS)
        )

    for index in range(count):
        t = index * 5 * cfg["dt_s"]
        canvas = Image.new("RGB", (width, height), bg)
        d = LayoutDraw(canvas, scale)
        d.text((24, 16), "CBFKit", font=bold(32), fill=white)
        d.text((185, 26), "A safety layer for robot policies", font=regular(20), fill=muted)
        d.text(
            (1090, 24),
            f"{t:4.1f} / {cfg['steps'] * cfg['dt_s']:.0f} s",
            font=regular(20),
            fill=white,
        )
        for side, mode in enumerate(("nominal", "filtered")):
            result, rows = runs[mode]
            row = rows[max(0, index - 1)]
            if index == 0:
                row = dict(
                    row,
                    xy=result["initial_xy"],
                    nominal=[[0.0, 0.0]] * cfg["num_envs"],
                    applied=[[0.0, 0.0]] * cfg["num_envs"],
                    intervened=[False] * cfg["num_envs"],
                    collided=[False] * cfg["num_envs"],
                    reached=[False] * cfg["num_envs"],
                )
            x = side * 640
            color = amber if side == 0 else teal
            d.text(
                (x + 24, 75),
                "ROUTE FOLLOWER" if side == 0 else "+ CBF SAFETY FILTER",
                font=bold(19),
                fill=color,
            )
            canvas.paste(decorated(mode, index, row), (x * scale, 108 * scale))
            d = LayoutDraw(canvas, scale)
            status = "DELIVERY REACHED" if row["reached"][0] else "FOLLOWING ROUTE"
            if row["intervened"][0] and side == 1:
                status = "ADJUSTING MOTION"
            if row["collided"][0]:
                status = "OBSTACLE CONTACT"
            status_color = red if row["collided"][0] else color
            d.rounded_rectangle((x + 20, 423, x + 309, 457), radius=5, fill=bg)
            d.text((x + 30, 431), status, font=bold(15), fill=status_color)
            n = cfg["num_envs"]
            d.text(
                (x + 24, 487),
                f"Contacts  {sum(row['collided'])}/{n}",
                font=bold(20),
                fill=red if any(row["collided"]) else white,
            )
            d.text((x + 338, 487), f"Goals  {sum(row['reached'])}/{n}", font=bold(20), fill=white)
        d.line((640, 66, 640, 527), fill=bg, width=5)
        d.line((24, 532, 1256, 532), fill="#283A47", width=1)
        d.text((24, 547), "Same route + walking policy", font=regular(16), fill=muted)
        d.text((450, 547), "White: requested motion", font=regular(16), fill=white)
        d.text((825, 547), "Teal: filtered motion", font=regular(16), fill=teal)
        d.text(
            (24, 576),
            "Actual Isaac Lab recording · simulator obstacle state · outline is a geometric margin",
            font=regular(13),
            fill=muted,
        )
        rendered.append(canvas)
    hero_end = len(rendered)
    rendered.extend([rendered[-1]] * 15)
    if args.grid:
        # A short synchronized excerpt of the separately simulated batch.
        start = min(15, grid_cfg["frame_count"] - 1)
        stop = min(start + 60, grid_cfg["frame_count"])
        for index in range(start, stop):
            canvas = Image.new("RGB", (width, height), bg)
            d = LayoutDraw(canvas, scale)
            d.text((24, 14), "16 scenarios. One GPU safety filter.", font=bold(29), fill=white)
            d.text(
                (24, 55),
                "Independent simulations on the RTX 3090 · shared policy, different cart trajectories",
                font=regular(16),
                fill=muted,
            )
            for camera_id in range(16):
                with Image.open(
                    args.grid / "frames" / f"env-{camera_id:02d}-{index:04d}.png"
                ) as im:
                    tile = im.convert("RGB").resize(
                        (220 * scale, 124 * scale), Image.Resampling.LANCZOS
                    )
                x = 196 + (camera_id % 4) * 222
                y = 88 + (camera_id // 4) * 126
                canvas.paste(tile, (x * scale, y * scale))
            rendered.append(canvas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "10",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        for frame in rendered:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoding failed")
    if args.gif:
        # README uses the concise comparison; the full MP4 contains the GPU grid.
        hero = [f.resize((1024, 480), Image.Resampling.LANCZOS) for f in rendered[: hero_end + 15]]
        sample = Image.new("RGB", (320 * 6, 150))
        for i, idx in enumerate(np.linspace(0, len(hero) - 1, 6, dtype=int)):
            sample.paste(hero[idx].resize((320, 150)), (i * 320, 0))
        palette = sample.quantize(colors=192)
        frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in hero]
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            args.gif, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True
        )
    rendered[min(40, hero_end - 1)].save(args.output.with_suffix(".preview.png"))
    print(f"Saved {args.output}: {len(rendered)} actual-camera frames")


if __name__ == "__main__":
    main()
