from itertools import permutations
import subprocess
import os

channel_swizzles = [''.join(p) for p in permutations("0123")]
script_dir = os.path.dirname(os.path.abspath(__file__))

for channels in channel_swizzles:
    cmd = [
        "python3",
        f"{script_dir}/wirconv.py",
        f"--outdir=out/out-{channels}",
        "--indir=.",
        f"--true-stereo-channel-swizzle={channels}",
    ]
    subprocess.run(cmd)
