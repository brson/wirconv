# wirconv

**This does not work yet. Still debugging.**

Converts `.wir` files, as used by Waves IR-1,
to standard `.wav` files,
loadable by typical convolution reverbs.

Recursively searches for `.wir` files and outputs `.wav` files .

By default it searches the current directory,
but if the `--indir` argument is provided it will search that directory.

By default it places its output next to the input,
but if the `--outdir` argument is provided it will place output in that directory,
recreating the original directory hierarchy within.

It requires python3.

Example:

```
python3 wirconv.py --indir ~/Waves_Complete_IR_Library --outdir ~/Waves_Complete_IR_Library_Converted
```

Each `.wav` file name is appended with either
`Mono`, `Stereo`, or `TrueStereo`,
indicating what kind of IR it is.

True stereo files are quad-channel,
with the channel order being the typical
L->L, L->R, R->L, R->R.

Only "direct" IRs are converted;
ambisonic (B-format) IRs are not converted.

The resulting files are 32-bit float, 96 kHz.


## Normalization

The sample data in `.wir` files is not normalized to the `(-1.0, 1.0)`
amplitude expected by typical audio software.
This converter does perform this normalization
such that the resulting `.wav` files are usable as-is.


## Bugs

The output of this conversion has not been
verified against the behavior of Waves IR-1,
and it is possible that it is incorrect.

In particular, it is possible that the channel order
of true stereo files is incorrect:
on aural and visual inspection most _seem_ to be correct,
but some seem to be suspicous.

It is likely that the samples are not normalized
identically to IR-1.

Several IRs have strange headers or data segments.
Some of the "synthetic" IRs in the Waves IR Library
seem to have samples that exhibit a specific corruption.
These are logged as "bogus" during the conversion process
and not converted.


## Credit

Based on https://github.com/opcode81/wir2wav

