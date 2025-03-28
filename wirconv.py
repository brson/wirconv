from itertools import permutations
import wave
import struct
import io
import argparse
from enum import Enum
from fnmatch import fnmatch
from types import SimpleNamespace
import os


default_max_db = -18.0
default_ts_channel_swizzle_str = "1032"
# stereo channels seem to be backwards
stereo_channel_swizzle = [1,0]

# candidate channel orders - these sound good in bitwig
#
# 1302 - i think this is it, RL,LL,RR,LR??
# 1032
# 3021

class Channels(Enum):
    MONO = 4
    STEREO = 8
    TRUE_STEREO = 16

class TrueStereoOut(Enum):
    MUXED = 1
    SPLIT = 2

class Wir:
    def __init__(self, path, ts_format: TrueStereoOut, ts_channel_swizzle: list[int]):
        self.ts_format = ts_format
        self.ts_channel_swizzle = ts_channel_swizzle
    
        with open(path, "rb") as f:
            self.path = path
            self.header = f.read(40)
            """
            Header format
            (based on http://freeverb3vst.osdn.jp/tips/tips.shtml)

            // 32bit LE
            typedef struct{
              0: char magic[4]; // "wvIR"
              4: int fileSizeLE; // filesize-8
              8: char version[8]; // version "ver1fmt "
              16: int headerSizeLE;
              20: short int i3; // 0x3
              22: short int channels;
              24: int fs; // sample/frame rate
              28: int fs2;
              32: short int i4; // MONO 0x4 STEREO 0x8 4CH 0x10
              34: short int i5; // 0x17
              36: char data[4]; // "data"
            } WirHeader;
            40: // rest of the data is FLOAT_LE (32bit float)
            """
            self.data = f.read()
            self.num_channels = struct.unpack("H", self.header[22:24])[0]
            self.framerate = struct.unpack("I", self.header[24:28])[0]
            self.fs2 = struct.unpack("I", self.header[28:32])[0]
            self.channels_mask = struct.unpack("H", self.header[32:34])[0]
            self.file_size = struct.unpack("I", self.header[4:8])[0]
            self.version = struct.unpack("8s", self.header[8:16])[0]

    def duration_secs(self):
        return len(self.data) / 4 / self.num_channels / self.framerate

    def __str__(self):
        chan_strings = []
        for chan in list(Channels):
            if self.channels_mask & chan.value:
                chan_strings.append(str(chan))
        return f"WIR[{self.path}, {self.num_channels} channels [{' + '.join(chanStrings)}], {self.framerate} Hz, {self.duration_secs():.3f} secs]"

    def actual_sample_len(self):
        # A few data sections are not evenly divisble by 4.
        return int(len(self.data) / 4)

    def scaled_samples(self):
        amplitude = 10 ** (default_max_db/20)
        actual_sample_len = self.actual_sample_len()
        samples = struct.unpack(f"{actual_sample_len}f", self.data[:actual_sample_len * 4])
        max_val = max(abs(s) for s in samples)
        samples = [s / max_val * amplitude for s in samples] if max_val > 0 else samples
        samples = struct.pack(f'<{len(samples)}f', *samples)
        return samples

    def swizzle_channels(self, samples, channels):
        num_channels = len(channels)
        num_bytes = len(samples)
        framesize = num_channels * 4
        assert num_bytes % framesize == 0

        buf = io.BytesIO()
        offs = 0

        while offs < num_bytes:
            frame = samples[offs:offs+framesize]
            for channel in channels:
                sample = frame[channel*4:channel*4+4]
                buf.write(sample)
            offs += framesize

        return buf.getvalue()

    def write_wavs(self, outpath, file_stem):
        props = self.get_props()

        if props.is_bogus:
            print(f"BOGUS: {self.path}")
            return 0
        if props.is_ambisonic:
            return 0

        os.makedirs(outpath, exist_ok=True)

        samples = self.scaled_samples()

        # Many files seem to have unequal channel lengths :-/
        framesize = props.num_channels * 4
        num_frames = len(samples) // framesize
        truncated_bytes = num_frames * framesize
        samples = samples[:truncated_bytes]

        sample_rate = self.framerate

        if props.direct_config == Channels.TRUE_STEREO and self.ts_format == TrueStereoOut.MUXED:
            assert props.num_channels == 4
            assert len(self.ts_channel_swizzle) == 4
            samples = self.swizzle_channels(samples, ts_channel_swizzle)
            file_name = self.out_file_name(outpath, file_stem, props, "")
            self.write_single_wav(file_name, props.num_channels, sample_rate, samples)
        elif props.direct_config == Channels.TRUE_STEREO and self.ts_format == TrueStereoOut.SPLIT:
            assert props.num_channels == 4
            assert len(self.ts_channel_swizzle) == 4
            samples = self.swizzle_channels(samples, ts_channel_swizzle[0:2])
            file_name = self.out_file_name(outpath, file_stem, props, " L")
            self.write_single_wav(file_name, 2, sample_rate, samples)
            samples = self.swizzle_channels(samples, ts_channel_swizzle[2:4])
            file_name = self.out_file_name(outpath, file_stem, props, " R")
            self.write_single_wav(file_name, 2, sample_rate, samples)
        elif props.direct_config == Channels.STEREO:
            assert props.num_channels == 2
            samples = self.swizzle_channels(samples, stereo_channel_swizzle)
            file_name = self.out_file_name(outpath, file_stem, props, "")
            self.write_single_wav(file_name, props.num_channels, sample_rate, samples)
        else:
            assert props.num_channels == 1
            file_name = self.out_file_name(outpath, file_stem, props, "")
            self.write_single_wav(file_name, props.num_channels, sample_rate, samples)

        return 1

    def out_file_name(self, outpath, file_stem, props, tag):
        assert not props.is_bogus and not props.is_ambisonic
        channel_slug = ""
        if props.direct_config == Channels.MONO:
            channel_slug = "Mono"
        if props.direct_config == Channels.STEREO:
            channel_slug = "Stereo"
        if props.direct_config == Channels.TRUE_STEREO:
            channel_slug = "TrueStereo"
        return f"{outpath}/{file_stem} {channel_slug}{tag}.wav"

    def get_props(self):
        num_channels = self.num_channels

        # a few files have suspicious sizes
        is_good_data_len = len(self.data) % 4 == 0
        # one file declares a suspicous data size in the header
        is_correct_header_file_size = (self.file_size - (40 - 8)) == len(self.data)

        num_samples = int(len(self.data) / 4)

        has_mono = self.channels_mask & Channels.MONO.value != 0
        has_stereo = self.channels_mask & Channels.STEREO.value != 0
        has_true_stereo = self.channels_mask & Channels.TRUE_STEREO.value != 0
        assert self.channels_mask != 0

        # 1 file has strange channel mask
        has_strange_channel_mask = False
        if not (has_mono or has_stereo or has_true_stereo):
            assert self.channels_mask == 3
            assert self.num_channels == 1
            has_strange_channel_mask = True

        # these are the only combinations of channels that appear
        dual_mono_stereo = has_mono and has_stereo and not has_true_stereo
        dual_stereo_true_stereo = has_stereo and has_true_stereo and not has_mono
        single_config = has_mono ^ has_stereo ^ has_true_stereo
        assert dual_mono_stereo ^ dual_stereo_true_stereo ^ single_config ^ has_strange_channel_mask

        # some declare mono but have two channels
        is_mono_with_two_channels = single_config and has_mono and num_channels == 2

        # direct (not ambisonic / b-format) IRs only have one channel config
        is_ambisonic = not single_config

        # some of the synthetic IRs have specificly-corrupted samples
        bogus_sample1 = 0x726f4620
        bogus_sample2 = 0x796e6f53
        def check_bogus(data):
            for index in range(int(len(data) / 4)):
                sample = struct.unpack_from('I', data, index * 4)[0]
                is_bogus1 = sample == bogus_sample1
                is_bogus2 = sample == bogus_sample2
                if is_bogus1 or is_bogus2:
                    return True
            return False
        has_bogus_sample = check_bogus(self.data)

        direct_config = None
        if not is_ambisonic and has_mono:
            direct_config = Channels.MONO
        if not is_ambisonic and has_stereo:
            direct_config = Channels.STEREO
        if not is_ambisonic and has_true_stereo:
            direct_config = Channels.TRUE_STEREO

        is_bogus = not is_good_data_len \
            or not is_correct_header_file_size \
            or has_strange_channel_mask \
            or is_mono_with_two_channels \
            or has_bogus_sample

        if direct_config == Channels.MONO and not is_bogus:
            assert num_channels == 1
        if direct_config == Channels.STEREO and not is_bogus:
            assert num_channels == 2
        if direct_config == Channels.TRUE_STEREO and not is_bogus:
            assert num_channels == 4

        return SimpleNamespace(
            is_bogus = is_bogus,
            is_ambisonic = is_ambisonic,
            num_channels = num_channels,
            num_samples = num_samples,
            direct_config = direct_config,
            has_mono = has_mono,
            has_stereo = has_stereo,
            has_true_stereo = has_true_stereo,
        )

    def write_single_wav(self, file_name, num_channels, sample_rate, samples):
        byte_count = len(samples)
        with open(file_name, "wb") as wav:
            wav.write(struct.pack('<ccccIccccccccIHHIIHH',
                b'R', b'I', b'F', b'F',
                byte_count + 0x2c - 8,  # header size
                b'W', b'A', b'V', b'E', b'f', b'm', b't', b' ',
                0x10,  # size of 'fmt ' header
                3,  # format 3 = floating-point PCM
                num_channels,  # channels
                sample_rate,  # samples / second
                num_channels * sample_rate * 4,  # bytes / second
                4,  # block alignment
                32))  # bits / sample
            wav.write(struct.pack('<ccccI', b'd', b'a', b't', b'a', byte_count))
            wav.write(samples)

    def write_single_wav2(self, file_name, num_channels, sample_rate, samples):
        with wave.open(file_name, "wb") as wave_file:
            wave_file.setnchannels(num_channels)
            wave_file.setsampwidth(4)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(samples)

def parse_channel_swizzle(swizzle_str: str) -> list[int]:
    if len(swizzle_str) != 4 or not swizzle_str.isdigit():
        raise ValueError("Input must be a string of exactly four digits.")
    
    return [int(char) for char in swizzle_str]    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=False, default=".", help="Input directory")
    parser.add_argument("--outdir", type=str, required=False, default=".", help="Output directory")
    parser.add_argument("--split-true-stereo", action="store_true", required=False, default=False, help="Use split true stereo files")
    parser.add_argument("--true-stereo-channel-swizzle", type=str, required=False, default=default_ts_channel_swizzle_str, help="True stereo channel swizzle")
    args = parser.parse_args()

    in_dir = args.indir
    out_dir = args.outdir
    ts_format = TrueStereoOut.SPLIT if args.split_true_stereo else TrueStereoOut.MUXED
    ts_channel_swizzle = parse_channel_swizzle(args.true_stereo_channel_swizzle)

    print(f"in dir: {in_dir}")
    print(f"out dir: {out_dir}")
    print(f"true stereo out format: {ts_format}")
    print(f"true stereo channel swizzle: {ts_channel_swizzle}")

    num_found = 0
    num_conversions = 0
    for path, dirs, files in os.walk(in_dir):
        for file_name in files:
            if fnmatch(file_name, "*.wir"):
                infile = os.path.join(path, file_name)
                wir = Wir(infile, ts_format, ts_channel_swizzle)
                relpath = path.removeprefix(in_dir)
                outpath = f"{out_dir}/{relpath}"
                file_stem = os.path.splitext(file_name)[0]
                files_written = wir.write_wavs(outpath, file_stem)
                num_found += 1
                num_conversions += files_written
    print(f"{num_found} wirs found, {num_conversions} converted.")
