Whisper VAD
===========

[Whisper.cpp](https://github.com/ggerganov/whisper.cpp) Speech-to-Text engine combined with [Silero Voice Activity Detector](https://github.com/snakers4/silero-vad).
This improves transcription speed and quality, and can avoid hallucination of the model.

Run `whisper_vad.py` directly for transcribing any video/audio files into SRT subtitles, or import it as a library.

## Dependencies

* ffmpeg (command)
* openblas (system library)
* cffi
* torch
* scipy
* zhconv: Chinese postprocess

## Build and usage

### Simple

1. `pip install -r requirements.txt`
2. `make`

### Custom Device

1. `git submodule update --init --recursive`
2. `cd whisper.cpp`
3. [Compile whisper.cpp to match your device](https://github.com/ggerganov/whisper.cpp)
   1. `cmake -B build` (add any build options)
   2. `cmake --build build --config Release -j8`
4. `pip install -r requirements.txt`
5. `make`

### Usage

`python3 whisper_vad.py --help` to see usage.

