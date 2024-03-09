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

1. `pip install -r requirements.txt`
2. `make`
3. `python3 whisper_vad.py --help` to see usage.


## GPU
This currently only supports CLBlast and AMD HIPBLAS.

### CLBlast

Dependencies: `libclblast`, `OpenCL`

Build: `WHISPER_CLBLAST=1 make`

### HIPBLAS

Dependencies: `libhipblas`, `libamdhip64`, `librocblas`

Build:
1. Build original [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
2. Copy `ggml-cuda.o` to `whisper_cpp`
3. `WHISPER_HIPBLAS=1 make`


