#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import argparse
import tempfile
import subprocess

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch.jit
import scipy.io.wavfile
import _whisper_cpp

SAMPLE_RATE = 16000

re_punct = re.compile(r'([\'\"#\(\)*+/:;<=>@\[\\\]^_`\{\|\}~，。、；「」『』 ]+)')
re_filter_zh = re.compile(r'^（(音量|字幕|互动中|人声|音乐|喝)')


def load_audio(filename):
    rate, data = scipy.io.wavfile.read(filename, mmap=True)
    if rate != SAMPLE_RATE:
        raise ValueError('Not 16k wav file')
    bit_range = 1 << ((data.dtype.itemsize*8) - 1)
    data = data.astype('float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data /= bit_range
    return data


def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def silero_get_speech_timestamps(
    audio: torch.Tensor, model,
    threshold: float = 0.5, sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
    return_ms: bool = False
):

    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100s (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_ms: bool (default - False)
        whether return timestamps in seconds (default - samples)

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_ms)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn('window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!')
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn('Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0 # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0 # to save potential segment limits in case of maximum segment size reached

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
               next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue
        
        if triggered and (window_size_samples * i) - current_speech['start'] > max_speech_samples:
            if prev_end:
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end: # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech['start'] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech['end'] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
                

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if ((window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech : # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_ms:
        for speech_dict in speeches:
            speech_dict['start'] = int(speech_dict['start'] * 1000 // sampling_rate)
            speech_dict['end'] = int(speech_dict['end'] * 1000 // sampling_rate)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


def format_srt_timestamp(ms):
    sec, ms = divmod(ms, 1000)
    minutes, seconds = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    return '%02d:%02d:%02d,%03d' % (hours, minutes, seconds, ms)


def segments_to_srt(segments, fp):
    for i, seg in enumerate(segments, 1):
        print(i, file=fp)
        t0, t1, txt = seg
        print("%s --> %s" % (format_srt_timestamp(t0), format_srt_timestamp(t1)), file=fp)
        print("%s\n" % txt, file=fp)
        fp.flush()


def merge_vad_segments(segments, min_interval_ms=1000):
    fixed_speeches = []
    last_end = None
    for segment in segments:
        if last_end and segment['start'] - last_end < min_interval_ms:
            fixed_speeches[-1] = (fixed_speeches[-1][0], segment['end'])
        else:
            fixed_speeches.append((segment['start'], segment['end']))
        last_end = segment['end']
    return fixed_speeches


def text_postprocess(text, language):
    if language == 'zh':
        import zhconv
        text = text.replace(',', '，').replace(';', '；').replace('(', '（').replace(')', '）').replace('?', '？').replace('!', '！')
        text = zhconv.convert(text, 'zh-hans')
        if re_filter_zh.search(text):
            return ''
    return text


class WhisperStuck(RuntimeError):
    pass


def fix_whisper_timestamps(
    last_segment, segments, result_offset_ms: int, start_ms: int, end_ms: int,
    overlap_chars=6, is_retry=False
):
    fixed = []
    if last_segment:
        fixed.append(last_segment)
    for t1, t2, txt in segments:
        t1 += result_offset_ms
        t2 += result_offset_ms
        if t1 < start_ms:
            t1 = start_ms
        elif t1 >= end_ms:
            continue
        if t2 > end_ms:
            t2 = end_ms
        txt = txt.strip()
        if t1 == t2 or not txt:
            continue
        if not fixed:
            fixed.append((t1, t2, txt))
            continue
        last_t1, last_t2, last_txt = fixed[-1]
        if t1 < last_t2:
            if txt == last_txt:
                if not is_retry:
                    raise WhisperStuck()
                fixed[-1] = (last_t1, t2, last_txt)
            elif last_txt.endswith(txt[:overlap_chars]):
                fixed[-1] = (last_t1, t2, last_txt + txt[overlap_chars:])
            else:
                fixed[-1] = (last_t1, t1, last_txt)
                fixed.append((t1, t2, txt))
        else:
            fixed.append((t1, t2, txt))
    return fixed


class WhisperCppVAD:

    def __init__(self, model: str, language='en', n_threads=4, translate=False) -> None:
        self.silero_model = init_jit_model(os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'silero', 'silero_vad.jit')))
        self.ctx = _whisper_cpp.lib.whisper_init_from_file(
            _whisper_cpp.ffi.new("char[]", model.encode('utf-8')))
        self.language = language
        self.cstr_language = _whisper_cpp.ffi.new("char[]", language.encode('utf-8'))
        self.params = _whisper_cpp.lib.whisper_full_default_params(
            _whisper_cpp.lib.WHISPER_SAMPLING_GREEDY)
        self.params.print_realtime = False
        self.params.print_progress = False
        self.params.print_timestamps = True
        self.params.print_special = False
        self.params.translate = bool(translate)
        self.params.language = self.cstr_language
        self.params.n_threads = n_threads
        self.params.offset_ms = 0
        self.params.duration_ms = 0
        self.last_segment = None
        self.max_context_time = 3*1000

    def transcribe_file(self, audio_data):
        speeches = silero_get_speech_timestamps(
            audio_data, self.silero_model,
            threshold=0.5, sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=50,
            window_size_samples=1536,
            speech_pad_ms=30,
            return_ms=True
        )
        segments = merge_vad_segments(speeches)

        for seg_start, seg_end in segments:
            # print(seg_start, seg_end)
            if seg_end - seg_start < 1000:
                continue
            try:
                results = self.transcribe_segment(audio_data, seg_start, seg_end)
            except WhisperStuck:
                print('[%s] Whisper repeated output! retry transcribe.' %
                    format_srt_timestamp(seg_start))
                results = self.transcribe_segment(
                    audio_data, seg_start, seg_end, True)
            # retry
            if not results:
                continue
            for t1, t2, txt in results:
                txt = text_postprocess(txt, self.language)
                if not txt:
                    continue
                print('[%s --> %s] %s' % (
                    format_srt_timestamp(t1), format_srt_timestamp(t2), txt))
                yield (t1, t2, txt)

        if self.last_segment:
            t1, t2, txt = self.last_segment
            txt = text_postprocess(txt, self.language)
            if txt:
                print('[%s --> %s] %s' % (
                    format_srt_timestamp(t1), format_srt_timestamp(t2), txt))
                yield (t1, t2, txt)

    def transcribe_segment(self, audio_data, start_ms, end_ms, is_retry=False):
        start_sample = int(start_ms * SAMPLE_RATE // 1000)
        end_sample = int(end_ms * SAMPLE_RATE // 1000)
        audio_view = audio_data[start_sample:end_sample]
        audio_buf = _whisper_cpp.ffi.from_buffer('float[]', audio_view)
        # self.params.offset_ms = offset_ms
        # self.params.duration_ms = duration_ms
        if (is_retry or (self.last_segment and
            start_ms < self.last_segment[1] + self.max_context_time
        )):
            self.params.no_context = True
        self.params.prompt_tokens = _whisper_cpp.ffi.NULL
        self.params.prompt_n_tokens = 0
        result = _whisper_cpp.lib.whisper_full(
            self.ctx, self.params, audio_buf, len(audio_buf))
        if result != 0:
            raise RuntimeError('Error from whisper.cpp: %s' % result)

        segments = []
        n_segments = _whisper_cpp.lib.whisper_full_n_segments(self.ctx)
        for i in range(n_segments):
            txt = _whisper_cpp.lib.whisper_full_get_segment_text(self.ctx, i)
            t0 = _whisper_cpp.lib.whisper_full_get_segment_t0(self.ctx, i)
            t1 = _whisper_cpp.lib.whisper_full_get_segment_t1(self.ctx, i)
            txt_str = bytes(_whisper_cpp.ffi.string(txt)).decode('utf-8', errors='ignore')
            segments.append((t0*10, t1*10, txt_str))
        # print(segments)
        results = fix_whisper_timestamps(
            self.last_segment, segments, start_ms, start_ms, end_ms, is_retry)
        # print(results)
        if results:
            self.last_segment = results.pop()
        return results

    def __del__(self):
        _whisper_cpp.lib.whisper_free(self.ctx)


def load_with_ffmpeg(filename, threads=1):
    with tempfile.TemporaryDirectory(prefix='whisper-vad-') as tmpdir:
        temp_wav = os.path.join(tmpdir, '1.wav')
        subprocess.run((
            'ffmpeg', '-threads', str(threads), '-i', filename, '-af',
            'loudnorm,aresample=resampler=soxr',
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', temp_wav))
        return load_audio(temp_wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Whisper STT with Voice Activity Detection.")
    parser.add_argument("-f", "--ffmpeg", help="Use ffmpeg to convert source file", action='store_true')
    parser.add_argument("-m", "--model", help="GGML model file")
    parser.add_argument("-l", "--language", default='en', help="Language")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Threads number")
    parser.add_argument("-T", "--translate", action='store_true', help="Translate")
    parser.add_argument("-o", "--output", help=".srt output file")
    parser.add_argument("file", help="Input audio file. If --ffmpeg is not used, the input must be 16k wav file")
    args = parser.parse_args()

    whisper = WhisperCppVAD(args.model, args.language, args.threads, args.translate)

    if args.ffmpeg:
        audio_data = load_with_ffmpeg(args.file)
    else:
        audio_data = load_audio(args.file)

    with open(args.output, 'w', encoding='utf-8') as f:
        segments_to_srt(whisper.transcribe_file(audio_data), f)
