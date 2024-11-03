import os
import sys
import glob
import shutil
from cffi import FFI

if sys.platform == "win32":
    # fix mingw builds
    import distutils.cygwinccompiler
    distutils.cygwinccompiler.get_msvcr = lambda: []


script_dir = os.path.abspath(os.path.dirname(__file__))
whisper_cpp_path = os.path.join(script_dir, 'whisper.cpp')


def load_whisper_header():
    header = ["""
    typedef bool (*ggml_abort_callback)(void * data);
    typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
    """]
    with open(os.path.join(whisper_cpp_path, 'include', 'whisper.h'), 'r', encoding='utf-8') as f:
        state = ''
        for line in f:
            line_strip = line.strip()
            if state == '':
                if line.startswith('extern "C"'):
                    state = 'start'
            elif state == 'start':
                if line.startswith('#'):
                    continue
                elif line_strip.startswith('//'):
                    continue
                elif line.startswith('}'):
                    break
                elif line_strip.startswith('WHISPER_DEPRECATED('):
                    state = 'deprecated0'
                    continue
                header.append(line.replace('WHISPER_API ', ''))
            elif state == 'deprecated0':
                if line_strip.startswith('WHISPER_API'):
                    #header.append(line.replace('WHISPER_API ', '').rstrip().rstrip(','))
                    #header.append(';\n')
                    continue
                elif line_strip.startswith('"'):
                    continue
                elif line_strip.startswith(');'):
                    state = 'start'
    return ''.join(header)


def copy_sofile(sofile, target_dir):
    links = []
    so_link = sofile
    while os.path.islink(so_link):
        link_target = os.readlink(so_link)
        links.append((os.path.basename(so_link), os.path.basename(link_target)))
        so_link = os.path.join(os.path.dirname(so_link), link_target)
    shutil.copy2(so_link, os.path.join(target_dir, os.path.basename(so_link)))
    for link_name, link_target in reversed(links):
        os.symlink(link_target, os.path.join(target_dir, link_name))


ffibuilder = FFI()
ffibuilder.cdef(load_whisper_header())


ffi_source = """
#include "whisper.h"
"""

ffibuilder.set_source("_whisper_cpp", ffi_source,
    library_dirs=[os.path.join(whisper_cpp_path, 'build', 'src')],
    include_dirs=[
        os.path.join(whisper_cpp_path, 'include'),
        os.path.join(whisper_cpp_path, 'ggml', 'include'),
    ],
    libraries=['whisper'],
    extra_link_args=['-Wl,-rpath,$ORIGIN']
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    try:
        os.remove('_whisper_cpp.o')
    except FileNotFoundError:
        pass
    shutil.rmtree(os.path.join(os.path.dirname(os.curdir), 'Release'), True)
    libwhisper_so = os.path.join(whisper_cpp_path, 'build', 'src', 'libwhisper.so')
    for filename in glob.glob(os.path.join(script_dir, 'libwhisper.so*')):
        os.remove(filename)
    copy_sofile(libwhisper_so, script_dir)
