default: cffi

cffi: _whisper_cpp.*.so

_whisper_cpp.*.so: whisper.cpp/build/src/libwhisper.so
	python3 whisper_cffi_build.py

whisper.cpp/CMakeLists.txt:
	git submodule update --init --recursive

whisper.cpp/build/src/libwhisper.so: whisper.cpp/CMakeLists.txt
	cd whisper.cpp && cmake -B build && cmake --build build --config Release

clean:
	rm -rf _whisper_cpp.* libwhisper.so* whisper.cpp/build/

