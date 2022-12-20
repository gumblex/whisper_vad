default: cffi

cffi:
	cd whisper_cpp && python3 whisper_cffi_build.py && mv _whisper_cpp.*.so ../



clean:
	rm -f whisper_cpp/*.o _whisper_cpp.*.so

