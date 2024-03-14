from faster_whisper import WhisperModel
import sys
import os

if __name__ == "__main__":
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print("ERROR: Audio file not found")
        sys.exit(1)
    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    segments, _ = model.transcribe(
        audio_path, 
        language="vi",
        vad_filter=True,
        beam_size=1)

    segments = list(segments)  # The transcription will actually run here.

    list_seg = []
    for seg in segments:
        list_seg.append((seg.start, seg.end, seg.text))

    with open('./whisper_seg.pkl', 'wb') as f:
        pickle.dump(list_seg, f)
    print("OK")