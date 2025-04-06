# 🎵 Python Simple Autotune

A minimal Python implementation of a real-time-ish **autotune** effect using pitch detection and pitch correction. This project was created to explore the fundamentals of pitch shifting, tuning quantization, and digital audio signal processing using Python and scientific libraries.

> 🎓 **Developed as a course project for Voice Processing**  
> Built to demonstrate core techniques in pitch tracking, frequency-domain transformations, and basic autotuning.

---

## ✨ Features

- ✅ Fundamental frequency detection using autocorrelation
- ✅ Pitch shifting to the nearest note in a selected musical scale
- ✅ Audio loading and playback using `pydub` and `pyaudio`
- ✅ Simple GUI using `Tkinter` for scale selection and tuning control
- ✅ WAV file input/output support
- ✅ Customizable musical scales

---

## 🧠 How It Works

1. Load a WAV file (mono preferred).
2. Detect pitch using autocorrelation or spectral analysis.
3. Quantize pitch to the nearest semitone in a chosen musical scale.
4. Adjust the audio signal by shifting frequency.
5. Output corrected audio.

---

## 🎧 Notes
- Works best with clean, monophonic vocal recordings.
- Autotune is applied offline (not truly real-time).
- This is an educational prototype — for production use, consider DSP libraries in C++ or frameworks like `JUCE`.
