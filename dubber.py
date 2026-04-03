import os
import tempfile
import logging
import html
import subprocess
import re
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
from moviepy.config import FFMPEG_BINARY
from google.cloud import speech, translate_v2 as translate, texttospeech

logger = logging.getLogger(__name__)

def get_silence_timestamps(audio_path, threshold="-30dB", duration="0.4"):
    """Uses ffmpeg to detect silences in the audio."""
    cmd = [
        FFMPEG_BINARY, "-i", audio_path,
        "-af", f"silencedetect=noise={threshold}:d={duration}",
        "-f", "null", "-"
    ]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    _, stderr = process.communicate()
    
    # Extract silence_end timestamps
    silence_ends = [float(t) for t in re.findall(r"silence_end: ([\d\.]+)", stderr)]
    return silence_ends

def adjust_audio_speed_pitch_preserved(input_path, speed_factor):
    """Speeds up or slows down audio without changing pitch using ffmpeg atempo."""
    output_path = tempfile.mktemp(suffix=".mp3")
    
    # atempo filter is limited to [0.5, 2.0]. Chain filters if out of range.
    filters = []
    temp_factor = speed_factor
    while temp_factor > 2.0:
        filters.append("atempo=2.0")
        temp_factor /= 2.0
    while temp_factor < 0.5:
        filters.append("atempo=0.5")
        temp_factor /= 0.5
    filters.append(f"atempo={temp_factor}")
    
    filter_str = ",".join(filters)
    cmd = [FFMPEG_BINARY, "-y", "-i", input_path, "-filter:a", filter_str, output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

def process_video(input_video_path: str, output_video_path: str, source_lang_choice: str = "Farsi"):
    logger.info(f"Processing video with source language choice: {source_lang_choice}")
    
    # Smart Language & Model Selection
    if source_lang_choice == "Farsi":
        language_code = "fa-IR"
        alt_langs = []
        model = "default"
    elif source_lang_choice == "English":
        language_code = "en-US"
        alt_langs = []
        model = "default"
    else:  # "Auto-Detect"
        language_code = "fa-IR"
        alt_langs = ["en-US"]
        model = "default"
    
    video = None
    try:
        video = VideoFileClip(input_video_path)
        
        if not video.audio:
            raise ValueError("The provided video has no audio track.")

        original_audio_path = tempfile.mktemp(suffix=".wav")
        
        # 1. Audio Extraction: 16kHz mono 
        logger.info("Extracting audio from video...")
        video.audio.write_audiofile(
            original_audio_path,
            fps=16000,
            nbytes=2,
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        
        total_duration = video.audio.duration
        stt_results = []
        farsi_transcript_lines = []
        urdu_transcript_lines = []
        
        speech_client = speech.SpeechClient()
        translate_client = translate.Client()
        tts_client = texttospeech.TextToSpeechClient()

        # 2. Smart Chunking (Silence-based)
        logger.info("Detecting silences for smart chunking...")
        silence_points = get_silence_timestamps(original_audio_path)
        
        chunks = []
        last_cut = 0
        target_chunk_size = 50
        
        while last_cut < total_duration:
            next_target = last_cut + target_chunk_size
            if next_target >= total_duration:
                chunks.append((last_cut, total_duration))
                break
            
            # Find the best silence point between [next_target - 10, next_target + 5]
            best_point = next_target
            candidates = [p for p in silence_points if next_target - 15 < p < next_target + 10]
            if candidates:
                # Pick the one closest to next_target
                best_point = min(candidates, key=lambda p: abs(p - next_target))
            
            # Ensure we don't exceed the 60s Google STT limit (leave buffer)
            if best_point - last_cut > 58:
                best_point = last_cut + 58
                
            chunks.append((last_cut, best_point))
            last_cut = best_point

        for chunk_start, chunk_end in chunks:
            logger.info(f"Processing STT for smart chunk from {chunk_start:.2f}s to {chunk_end:.2f}s")
            
            chunk = video.audio.subclip(chunk_start, chunk_end)
            chunk_path = tempfile.mktemp(suffix=".wav")
            chunk.write_audiofile(
                chunk_path, fps=16000, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"], logger=None
            )
            
            with open(chunk_path, "rb") as f:
                audio_content = f.read()

            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                alternative_language_codes=alt_langs,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model=model
            )

            try:
                response = speech_client.recognize(config=config, audio=audio)
                
                for result in response.results:
                    alt = result.alternatives[0]
                    if not alt.words:
                        continue
                    
                    start_time_sec = alt.words[0].start_time.total_seconds()
                    end_time_sec = alt.words[-1].end_time.total_seconds()
                    original_duration = end_time_sec - start_time_sec
                    
                    text = alt.transcript
                    
                    global_start_time_sec = chunk_start + start_time_sec
                    logger.info(f"Transcript at {global_start_time_sec:.2f}s (dur {original_duration:.2f}s): {text}")
                    farsi_transcript_lines.append(text)
                    
                    # 3. Translation
                    translation = translate_client.translate(text, target_language="ur")
                    urdu_text = html.unescape(translation['translatedText'])
                    logger.info(f"Urdu Translation: {urdu_text}")
                    urdu_transcript_lines.append(urdu_text)
                    
                    # 4. Text-to-Speech (using SSML for better pronunciation)
                    ssml_text = f"<speak>{urdu_text}</speak>"
                    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
                    voice = texttospeech.VoiceSelectionParams(
                        language_code="ur-IN",
                        name="ur-IN-Wavenet-B"
                    )
                    audio_config = texttospeech.AudioConfig(
                        audio_encoding=texttospeech.AudioEncoding.MP3
                    )
                    
                    tts_response = tts_client.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )
                    
                    raw_tts_path = tempfile.mktemp(suffix=".mp3")
                    with open(raw_tts_path, "wb") as out:
                        out.write(tts_response.audio_content)
                    
                    # Pitch-Preserved Speed Sync
                    tts_clip_temp = AudioFileClip(raw_tts_path)
                    final_tts_path = raw_tts_path
                    
                    if tts_clip_temp.duration > original_duration and original_duration > 0:
                        speed_factor = tts_clip_temp.duration / original_duration
                        logger.info(f"Speeding up Urdu (factor {speed_factor:.2f}) with pitch preservation...")
                        final_tts_path = adjust_audio_speed_pitch_preserved(raw_tts_path, speed_factor)
                    
                    tts_clip_temp.close()
                    
                    stt_results.append({
                        "start_sec": global_start_time_sec,
                        "audio_path": final_tts_path,
                        "raw_path": raw_tts_path if final_tts_path != raw_tts_path else None
                    })
            except Exception as e:
                logger.error(f"Error processing chunk starting at {chunk_start}s: {e}")
                raise e
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                
        # 5. Final Assembly
        logger.info("Assembling final audio track with translated sentences...")
        # Maintain background audio at 15% volume (Audio Ducking)
        base_audio = AudioFileClip(original_audio_path).volumex(0.15)
        
        audio_clips = [base_audio]
        loaded_tts_clips = []
        
        for item in stt_results:
            tts_clip = AudioFileClip(item["audio_path"]).set_start(item["start_sec"])
            audio_clips.append(tts_clip)
            loaded_tts_clips.append(tts_clip)
                
        final_audio = CompositeAudioClip(audio_clips)
        final_audio = final_audio.set_duration(base_audio.duration)
            
        logger.info("Replacing original video audio track with the new Urdu track...")
        final_video = video.set_audio(final_audio)
        
        logger.info("Writing final video file...")
        final_video.write_videofile(
            output_video_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None
        )
        
        logger.info("Cleaning up pipeline temp files...")
        
        final_video.close()
        final_audio.close()
        base_audio.close()
        for c in loaded_tts_clips:
            c.close()
        
        for item in stt_results:
            if os.path.exists(item["audio_path"]):
                os.remove(item["audio_path"])
            if item.get("raw_path") and os.path.exists(item["raw_path"]):
                os.remove(item["raw_path"])
        
        if os.path.exists(original_audio_path):
            os.remove(original_audio_path)
            
        logger.info("Video padding and dubbing completed successfully.")
        return output_video_path, "\n".join(farsi_transcript_lines), "\n".join(urdu_transcript_lines)
    
    finally:
        if video:
            video.close()
