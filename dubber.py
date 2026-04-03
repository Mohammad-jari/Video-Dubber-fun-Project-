import os
import tempfile
import logging
import html
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
from google.cloud import speech, translate_v2 as translate, texttospeech

logger = logging.getLogger(__name__)

CHUNK_DURATION_SEC = 55  # 55 seconds chunk to stay strictly under 1 minute limit

def process_video(input_video_path: str, output_video_path: str):
    logger.info("Loading video to extract audio...")
    
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

        logger.info(f"Total audio duration: {total_duration} seconds. Chunking into {CHUNK_DURATION_SEC} sec segments.")

        # 2. Transcription (Farsi)
        for chunk_start in range(0, int(total_duration), CHUNK_DURATION_SEC):
            chunk_end = min(chunk_start + CHUNK_DURATION_SEC, total_duration)
            logger.info(f"Processing STT for chunk from {chunk_start}s to {chunk_end}s")
            
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
                language_code="fa-IR",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="video"
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
                    logger.info(f"Original Transcript at {global_start_time_sec}s (orig duration {original_duration:.2f}s): {text}")
                    farsi_transcript_lines.append(text)
                    
                    # 3. Translation
                    translation = translate_client.translate(text, target_language="ur")
                    urdu_text = html.unescape(translation['translatedText'])
                    logger.info(f"Urdu Translation: {urdu_text}")
                    urdu_transcript_lines.append(urdu_text)
                    
                    # 4. Text-to-Speech
                    synthesis_input = texttospeech.SynthesisInput(text=urdu_text)
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
                    
                    tts_path = tempfile.mktemp(suffix=".mp3")
                    with open(tts_path, "wb") as out:
                        out.write(tts_response.audio_content)
                    
                    stt_results.append({
                        "start_sec": global_start_time_sec,
                        "audio_path": tts_path,
                        "original_duration": original_duration
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
            tts_clip = AudioFileClip(item["audio_path"])
            
            # Speed up Urdu audio if it's longer than the original speech segment
            if tts_clip.duration > item["original_duration"] and item["original_duration"] > 0:
                speed_factor = tts_clip.duration / item["original_duration"]
                logger.info(f"Speeding up Urdu clip (factor {speed_factor:.2f}) to fit {item['original_duration']:.2f}s")
                tts_clip = tts_clip.fx(afx.time_stretch, speed_factor)
            
            tts_clip = tts_clip.set_start(item["start_sec"])
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
        
        if os.path.exists(original_audio_path):
            os.remove(original_audio_path)
            
        logger.info("Video padding and dubbing completed successfully.")
        return output_video_path, "\n".join(farsi_transcript_lines), "\n".join(urdu_transcript_lines)
    
    finally:
        if video:
            video.close()
