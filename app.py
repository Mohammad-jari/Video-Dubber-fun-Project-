import streamlit as st
import tempfile
import os

# Setup Google Cloud Credentials
local_key_path = "farsi-to-urdu-dubber-a197665a6a37.json"

if "gcp_service_account_json" in st.secrets:
    creds_path = os.path.join(tempfile.gettempdir(), "gcp_creds.json")
    with open(creds_path, "w") as f:
        f.write(st.secrets["gcp_service_account_json"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
elif os.path.exists(local_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key_path
else:
    st.error("Google Cloud credentials not found. Please configure Streamlit secrets (gcp_service_account_json) or provide a local JSON key file.")
    st.stop()

from dubber import process_video

st.set_page_config(page_title="Multi-Language to Urdu Dubber", layout="wide")

st.title("Auto-Detect Video Dubber (to Urdu)")
st.markdown("Upload a Farsi or English video to automatically translate and dub it into Urdu.")

uploaded_file = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4", "mov"])
source_lang = st.selectbox("Select Source Language", ["Farsi", "English", "Auto-Detect"], index=0)

if uploaded_file is not None:
    # We load the uploaded file into a temporary input location
    temp_input = tempfile.mktemp(suffix=".mp4")
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.read())
        
    output_path = tempfile.mktemp(suffix=".mp4")
    
    with st.spinner("Processing video... This may take several minutes (Extracting audio, STT, Translating, TTS, Assembling)."):
        try:
            # Run the dubbing pipeline
            final_video_path, farsi_text, urdu_text = process_video(temp_input, output_path, source_lang_choice=source_lang)
            
            # Load the bytes so we can safely delete the local temp files afterwards
            with open(temp_input, "rb") as f:
                input_bytes = f.read()
            with open(final_video_path, "rb") as f:
                output_bytes = f.read()
            
            st.success("Video successfully processed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(input_bytes)
                
            with col2:
                st.subheader("Final Dubbed Urdu Video")
                st.video(output_bytes)
                st.download_button(
                    label="Download Dubbed Video",
                    data=output_bytes,
                    file_name=f"dubbed_{uploaded_file.name}",
                    mime="video/mp4"
                )
                
            st.subheader("Extracted Transcripts")
            tcol1, tcol2 = st.columns(2)
            
            with tcol1:
                st.text_area("Original Transcript", value=farsi_text, height=300)
            with tcol2:
                st.text_area("Urdu Translation", value=urdu_text, height=300)
                
        except Exception as e:
            import traceback
            st.error(f"An error occurred during processing:\n\n{traceback.format_exc()}")
        finally:
            try:
                if os.path.exists(temp_input):
                    os.remove(temp_input)
            except Exception:
                pass
            
            try:
                if 'final_video_path' in locals() and os.path.exists(final_video_path):
                    os.remove(final_video_path)
            except Exception:
                pass
