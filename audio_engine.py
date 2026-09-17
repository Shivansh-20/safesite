"""
SafeSite AI — Multilingual Audio Voice Prompt Engine
Synthesizes and delivers conversational, natural voice prompts in Hindi, Bhojpuri, and Maithili
blended smoothly with familiar everyday Hindi terms (e.g. 'सेफ्टी गियर', 'गेट', 'बॉक्स').
Zero equipment-specific names (helmet/vest) are mentioned in audio.
"""

import os
import time
import logging
from gtts import gTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeSiteAudio")

# Static Audio Assets Directory
BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Toned-down, natural, Hindi-blended regional audio scripts
AUDIO_SCRIPTS = {
    "hi": {  # Hindi
        "NOT_VISIBLE": "कृपया कैमरे के सामने ठीक से खड़े हों।",
        "HELMET_MISSING": "कृपया ध्यान दें, आपका सेफ्टी गियर अधूरा है। गेट के बॉक्स से गियर पहन लें।",
        "VEST_MISSING": "कृपया ध्यान दें, आपका सेफ्टी गियर अधूरा है। गेट के बॉक्स से गियर पहन लें।",
        "ALL_MISSING": "कृपया ध्यान दें, आपका सेफ्टी गियर अधूरा है। गेट के बॉक्स से गियर पहन लें।",
        "CLEARED": "सुरक्षा जांच पूरी हुई। काम पर आपका स्वागत है।",
        "DANGER_ZONE": "खतरा! इस प्रतिबंधित क्षेत्र से तुरंत पीछे हटें।"
    },
    "bho": {  # Bhojpuri (Natural Conversational / Blended with Hindi)
        "NOT_VISIBLE": "भइया, कैमरा के सामने ठीक से खड़ा हो जाईं।",
        "HELMET_MISSING": "भइया ध्यान दीं, राउर सेफ्टी गियर बाकी बा। गेट के बॉक्स से गियर पहिन लीं।",
        "VEST_MISSING": "भइया ध्यान दीं, राउर सेफ्टी गियर बाकी बा। गेट के बॉक्स से गियर पहिन लीं।",
        "ALL_MISSING": "भइया ध्यान दीं, राउर सेफ्टी गियर बाकी बा। गेट के बॉक्स से गियर पहिन लीं।",
        "CLEARED": "जांच पूरा भइल। काम पर स्वागत बा।",
        "DANGER_ZONE": "खतरा बा! एहि जगह से तुरंत बाहर निकलीं!"
    },
    "mai": {  # Bihari / Maithili (Natural Conversational / Blended with Hindi)
        "NOT_VISIBLE": "भाई जी, कैमरा के सामने ठीक सँ खड़ा रहू।",
        "HELMET_MISSING": "भाई जी ध्यान दियौ, अहाँक सेफ्टी गियर बाकी अछि। गेट के बॉक्स सँ गियर पहिर लियौ।",
        "VEST_MISSING": "भाई जी ध्यान दियौ, अहाँक सेफ्टी गियर बाकी अछि। गेट के बॉक्स सँ गियर पहिर लियौ।",
        "ALL_MISSING": "भाई जी ध्यान दियौ, अहाँक सेफ्टी गियर बाकी अछि। गेट के बॉक्स सँ गियर पहिर लियौ।",
        "CLEARED": "जांच पूरा भेल। काम पर स्वागत अछि।",
        "DANGER_ZONE": "खतरा अछि! एहि ठाउँ सँ तुरंत बाहर भऽ जाऊ।"
    }
}

class MultilingualAudioEngine:
    """
    Manages pre-synthesized regional MP3 assets and delivers sequential 
    multilingual audio playlists (Hindi -> Bhojpuri -> Maithili).
    """
    def __init__(self):
        self._pregenerate_audio_files()

    def _pregenerate_audio_files(self):
        """Generates MP3 audio files for low-latency playback."""
        logger.info("Initializing Multilingual Audio Assets (Toned-down Hindi-blended)...")
        for lang_code, messages in AUDIO_SCRIPTS.items():
            for alert_key, text_content in messages.items():
                filename = f"{lang_code}_{alert_key}.mp3"
                filepath = os.path.join(AUDIO_DIR, filename)
                if not os.path.exists(filepath):
                    try:
                        tts = gTTS(text=text_content, lang="hi", slow=False)
                        tts.save(filepath)
                        logger.info(f"Generated Audio: {filename}")
                    except Exception as e:
                        logger.error(f"Error generating audio {filename}: {e}")

    def get_sequential_playlist(self, alert_key: str) -> list:
        """
        Returns an ordered audio playlist for Hindi, Bhojpuri, and Maithili sequentially.
        """
        playlist = []
        for lang in ["hi", "bho", "mai"]:
            filename = f"{lang}_{alert_key}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(filepath):
                playlist.append({
                    "lang": lang,
                    "url": f"/static/audio/{filename}?t=" + str(int(time.time())),
                    "text": AUDIO_SCRIPTS[lang].get(alert_key, "")
                })
        return playlist

audio_engine = MultilingualAudioEngine()
