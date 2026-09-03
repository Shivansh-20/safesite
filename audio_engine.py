"""
SafeSite AI — Multilingual Audio Voice Prompt Engine
Synthesizes and delivers native voice prompts in Hindi, Bhojpuri, and Maithili
sequentially to communicate safety compliance to regional laborers.
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

# Native Regional Language Audio Scripts
AUDIO_SCRIPTS = {
    "hi": {  # Hindi
        "NOT_VISIBLE": "कृपया कैमरे के सामने सीधे और ठीक से खड़े हों।",
        "HELMET_MISSING": "ध्यान दें! आपका हेलमेट नहीं पहना है। गेट पर रखे डिब्बे से हेलमेट पहनें।",
        "VEST_MISSING": "ध्यान दें! आपकी सुरक्षा जैकेट नहीं पहनी है। गेट से जैकेट पहनें।",
        "ALL_MISSING": "ध्यान दें! हेलमेट और सुरक्षा जैकेट दोनों नहीं पहने हैं! डिब्बे से सामान पहनें।",
        "CLEARED": "सुरक्षा जांच सफल! आपका काम पर स्वागत है।",
        "DANGER_ZONE": "खतरा! इस प्रतिबंधित क्षेत्र से तुरंत पीछे हटें।"
    },
    "bho": {  # Bhojpuri
        "NOT_VISIBLE": "अरे भाइया! कैमरा के सोझा ठीक से खड़ा हो जाईं।",
        "HELMET_MISSING": "अरे भाइया! राउर हेलमेट गायब बा। दुअरा पर राखल डिब्बा में से निकाल के पहिन ला!",
        "VEST_MISSING": "अरे भाइया! राउर सुरक्षा जैकेट गायब बा। दुअरा से जैकेट पहिन ला!",
        "ALL_MISSING": "अरे भाइया! हेलमेट आ जैकेट दोनो गायब बा! दुअरा से सामान पहिन ला!",
        "CLEARED": "सुरक्षा जाँच ठीक बा! काम पर राउर स्वागत बा।",
        "DANGER_ZONE": "खतरा बा! एहि जगह से तुरंत बहरा निकलीं!"
    },
    "mai": {  # Bihari / Maithili
        "NOT_VISIBLE": "सुनू भाई! क्यामरा कऽ सोझा ठीक सँ ठाढ़ि हू।",
        "HELMET_MISSING": "सुनू भाई! अहाँक हेलमेट नहि अछि। दुआरि पर राखल पेटी सँ हेलमेट लऽ कऽ पहिरू।",
        "VEST_MISSING": "सुनू भाई! अहाँक सुरक्षा जैकेट नहि अछि। दुआरि सँ जैकेट पहिरू।",
        "ALL_MISSING": "सुनू भाई! हेलमेट आ जैकेट दोनो नहि अछि! दुआरि सँ लऽ कऽ पहिरू।",
        "CLEARED": "सुरक्षा जाँच सफल! काज पर अहाँक स्वागत अछि।",
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
        logger.info("Initializing Multilingual Audio Assets...")
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
                    "url": f"/static/audio/{filename}",
                    "text": AUDIO_SCRIPTS[lang].get(alert_key, "")
                })
        return playlist

audio_engine = MultilingualAudioEngine()
