let isHalted = false;
let lastAlertKey = '';
let currentPlaylist = [];
let currentPlaylistIndex = 0;
let audioPlayer = document.getElementById('audio-player');
let isPamphletOpen = false;
let videoStream = document.getElementById('video-stream');

function toggleHalt() {
    fetch('/api/halt_toggle', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            isHalted = data.is_halted;
            const btn = document.getElementById('btn-halt');
            const haltedOverlay = document.getElementById('halted-overlay');
            
            if (isHalted) {
                btn.innerText = "▶ RESUME SYSTEM";
                btn.classList.add('active');
                haltedOverlay.classList.add('visible');
                videoStream.src = ""; // Clear image stream to release webcam HTTP connection
                audioPlayer.pause();
                audioPlayer.currentTime = 0;
            } else {
                btn.innerText = "⏹ HALT / STOP SYSTEM";
                btn.classList.remove('active');
                haltedOverlay.classList.remove('visible');
                videoStream.src = "/video_feed?t=" + new Date().getTime(); // Re-enable camera stream
            }
            fetchStatus();
        });
}

function rescanWorker() {
    // Hide pamphlet during scan so webcam feed is 100% visible
    if (isPamphletOpen) hidePamphlet();

    fetch('/api/rescan_worker', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Re-scan initiated:", data);
            lastAlertKey = '';
            fetchStatus();
        });
}

function nextWorker() {
    // Hide pamphlet for next worker
    if (isPamphletOpen) hidePamphlet();

    fetch('/api/next_worker', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Next worker loaded:", data);
            lastAlertKey = '';
            fetchStatus();
        });
}

function togglePamphletMode() {
    isPamphletOpen = !isPamphletOpen;
    const overlay = document.getElementById('pamphlet-overlay');
    const btn = document.getElementById('btn-pamphlet');
    
    if (isPamphletOpen) {
        overlay.classList.add('visible');
        btn.classList.add('active');
    } else {
        overlay.classList.remove('visible');
        btn.classList.remove('active');
    }
}

function hidePamphlet() {
    isPamphletOpen = false;
    document.getElementById('pamphlet-overlay').classList.remove('visible');
    document.getElementById('btn-pamphlet').classList.remove('active');
}

function showPamphlet() {
    isPamphletOpen = true;
    document.getElementById('pamphlet-overlay').classList.add('visible');
    document.getElementById('btn-pamphlet').classList.add('active');
}

function playNextInPlaylist() {
    if (currentPlaylistIndex < currentPlaylist.length) {
        const item = currentPlaylist[currentPlaylistIndex];
        const scriptText = document.getElementById('script-text');
        
        const langNames = { hi: '🇮🇳 Hindi', bho: '🌾 Bhojpuri', mai: '🚩 Maithili' };
        scriptText.innerText = `[${langNames[item.lang] || item.lang}]: "${item.text}"`;

        audioPlayer.src = item.url;
        audioPlayer.play().catch(e => console.log("Autoplay blocked:", e));

        currentPlaylistIndex++;
    }
}

audioPlayer.onended = function() {
    setTimeout(playNextInPlaylist, 1000);
};

function fetchStatus() {
    fetch('/api/get_status')
        .then(res => res.json())
        .then(data => {
            const stats = data.status;
            document.getElementById('stat-scans').innerText = stats.total_scans;
            document.getElementById('stat-cleared').innerText = stats.cleared_count;
            document.getElementById('stat-spare').innerText = stats.spare_ppe_issued;
            document.getElementById('stat-breaches').innerText = stats.violations_count;
            document.getElementById('val-uptime').innerText = data.uptime || '0s';

            document.getElementById('worker-title').innerText = `Worker #${data.worker_id} Gate Check-In`;

            const dot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');

            statusText.innerText = data.state_text;

            // Visual Status Colors
            if (data.alert_key === 'CLEARED') {
                dot.style.backgroundColor = '#10b981';
                dot.style.boxShadow = '0 0 10px #10b981';
            } else if (data.alert_key.includes('MISSING')) {
                dot.style.backgroundColor = '#ef4444';
                dot.style.boxShadow = '0 0 10px #ef4444';
            } else {
                dot.style.backgroundColor = '#00f0ff';
                dot.style.boxShadow = '0 0 10px #00f0ff';
            }

            // Pamphlet Auto-Display Logic: ONLY show pamphlet AFTER missing gear decision is locked
            if (data.is_locked && data.alert_key.includes('MISSING') && !isPamphletOpen) {
                showPamphlet();
            }

            // Sequential 3-Language Audio Loop
            if (data.alert_key !== lastAlertKey && data.playlist && data.playlist.length > 0 && !data.is_halted) {
                lastAlertKey = data.alert_key;
                currentPlaylist = data.playlist;
                currentPlaylistIndex = 0;
                playNextInPlaylist();
            }
        });
}

// Auto-reconnect camera stream if disconnected or interrupted
videoStream.onerror = function() {
    if (!isHalted) {
        console.warn("Camera stream interrupted. Reconnecting in 1s...");
        setTimeout(() => {
            videoStream.src = "/video_feed?t=" + new Date().getTime();
        }, 1000);
    }
};

// =============================================================================
// PRESENTER SECRET HOTKEYS (TRICK C)
// =============================================================================
function forceClear() {
    // 1. Instantly kill any playing warning audio
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    currentPlaylist = [];
    currentPlaylistIndex = 0;

    // 2. Instantly hide pamphlet overlay
    hidePamphlet();

    // 3. Immediate zero-latency UI update
    const dot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    dot.style.backgroundColor = '#10b981';
    dot.style.boxShadow = '0 0 15px #10b981';
    statusText.innerText = "🟢 SHIFT CLEARED! (VERIFIED)";
    document.getElementById('script-text').innerText = "Status: Worker Verified & Cleared for Shift.";

    fetch('/api/force_clear', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Presenter override: CLEARED", data);
            lastAlertKey = 'CLEARED';
            fetchStatus();
        })
        .catch(err => console.error("Override clear error:", err));
}

function forceMissing() {
    // 1. Instantly kill previous audio
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    currentPlaylist = [];
    currentPlaylistIndex = 0;

    // 2. Immediate zero-latency UI update
    const dot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    dot.style.backgroundColor = '#ef4444';
    dot.style.boxShadow = '0 0 15px #ef4444';
    statusText.innerText = "🔴 HELMET & VEST MISSING! COLLECT FROM BIN A";

    // 3. Pop up visual pamphlet
    showPamphlet();

    fetch('/api/force_missing', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Presenter override: MISSING", data);
            lastAlertKey = 'ALL_MISSING';
            fetchStatus();
        })
        .catch(err => console.error("Override missing error:", err));
}

// Listen for presenter hotkeys:
// Press 'C' or '2' -> Force Clear (Shift Cleared)
// Press 'M' or '1' -> Force Missing Gear (Red warning + pamphlet)
// Press 'R'        -> Re-Scan current worker
// Press 'N'        -> Next worker in line
// Press 'H'        -> Halt / Resume system
window.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const key = e.key.toLowerCase();
    if (key === 'c' || key === '2') {
        forceClear();
    } else if (key === 'm' || key === '1') {
        forceMissing();
    } else if (key === 'r') {
        rescanWorker();
    } else if (key === 'n') {
        nextWorker();
    } else if (key === 'h') {
        toggleHalt();
    }
});

setInterval(fetchStatus, 1500);
fetchStatus();
