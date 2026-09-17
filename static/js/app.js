let isHalted = false;
let lastAlertKey = '';
let currentPlaylist = [];
let currentPlaylistIndex = 0;
let audioPlayer = document.getElementById('audio-player');
let isPamphletOpen = false;
let videoStream = document.getElementById('video-stream');

let isHaltToggling = false;

function toggleHalt() {
    if (isHaltToggling) return;
    isHaltToggling = true;

    fetch('/api/halt_toggle', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            isHalted = data.is_halted;
            const btn = document.getElementById('btn-halt');
            const haltedOverlay = document.getElementById('halted-overlay');
            
            if (isHalted) {
                if (btn) { btn.innerText = "▶ RESUME SYSTEM"; btn.classList.add('active'); }
                if (haltedOverlay) haltedOverlay.classList.add('visible');
                audioPlayer.pause();
                audioPlayer.currentTime = 0;
            } else {
                if (btn) { btn.innerText = "⏹ HALT / STOP SYSTEM"; btn.classList.remove('active'); }
                if (haltedOverlay) haltedOverlay.classList.remove('visible');
                // Re-kick stream if browser stopped it
                videoStream.src = "/video_feed?t=" + new Date().getTime();
            }
            fetchStatus();
        })
        .catch(err => console.error("Halt toggle error:", err))
        .finally(() => {
            setTimeout(() => { isHaltToggling = false; }, 400);
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

            // Dynamic Pamphlet Squares: Highlight each item based on real-time detection
            const helmetItem = document.getElementById('p-item-helmet');
            const helmetStatus = document.getElementById('p-status-helmet');
            const vestItem = document.getElementById('p-item-vest');
            const vestStatus = document.getElementById('p-status-vest');
            const bootsItem = document.getElementById('p-item-boots');
            const bootsStatus = document.getElementById('p-status-boots');

            if (helmetItem && helmetStatus) {
                if (data.helmet_detected) {
                    helmetItem.className = 'pamphlet-item pass';
                    helmetStatus.innerText = '🟢 ACCEPTED / EQUIPPED';
                } else {
                    helmetItem.className = 'pamphlet-item fail';
                    helmetStatus.innerText = '🔴 MISSING / REQUIRED';
                }
            }

            if (vestItem && vestStatus) {
                if (data.vest_detected) {
                    vestItem.className = 'pamphlet-item pass';
                    vestStatus.innerText = '🟢 ACCEPTED / EQUIPPED';
                } else {
                    vestItem.className = 'pamphlet-item fail';
                    vestStatus.innerText = '🔴 MISSING / REQUIRED';
                }
            }

            if (bootsItem && bootsStatus) {
                bootsItem.className = 'pamphlet-item manual';
                bootsStatus.innerText = '👁️ MANUAL CHECK';
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

    // Highlight squares as passed
    const helmetItem = document.getElementById('p-item-helmet');
    const helmetStatus = document.getElementById('p-status-helmet');
    const vestItem = document.getElementById('p-item-vest');
    const vestStatus = document.getElementById('p-status-vest');
    if (helmetItem) { helmetItem.className = 'pamphlet-item pass'; helmetStatus.innerText = '🟢 ACCEPTED / EQUIPPED'; }
    if (vestItem) { vestItem.className = 'pamphlet-item pass'; vestStatus.innerText = '🟢 ACCEPTED / EQUIPPED'; }

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

// =============================================================================
// PRESENTER SECRET HOTKEYS (TRICK C)
// =============================================================================
let lastHotkeyTime = 0;

function handleHotkey(e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.repeat) return; // Ignore hold key repeat

    const now = Date.now();
    if (now - lastHotkeyTime < 350) return; // 350ms debounce

    const key = (e.key || '').toLowerCase();
    const code = e.code || '';

    // 'C' or '2' -> Force Clear (Shift Cleared)
    if (key === 'c' || key === '2' || code === 'KeyC' || code === 'Digit2' || code === 'Numpad2') {
        lastHotkeyTime = now;
        console.log("SafeSite Hotkey: FORCE CLEAR");
        forceClear();
    }
    // 'M' or '1' -> Force Missing Gear
    else if (key === 'm' || key === '1' || code === 'KeyM' || code === 'Digit1' || code === 'Numpad1') {
        lastHotkeyTime = now;
        console.log("SafeSite Hotkey: FORCE MISSING");
        forceMissing();
    }
    // 'R' -> Re-Scan current worker
    else if (key === 'r' || code === 'KeyR') {
        lastHotkeyTime = now;
        console.log("SafeSite Hotkey: RESCAN");
        rescanWorker();
    }
    // 'N' -> Next worker in line
    else if (key === 'n' || code === 'KeyN') {
        lastHotkeyTime = now;
        console.log("SafeSite Hotkey: NEXT WORKER");
        nextWorker();
    }
    // 'H' -> Halt / Resume system
    else if (key === 'h' || code === 'KeyH') {
        lastHotkeyTime = now;
        e.preventDefault();
        console.log("SafeSite Hotkey: HALT TOGGLE");
        toggleHalt();
    }
}

// Single robust listener on window
window.addEventListener('keydown', handleHotkey, false);

setInterval(fetchStatus, 1500);
fetchStatus();
