/**
 * script.js
 * ==========
 * Frontend logic for Exercise Form Detection web app.
 *
 * ARCHITECTURE OVERVIEW
 * ─────────────────────
 * 1. Camera  : getUserMedia() → <video> element (raw webcam, no processing)
 * 2. Capture : setInterval captures video frame → JPEG → WebSocket binary
 * 3. WS recv : JSON response with keypoints + form feedback arrives
 * 4. Canvas  : draw_skeleton() + draw_joints() render over the video
 * 5. UI      : update rep counter, joint cards, status banner
 *
 * WHY capture on the frontend and send to backend?
 *   Alternative: backend streams processed video back.
 *   Problem: re-encoding a JPEG each frame on backend = high CPU + latency.
 *   Our approach: backend returns lightweight JSON (< 2 KB/frame), frontend
 *   draws the overlay locally. Result: 2–3× lower latency.
 *
 * WHY Canvas overlay instead of CSS filter?
 *   Canvas gives pixel-level control: anti-aliased circles, glow effects,
 *   dynamic colors based on form status. CSS filters can't do any of that.
 */

'use strict';

// =============================================================================
// SECTION 1 — CONFIGURATION
// =============================================================================

const CONFIG = {
    WS_URL:          'ws://localhost:8000/ws',
    CAPTURE_FPS:     30,           // frames sent to backend per second (increased from 20)
    JPEG_QUALITY:    0.75,         // 0-1, quality of captured frames
    CONF_THRESHOLD:  0.50,         // hide keypoints below this confidence
    JOINT_RADIUS:    8,            // base circle radius (px on canvas)
    BONE_THICKNESS:  3,            // skeleton line width (px)
    GLOW_ALPHA:      0.35,         // glow layer opacity
    TRAIL_FRAMES:    5,            // ghost frames behind current pose
    EMA_ALPHA:       0.75,         // keypoint position smoothing (high responsiveness)
};

// =============================================================================
// SECTION 2 — SKELETON DEFINITION
// =============================================================================
// Mirrors src/pose_extractor.py SKELETON_CONNECTIONS
// Each entry: [indexA, indexB, side]  side → color key

const BONES = [
    [5,  6,  'torso'],  // shoulders
    [5,  7,  'left' ],  // L upper arm
    [7,  9,  'left' ],  // L forearm
    [6,  8,  'right'],  // R upper arm
    [8,  10, 'right'],  // R forearm
    [5,  11, 'left' ],  // L torso
    [6,  12, 'right'],  // R torso
    [11, 12, 'torso'],  // pelvis
    [11, 13, 'left' ],  // L thigh
    [13, 15, 'left' ],  // L shin
    [12, 14, 'right'],  // R thigh
    [14, 16, 'right'],  // R shin
];

// Keypoint group assignment (index → group)
// Mirrors src/renderer.py KP_GROUP
const KP_GROUP = [
    'face','face','face','face','face',  // 0-4  nose, eyes, ears
    'left','right',                       // 5-6  shoulders
    'left','right',                       // 7-8  elbows
    'left','right',                       // 9-10 wrists
    'left','right',                       // 11-12 hips
    'left','right',                       // 13-14 knees
    'left','right',                       // 15-16 ankles
];

// Per-exercise active joints (indices to highlight with glow ring)
const ACTIVE_JOINTS = {
    bicep_curl:     [5,6,7,8,9,10,11,12],
    squat:          [11,12,13,14],
    lateral_raise:  [5,6,7,8],
    push_up:        [7,8,11,12],
    lunge:          [11,12,13,14,15,16],
    plank:          [11,12,13,14,15,16],
};

// Exact colors from src/renderer.py Colors class (converted BGR→RGB)
const COLORS = {
    // Skeleton bones
    left:   '#1E50DC',   // LEFT_BONE = (220, 80, 30) BGR
    right:  '#DC501E',   // RIGHT_BONE = (30, 80, 220) BGR
    torso:  '#64C864',   // TORSO_BONE = (100, 200, 100) BGR
    // Keypoint groups
    face:   '#C8C8C8',   // FACE = (200, 200, 200) BGR
    leftKp: '#1E64FF',   // LEFT_BODY = (255, 100, 30) BGR
    rightKp:'#FF641E',   // RIGHT_BODY = (30, 100, 255) BGR
    centerKp:'#64FFC8',  // CENTER_BODY = (200, 255, 100) BGR
    // Form feedback
    correct:'#00DC00',   // CORRECT = (0, 220, 0) BGR
    incorrect:'#DC1E00', // INCORRECT = (0, 30, 220) BGR
    unknown:'#969696',   // UNKNOWN = (150, 150, 150) BGR
    active: '#F5A34F',   // A custom color for the active joint ring, can be anything bright
};

// =============================================================================
// SECTION 3 — STATE
// =============================================================================

const state = {
    ws:              null,
    stream:          null,
    captureInterval: null,
    currentExercise: 'squat',
    reps:            0,
    lastReps:        0,           // detect rep increment for bump animation
    smoothKpts:      null,        // EMA-smoothed keypoints: [{x,y,conf,visible}, ...]
    trailBuffer:     [],          // last N smoothed keypoint arrays
    evalColors:      {},          // {kpIndex: cssColor} from backend form eval
    connected:       false,
    // --- Video Upload tab ---
    activeTab:       'camera',    // 'camera' | 'upload'
    uploadedFile:    null,        // File object from picker
    uploadedBlobURL: null,        // createObjectURL result
    analysisResults: null,        // full JSON from /analyze-video
    rafId:           null,        // requestAnimationFrame handle
    mirrorEnabled:   true,        // mirror toggle state
};

// =============================================================================
// SECTION 4 — DOM ELEMENTS
// =============================================================================

const video           = document.getElementById('video');
const canvas          = document.getElementById('overlay-canvas');
const ctx             = canvas.getContext('2d');
const startOverlay    = document.getElementById('start-overlay');
const startBtn        = document.getElementById('start-btn');
const statusDot       = document.getElementById('status-dot');
const statusText      = document.getElementById('status-text');
const fpsBadge        = document.getElementById('fps-badge');
const infBadge        = document.getElementById('inf-badge');
const repNumber       = document.getElementById('rep-number');
const repState        = document.getElementById('rep-state');
const formBanner      = document.getElementById('form-banner');
const jointList       = document.getElementById('joint-list');
const exerciseLabel   = document.getElementById('exercise-label');
const detectedLabel   = document.getElementById('detected-label');
const latencyLabel    = document.getElementById('latency-label');
const resetBtn        = document.getElementById('reset-btn');
const exerciseBtns    = document.querySelectorAll('.exercise-btn');
// --- Upload tab DOM ---
const tabBtns         = document.querySelectorAll('.tab-btn');
const uploadOverlay   = document.getElementById('upload-overlay');
const dropZone        = document.getElementById('drop-zone');
const dropZoneFilename= document.getElementById('drop-zone-filename');
const fileInput       = document.getElementById('video-file-input');
const analyzeBtn      = document.getElementById('analyze-btn');
const processingOvl   = document.getElementById('processing-overlay');
const progressFill    = document.getElementById('progress-fill');
const playbackBar     = document.getElementById('playback-bar');
const playbackTime    = document.getElementById('playback-time');
const scrubber        = document.getElementById('playback-scrubber');
const playPauseBtn    = document.getElementById('playback-play-btn');
const mirrorToggleBtn = document.getElementById('mirror-toggle-btn');
const resultBadge     = document.getElementById('result-badge');
const resultFrames    = document.getElementById('result-frames');
const resultReps      = document.getElementById('result-reps');

// =============================================================================
// SECTION 5 — CAMERA SETUP
// =============================================================================

/**
 * Request webcam access and stream it into the <video> element.
 * WHY muted + autoplay? Browsers block autoplay with audio; muted bypasses
 * that restriction. playsinline prevents iOS from going fullscreen.
 */
async function startCamera() {
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false,
        });
        video.srcObject = state.stream;
        await video.play();

        // Sync canvas dimensions to the actual rendered video size
        // WHY here (not in CSS)? Canvas has an internal pixel buffer separate
        // from its CSS display size. Mismatch causes blurry drawing.
        resizeCanvas();
        startOverlay.classList.add('hidden');

        connectWebSocket();
        startCapture();

    } catch (err) {
        alert('Camera access denied or unavailable: ' + err.message);
    }
}

function resizeCanvas() {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
}

// Resize canvas if the window resizes
window.addEventListener('resize', resizeCanvas);

// =============================================================================
// SECTION 6 — WEBSOCKET
// =============================================================================

/**
 * Open a persistent WebSocket connection to the backend.
 * Handles reconnection on unexpected close.
 */
function connectWebSocket() {
    if (state.activeTab === 'upload') return;   // never open WS in upload mode
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;

    state.ws = new WebSocket(CONFIG.WS_URL);
    state.ws.binaryType = 'arraybuffer';   // we send binary, receive text JSON

    state.ws.onopen = () => {
        state.connected = true;
        setStatus('connected', 'Connected');
        // Tell backend which exercise is active
        sendExercise(state.currentExercise);
    };

    state.ws.onclose = () => {
        state.connected = false;
        setStatus('error', 'Disconnected');
        // Only retry in camera mode — not when we intentionally disconnected for upload
        if (state.activeTab === 'camera') setTimeout(connectWebSocket, 2000);
    };

    state.ws.onerror = () => {
        setStatus('error', 'WS Error');
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        } catch (e) {
            console.error('Bad JSON from server:', e);
        }
    };
}

function sendExercise(name) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(`EXERCISE:${name}`);
    }
}

function sendReset() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send('RESET');
        state.reps = 0;
        repNumber.textContent = '0';
    }
}

// =============================================================================
// SECTION 7 — FRAME CAPTURE & SEND
// =============================================================================

/**
 * Capture video frames at CONFIG.CAPTURE_FPS and send as binary JPEG.
 *
 * WHY an offscreen canvas?
 *   Drawing to an offscreen canvas (not displayed) and calling toBlob()
 *   is the most efficient way to encode a JPEG from a <video> in the browser.
 *   It avoids any interaction with the overlay canvas.
 */
const captureCanvas  = document.createElement('canvas');
const captureCtx     = captureCanvas.getContext('2d');
captureCanvas.width  = 640;
captureCanvas.height = 480;

function startCapture() {
    if (state.captureInterval) clearInterval(state.captureInterval);

    state.captureInterval = setInterval(() => {
        if (!state.connected || video.readyState < 2) return;

        // Draw current video frame onto the capture canvas
        captureCtx.drawImage(video, 0, 0, 640, 480);

        // Encode to JPEG blob and send as binary over WebSocket
        // WHY not send every frame? setInterval at 20 fps is plenty;
        // backend YOLOv8 takes ~20-40 ms per frame anyway.
        captureCanvas.toBlob((blob) => {
            if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
            blob.arrayBuffer().then(buf => state.ws.send(buf));
        }, 'image/jpeg', CONFIG.JPEG_QUALITY);

    }, 1000 / CONFIG.CAPTURE_FPS);
}

// =============================================================================
// SECTION 8 — SERVER MESSAGE HANDLER
// =============================================================================

function handleServerMessage(data) {
    if (data.type === 'ack') return;   // control message ACK

    if (data.type === 'error') {
        console.error('[Server error]', data.msg);
        return;
    }

    if (data.type !== 'frame') return;

    // --- Update stats badges ---
    fpsBadge.textContent  = `${data.fps} FPS`;
    infBadge.textContent  = `${data.inf_ms} ms`;
    latencyLabel.textContent = `${data.inf_ms} ms`;

    if (!data.detected) {
        detectedLabel.textContent = 'No person';
        clearCanvas();
        updateFormBanner('unknown', 0);
        return;
    }

    detectedLabel.textContent = '✓ Detected';

    // --- Smooth keypoints (EMA) ---
    state.smoothKpts = applyEMA(state.smoothKpts, data.keypoints);

    // Save to trail buffer
    state.trailBuffer.push(state.smoothKpts.map(kp => ({ ...kp })));
    if (state.trailBuffer.length > CONFIG.TRAIL_FRAMES) {
        state.trailBuffer.shift();
    }

    // --- Build eval color map from joints ---
    // WHY here? Backend sends per-joint status, not per-keypoint-index status.
    // We need to map joint names → COCO indices for the renderer.
    state.evalColors = buildEvalColors(data.joints);

    // --- Draw ---
    drawFrame(state.smoothKpts, state.evalColors);

    // --- Update rep counter ---
    const prevReps = state.reps;
    state.reps = data.reps;
    repNumber.textContent = (data.exercise === 'plank') ? 'HOLD' : data.reps;
    if (data.reps > prevReps) {
        // Rep incremented — play bump animation
        repNumber.classList.remove('bump');
        void repNumber.offsetWidth;   // force reflow to restart animation
        repNumber.classList.add('bump');
        setTimeout(() => repNumber.classList.remove('bump'), 300);
    }

    // --- Form feedback ---
    updateFormBanner(data.overall, data.issues);
    updateJointList(data.joints);
}

// =============================================================================
// SECTION 9 — EMA KEYPOINT SMOOTHING
// =============================================================================

/**
 * Apply Exponential Moving Average to (x, y) positions.
 * alpha: weight of new frame. Lower = smoother but more lag.
 *
 * WHY smooth here AND in Python?
 *   Python EMA smooths the angles (for stable rep counting).
 *   JS EMA smooths the visual positions (for jitter-free canvas drawing).
 *   These are independent concerns with independent tuning.
 */
function applyEMA(prev, curr) {
    if (!prev || prev.length !== curr.length) return curr.map(kp => ({ ...kp }));

    const a = CONFIG.EMA_ALPHA;
    return curr.map((kp, i) => ({
        ...kp,
        x:  a * kp.x  + (1 - a) * prev[i].x,
        y:  a * kp.y  + (1 - a) * prev[i].y,
        // Also smooth normalised coords — the renderer's px() uses xn/yn,
        // so these must be smoothed for the skeleton to move smoothly.
        xn: a * kp.xn + (1 - a) * prev[i].xn,
        yn: a * kp.yn + (1 - a) * prev[i].yn,
    }));
}

// =============================================================================
// SECTION 10 — EVAL COLOR MAP
// =============================================================================

/**
 * Map joint names to their COCO keypoint indices, then store the
 * CSS hex color from the backend's form evaluation.
 *
 * This lets the renderer paint joints green (correct) or red (incorrect)
 * instead of the default anatomy color.
 *
 * JOINT NAME → COCO INDEX mapping (subset; full list in KEYPOINT_NAMES):
 */
const JOINT_TO_KP_IDX = {
    left_shoulder: 5,  right_shoulder: 6,
    left_elbow:    7,  right_elbow:    8,
    left_wrist:    9,  right_wrist:   10,
    left_hip:     11,  right_hip:     12,
    left_knee:    13,  right_knee:    14,
    left_ankle:   15,  right_ankle:   16,
};

function buildEvalColors(joints) {
    const map = {};
    for (const [name, data] of Object.entries(joints)) {
        const idx = JOINT_TO_KP_IDX[name];
        if (idx !== undefined) {
            // Use canonical frontend color constants derived from STATUS,
            // not the BGR-converted hex from the backend (which may differ
            // slightly and break the RANK lookup in boneColor/jointColor).
            if (data.status === 'incorrect') {
                map[idx] = COLORS.incorrect;
            } else if (data.status === 'correct') {
                map[idx] = COLORS.correct;
            }
            // 'unknown' → no entry → jointColor() defaults to COLORS.correct (green)
        }
    }
    return map;
}

// =============================================================================
// SECTION 11 — CANVAS DRAWING
// =============================================================================

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Main draw entry point — called every time a new server frame arrives.
 * Visual rules:
 *  1. Face keypoints (indices 0-4) are never drawn.
 *  2. All body joints colored by form status: green / red / grey.
 *  3. Inactive joints (not in ACTIVE_JOINTS for current exercise) are
 *     drawn at 35% opacity so the relevant joints stand out.
 *  4. Bones take the "worst" color of their two endpoints (red > grey > green).
 */
function drawFrame(keypoints, evalColors) {
    clearCanvas();

    const activeSet = new Set(ACTIVE_JOINTS[state.currentExercise] || []);

    // Every body joint defaults to GREEN unless the backend marks it red.
    // No grey/unknown — strict green/red coaching overlay.
    function jointColor(idx) {
        return evalColors[idx] || COLORS.correct;
    }

    // Bone takes the worst-status color: red beats green.
    const RANK = { [COLORS.incorrect]: 1, [COLORS.correct]: 0 };
    function boneColor(a, b) {
        const ca = jointColor(a), cb = jointColor(b);
        return (RANK[ca] || 0) >= (RANK[cb] || 0) ? ca : cb;
    }

    // Use normalised coordinates (xn, yn ∈ [0,1]) so the overlay works for
    // ANY video resolution — uploaded 1080p, 720p, or the 640×480 webcam feed.
    function px(kp) {
        return {
            x: kp.xn * canvas.width,
            y: kp.yn * canvas.height,
        };
    }

    // --- PASS 1: TRAILS ---
    const trail = state.trailBuffer;
    for (let t = 0; t < trail.length - 1; t++) {
        const alpha = 0.04 + 0.04 * t;
        drawBones(trail[t], alpha, px, false, activeSet, boneColor);
    }

    // --- PASS 2: BONES ---
    drawBones(keypoints, 1.0, px, true, activeSet, boneColor);

    // --- PASS 3: JOINTS ---
    drawJoints(keypoints, activeSet, jointColor, px);
}

/** Draw skeleton bone lines for one set of keypoints. */
function drawBones(keypoints, globalAlpha, px, withShadow, activeSet, boneColorFn) {
    ctx.save();
    ctx.globalAlpha = globalAlpha;

    for (const [a, b, side] of BONES) {
        // Skip bones that touch face keypoints (0-4)
        if (a <= 4 || b <= 4) continue;

        const kpA = keypoints[a];
        const kpB = keypoints[b];
        if (!kpA || !kpB) continue;
        if (kpA.conf < CONFIG.CONF_THRESHOLD || kpB.conf < CONFIG.CONF_THRESHOLD) continue;

        const pA = px(kpA);
        const pB = px(kpB);

        // Determine bone color from form status of endpoints
        const color = boneColorFn ? boneColorFn(a, b) : (COLORS[side] || COLORS.torso);

        // Dim bone if neither endpoint is in the active set
        const isActive = activeSet && (activeSet.has(a) || activeSet.has(b));
        const drawAlpha = isActive ? 1.0 : 0.35;

        if (withShadow) {
            ctx.beginPath();
            ctx.moveTo(pA.x + 1, pA.y + 1);
            ctx.lineTo(pB.x + 1, pB.y + 1);
            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth   = CONFIG.BONE_THICKNESS + 1;
            ctx.lineCap     = 'round';
            ctx.globalAlpha = globalAlpha * drawAlpha;
            ctx.stroke();
        }

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = color;
        ctx.lineWidth   = CONFIG.BONE_THICKNESS;
        ctx.lineCap     = 'round';
        ctx.globalAlpha = globalAlpha * drawAlpha;
        ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    ctx.restore();
}

/** Draw joint circles — strict green/red physiotherapy-style coaching overlay. */
function drawJoints(keypoints, activeSet, jointColorFn, px) {
    for (let i = 0; i < keypoints.length; i++) {
        // Skip face keypoints entirely (indices 0-4)
        if (i <= 4) continue;

        const kp = keypoints[i];
        if (!kp || kp.conf < CONFIG.CONF_THRESHOLD) continue;

        const p        = px(kp);
        const isActive = activeSet.has(i);
        const r        = CONFIG.JOINT_RADIUS + (isActive ? 4 : 0);

        // Strict green/red — no grey, no other anatomy colors
        const baseColor = jointColorFn(i);

        // All joints at full opacity — clean coaching look
        // Active joints get a stronger glow to highlight the evaluated region
        const glowAlpha  = isActive ? 0.45 : 0.18;
        const glowRadius = r * (isActive ? 3 : 2);
        const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowRadius);
        grd.addColorStop(0,   hexToRgba(baseColor, glowAlpha));
        grd.addColorStop(1,   'rgba(0,0,0,0)');
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();

        // Active joint: extra bright ring to mark the evaluated region
        if (isActive) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
            ctx.strokeStyle = hexToRgba(baseColor, 0.85);
            ctx.lineWidth   = 2.5;
            ctx.stroke();
        }

        // Dark outline for separation on any background
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(10,10,10,1)';
        ctx.fill();

        // Main joint circle — full brightness
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = baseColor;
        ctx.fill();

        // Inner highlight for 3D depth
        if (r > 5) {
            const innerR = Math.max(1, Math.floor(r / 3));
            ctx.beginPath();
            ctx.arc(p.x - 1, p.y - 1, innerR, 0, Math.PI * 2);
            ctx.fillStyle = hexToRgba(lighten(baseColor, 0.75), 0.8);
            ctx.fill();
        }
    }
}

// =============================================================================
// SECTION 12 — UI UPDATES
// =============================================================================

function updateFormBanner(overall, issues) {
    formBanner.className = 'form-banner';
    if (overall === 'correct') {
        formBanner.classList.add('correct');
        formBanner.textContent = '✅ FORM: CORRECT';
    } else if (overall === 'incorrect') {
        formBanner.classList.add('incorrect');
        formBanner.textContent = `⚠️ ${issues} ISSUE${issues > 1 ? 'S' : ''} DETECTED`;
    } else {
        formBanner.textContent = '📍 Position yourself in frame';
    }
}

function updateJointList(joints) {
    if (!joints || Object.keys(joints).length === 0) return;

    let html = '<div class="panel-section-title">Joint Feedback</div>';

    for (const [name, data] of Object.entries(joints)) {
        const status = data.status || 'unknown';
        const angle  = data.angle !== null ? `${data.angle}°` : '—';
        const cue    = data.cue   || '';
        const disp   = data.display || name.replace(/_/g, ' ');

        html += `
        <div class="joint-card ${status}">
            <div class="joint-card-header">
                <span class="joint-name">${disp}</span>
                <span class="joint-angle">${angle}</span>
            </div>
            <div class="joint-cue ${status}">${cue}</div>
        </div>`;
    }

    jointList.innerHTML = html;
}

function setStatus(state_, text) {
    statusDot.className  = `status-dot ${state_}`;
    statusText.textContent = text;
}

// =============================================================================
// SECTION 13 — EXERCISE SWITCHING
// =============================================================================

exerciseBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const ex = btn.dataset.exercise;
        if (ex === state.currentExercise) return;

        // Update UI
        exerciseBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update state
        state.currentExercise = ex;
        state.smoothKpts      = null;
        state.trailBuffer     = [];
        state.reps            = 0;
        repNumber.textContent = '0';
        repState.textContent  = 'waiting';
        exerciseLabel.textContent = ex.replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());

        // Notify backend
        sendExercise(ex);
        clearCanvas();
    });
});

resetBtn.addEventListener('click', () => {
    sendReset();
    state.trailBuffer = [];
    clearCanvas();
});

startBtn.addEventListener('click', startCamera);

// =============================================================================
// SECTION 14 — COLOR UTILITIES
// =============================================================================
// (unchanged — hexToRgba and lighten defined below)

/** Convert #rrggbb to rgba(r,g,b,a) for canvas usage. */
function hexToRgba(hex, alpha) {
    if (!hex || hex[0] !== '#') return `rgba(150,150,150,${alpha})`;
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

/** Lighten a hex color toward white by factor (0-1). */
function lighten(hex, factor) {
    if (!hex || hex[0] !== '#') return '#ffffff';
    const r = Math.min(255, Math.round(parseInt(hex.slice(1,3), 16) + (255 - parseInt(hex.slice(1,3), 16)) * factor));
    const g = Math.min(255, Math.round(parseInt(hex.slice(3,5), 16) + (255 - parseInt(hex.slice(3,5), 16)) * factor));
    const b = Math.min(255, Math.round(parseInt(hex.slice(5,7), 16) + (255 - parseInt(hex.slice(5,7), 16)) * factor));
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

// =============================================================================
// SECTION 15 — TAB SWITCHING
// =============================================================================

/**
 * Switch between 'camera' and 'upload' tabs.
 * Stops/starts the relevant systems cleanly on each switch.
 */
function switchTab(tab) {
    if (tab === state.activeTab) return;
    state.activeTab = tab;

    tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));

    if (tab === 'camera') {
        // Stop video playback RAF and hide upload UI
        stopVideoPlayback();
        uploadOverlay.classList.remove('hidden');
        uploadOverlay.classList.add('hidden');   // ensure hidden
        startOverlay.classList.remove('hidden');
        if (state.stream) startOverlay.classList.add('hidden');  // already running
        playbackBar.classList.remove('visible');
        resultBadge.classList.remove('visible');
        setMirror(true);
        // Reconnect WS if stream already live
        if (state.stream) {
            video.srcObject = state.stream;
            connectWebSocket();
            startCapture();
        }
    } else {
        // Upload tab: stop webcam capture & WS
        stopCameraForUpload();
        startOverlay.classList.add('hidden');
        uploadOverlay.classList.remove('hidden');
        setMirror(false);   // uploaded videos shouldn't be mirrored
        clearCanvas();
    }
}

/** Pause camera sending (but keep stream alive so switching back is instant). */
function stopCameraForUpload() {
    if (state.captureInterval) { clearInterval(state.captureInterval); state.captureInterval = null; }
    if (state.ws) { state.ws.close(); state.ws = null; }
    state.connected = false;
    setStatus('', 'Disconnected');
    fpsBadge.textContent = '-- FPS';
    infBadge.textContent = '-- ms';
}

tabBtns.forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

// =============================================================================
// SECTION 16 — MIRROR TOGGLE
// =============================================================================

function setMirror(enabled) {
    state.mirrorEnabled = enabled;
    video.classList.toggle('no-mirror', !enabled);
    canvas.classList.toggle('no-mirror', !enabled);
    mirrorToggleBtn.style.opacity = enabled ? '1' : '0.5';
}

mirrorToggleBtn.addEventListener('click', () => setMirror(!state.mirrorEnabled));

// =============================================================================
// SECTION 17 — FILE PICKER & DRAG-AND-DROP
// =============================================================================

/** Called whenever a new file is chosen (via picker or drop). */
function onFileSelected(file) {
    if (!file || !file.type.startsWith('video/')) {
        alert('Please select a valid video file (MP4, MOV, AVI, WebM).');
        return;
    }
    state.uploadedFile = file;
    // Revoke previous blob URL to free memory
    if (state.uploadedBlobURL) URL.revokeObjectURL(state.uploadedBlobURL);
    state.uploadedBlobURL = URL.createObjectURL(file);

    // Show filename
    dropZoneFilename.textContent = file.name;
    dropZoneFilename.classList.remove('hidden');
    analyzeBtn.disabled = false;
    state.analysisResults = null;
    resultBadge.classList.remove('visible');
    playbackBar.classList.remove('visible');
    clearCanvas();
}

// Click on drop-zone → open file picker
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) onFileSelected(fileInput.files[0]);
});

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) onFileSelected(file);
});

// =============================================================================
// SECTION 18 — VIDEO ANALYSIS (POST to /analyze-video)
// =============================================================================

analyzeBtn.addEventListener('click', async () => {
    if (!state.uploadedFile) return;

    // Show spinner, hide upload UI
    analyzeBtn.disabled = true;
    uploadOverlay.classList.add('hidden');
    processingOvl.classList.remove('hidden');
    progressFill.style.width = '15%';
    clearCanvas();
    stopVideoPlayback();

    const formData = new FormData();
    formData.append('file', state.uploadedFile);
    formData.append('exercise', state.currentExercise);

    // Animate progress bar while waiting
    let fakeProgress = 15;
    const progressInterval = setInterval(() => {
        fakeProgress = Math.min(fakeProgress + Math.random() * 4, 88);
        progressFill.style.width = fakeProgress + '%';
    }, 400);

    try {
        const res = await fetch('/analyze-video', { method: 'POST', body: formData });
        clearInterval(progressInterval);

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Server error');
        }

        progressFill.style.width = '100%';
        const data = await res.json();
        state.analysisResults = data;

        // Brief pause so the 100% fill is visible
        await new Promise(r => setTimeout(r, 300));
        processingOvl.classList.add('hidden');

        // Update result badge
        const maxReps = Math.max(...data.results.map(f => f.reps), 0);
        resultFrames.textContent = data.results.length;
        resultReps.textContent   = maxReps;
        resultBadge.classList.add('visible');

        // Load video into the player and wait for metadata before rendering
        video.srcObject = null;
        video.src       = state.uploadedBlobURL;
        video.muted     = true;
        video.loop      = false;
        video.autoplay  = false;   // we control playback manually below
        video.currentTime = 0;

        // Wait for the browser to decode the video header so
        // video.videoWidth / videoHeight are available for canvas sizing.
        await new Promise(resolve => {
            if (video.readyState >= 1) { resolve(); return; }
            video.onloadedmetadata = resolve;
        });

        // Pause any premature playback from the HTML autoplay attr,
        // then reset to the start so we begin from frame 0.
        video.pause();
        video.currentTime = 0;

        resizeCanvas();

        // Reset smoothing state so first frame renders at true position
        state.smoothKpts  = null;
        state.trailBuffer = [];

        // Set scrubber max
        const duration = data.total_frames / data.fps;
        scrubber.max = duration;
        scrubber.value = 0;
        playbackBar.classList.add('visible');

        // Pre-render first frame overlay
        renderFrameAtTime(0, data);

        detectedLabel.textContent = '✓ Ready';
        latencyLabel.textContent  = `${data.results.length} frames`;

        // Auto-start playback + overlay loop together so the video
        // plays immediately after analysis with the skeleton tracking.
        video.play().then(() => {
            playPauseBtn.textContent = '⏸ Pause';
            if (state.rafId) cancelAnimationFrame(state.rafId);
            state.rafId = requestAnimationFrame(videoPlaybackLoop);
        }).catch(() => {
            // Autoplay blocked by browser — fall back to manual play
            playPauseBtn.textContent = '▶ Play';
        });

    } catch (err) {
        clearInterval(progressInterval);
        processingOvl.classList.add('hidden');
        uploadOverlay.classList.remove('hidden');
        analyzeBtn.disabled = false;
        alert('Analysis failed: ' + err.message);
    }
});

// =============================================================================
// SECTION 19 — VIDEO PLAYBACK SYNC
// =============================================================================

/** Find the closest analysis result for a given video timestamp. */
function findResultAtTime(timeS, results) {
    // Binary-search-ish: results are sorted by time_s
    let lo = 0, hi = results.length - 1, best = 0;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (results[mid].time_s <= timeS) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return results[best];
}

/** Render the skeleton and UI for the result closest to `timeS`. */
function renderFrameAtTime(timeS, data) {
    const r = findResultAtTime(timeS, data.results);
    if (!r) return;

    if (!r.detected) {
        clearCanvas();
        updateFormBanner('unknown', 0);
        detectedLabel.textContent = 'No person';
        return;
    }

    detectedLabel.textContent = '✓ Detected';
    // EMA smooth
    state.smoothKpts = applyEMA(state.smoothKpts, r.keypoints);
    state.trailBuffer.push(state.smoothKpts.map(kp => ({ ...kp })));
    if (state.trailBuffer.length > CONFIG.TRAIL_FRAMES) state.trailBuffer.shift();

    state.evalColors = buildEvalColors(r.joints);
    drawFrame(state.smoothKpts, state.evalColors);

    // Rep counter
    const prevReps = state.reps;
    state.reps = r.reps;
    repNumber.textContent = (data.exercise === 'plank') ? 'HOLD' : r.reps;
    if (r.reps > prevReps) {
        repNumber.classList.remove('bump');
        void repNumber.offsetWidth;
        repNumber.classList.add('bump');
        setTimeout(() => repNumber.classList.remove('bump'), 300);
    }

    updateFormBanner(r.overall, r.issues);
    updateJointList(r.joints);
}

/** RAF loop that runs while the video is active (playing OR scrubbing). */
function videoPlaybackLoop() {
    if (!state.analysisResults || state.activeTab !== 'upload') {
        state.rafId = null;
        return;
    }

    // Always render the current frame (works during play AND pause/seek)
    renderFrameAtTime(video.currentTime, state.analysisResults);

    // Update scrubber + time display
    const dur = video.duration || 1;
    scrubber.value = video.currentTime;
    const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    playbackTime.textContent = `${fmt(video.currentTime)} / ${fmt(dur)}`;

    if (video.ended) {
        // Video finished — stop the loop and reset button
        playPauseBtn.textContent = '▶ Play';
        state.rafId = null;
    } else {
        // Keep looping (the loop itself is always active while video is loaded)
        state.rafId = requestAnimationFrame(videoPlaybackLoop);
    }
}

function stopVideoPlayback() {
    if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = null; }
    if (!video.paused) video.pause();
}

// Play / Pause button
playPauseBtn.addEventListener('click', () => {
    if (!state.analysisResults) return;

    // If paused or ended → start/resume playback
    if (video.paused || video.ended) {
        if (video.ended) {
            video.currentTime = 0;
            state.smoothKpts  = null;
            state.trailBuffer = [];
        }
        // Start loop AFTER play() promise resolves so video.paused is false
        video.play().then(() => {
            playPauseBtn.textContent = '⏸ Pause';
            // Always ensure the RAF loop is running when playing
            if (state.rafId) cancelAnimationFrame(state.rafId);
            state.rafId = requestAnimationFrame(videoPlaybackLoop);
        }).catch(err => console.error('[Video] play() failed:', err));
    } else {
        // Playing → pause, but keep RAF alive for scrub-while-paused
        video.pause();
        playPauseBtn.textContent = '▶ Play';
    }
});

// Scrubber seek
scrubber.addEventListener('input', () => {
    if (!state.analysisResults) return;
    video.currentTime = parseFloat(scrubber.value);
    state.smoothKpts  = null;   // reset EMA on seek
    state.trailBuffer = [];
    renderFrameAtTime(video.currentTime, state.analysisResults);
});

// Resume RAF after seeking while playing
video.addEventListener('seeked', () => {
    if (!video.paused && state.activeTab === 'upload') {
        if (state.rafId) cancelAnimationFrame(state.rafId);
        state.rafId = requestAnimationFrame(videoPlaybackLoop);
    }
});

