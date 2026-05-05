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
    bicep_curl:     [7,8,9,10],
    squat:          [11,12,13,14],
    lateral_raise:  [5,6,7,8],
    push_up:        [7,8,11,12],
    shoulder_press: [5,6,7,8],
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
    active: '#4FA3F5',   // active joint ring highlight
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
};

// =============================================================================
// SECTION 4 — DOM ELEMENTS
// =============================================================================

const video        = document.getElementById('video');
const canvas       = document.getElementById('overlay-canvas');
const ctx          = canvas.getContext('2d');
const startOverlay = document.getElementById('start-overlay');
const startBtn     = document.getElementById('start-btn');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const fpsBadge     = document.getElementById('fps-badge');
const infBadge     = document.getElementById('inf-badge');
const repNumber    = document.getElementById('rep-number');
const repState     = document.getElementById('rep-state');
const formBanner   = document.getElementById('form-banner');
const jointList    = document.getElementById('joint-list');
const exerciseLabel= document.getElementById('exercise-label');
const detectedLabel= document.getElementById('detected-label');
const latencyLabel = document.getElementById('latency-label');
const resetBtn     = document.getElementById('reset-btn');
const exerciseBtns = document.querySelectorAll('.exercise-btn');

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
        // Retry after 2 seconds
        setTimeout(connectWebSocket, 2000);
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
        x: a * kp.x + (1 - a) * prev[i].x,
        y: a * kp.y + (1 - a) * prev[i].y,
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
            map[idx] = data.color;
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
 * Rendering order (painter's algorithm):
 *   1. Ghost trails (oldest → newest, fading)
 *   2. Bone lines (with shadow)
 *   3. Joint circles (on top of bones, with glow for active joints)
 */
function drawFrame(keypoints, evalColors) {
    clearCanvas();

    const activeSet = new Set(ACTIVE_JOINTS[state.currentExercise] || []);
    const scaleX = canvas.width;
    const scaleY = canvas.height;

    // px coordinates (already absolute from backend, just use them)
    // WHY not use normalised xn/yn? Pixel coords are exact; normalised needs
    // multiplication which introduces floating-point rounding at display size.
    function px(kp) {
        return { x: kp.x * (canvas.width / 640), y: kp.y * (canvas.height / 480) };
    }

    // --- PASS 1: TRAILS ---
    const trail = state.trailBuffer;
    for (let t = 0; t < trail.length - 1; t++) {
        const alpha = 0.04 + 0.05 * t;   // older = more transparent
        drawBones(trail[t], alpha, px, false);
    }

    // --- PASS 2: BONES ---
    drawBones(keypoints, 1.0, px, true);

    // --- PASS 3: JOINTS ---
    drawJoints(keypoints, activeSet, evalColors, px);
}

/** Draw skeleton bone lines for one set of keypoints. */
function drawBones(keypoints, globalAlpha, px, withShadow) {
    ctx.save();
    ctx.globalAlpha = globalAlpha;

    for (const [a, b, side] of BONES) {
        const kpA = keypoints[a];
        const kpB = keypoints[b];
        if (!kpA || !kpB) continue;
        if (kpA.conf < CONFIG.CONF_THRESHOLD || kpB.conf < CONFIG.CONF_THRESHOLD) continue;

        const pA = px(kpA);
        const pB = px(kpB);

        if (withShadow) {
            // Shadow: 1px offset, darker, 1px thicker — adds depth
            ctx.beginPath();
            ctx.moveTo(pA.x + 1, pA.y + 1);
            ctx.lineTo(pB.x + 1, pB.y + 1);
            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth   = CONFIG.BONE_THICKNESS + 1;
            ctx.lineCap     = 'round';
            ctx.stroke();
        }

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = COLORS[side] || COLORS.torso;
        ctx.lineWidth   = CONFIG.BONE_THICKNESS;
        ctx.lineCap     = 'round';
        ctx.stroke();
    }
    ctx.restore();
}

/** Draw joint circles with glow and confidence-based opacity. */
function drawJoints(keypoints, activeSet, evalColors, px) {
    for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        if (!kp || kp.conf < CONFIG.CONF_THRESHOLD) continue;

        const p    = px(kp);
        const group = KP_GROUP[i] || 'center';
        const isActive = activeSet.has(i);
        const r = CONFIG.JOINT_RADIUS + (isActive ? 3 : 0);

        // Confidence-based opacity: conf=1.0 → full color; conf=0.5 → dimmed
        // Matches renderer.py: opacity_factor = 0.55 + 0.45 * conf
        const opacity = 0.55 + 0.45 * kp.conf;

        // Form-eval color override, else anatomy color
        let baseColor = evalColors[i] || (
            group === 'left'  ? COLORS.leftKp :
            group === 'right' ? COLORS.rightKp :
            group === 'center' ? COLORS.centerKp :
            COLORS.face
        );

        // --- GLOW (only for body joints, not face) ---
        if (group !== 'face') {
            const glowRadius = r * 3;
            const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowRadius);
            grd.addColorStop(0,   hexToRgba(baseColor, CONFIG.GLOW_ALPHA));
            grd.addColorStop(1,   'rgba(0,0,0,0)');
            ctx.beginPath();
            ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
            ctx.fillStyle = grd;
            ctx.fill();
        }

        // --- ACTIVE RING (bright outline for emphasized joints) ---
        if (isActive) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
            ctx.strokeStyle = COLORS.active;
            ctx.lineWidth   = 2;
            ctx.globalAlpha = 0.8;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }

        // --- OUTLINE RING (dark ring for contrast on any background) ---
        // Matches renderer.py: dark (20, 20, 20) outline at radius + 2
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(20,20,20,1)';
        ctx.fill();

        // --- MAIN CIRCLE (colored keypoint dot with opacity from confidence) ---
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(baseColor, opacity);
        ctx.fill();

        // --- INNER HIGHLIGHT (3D effect, positioned at -1,-1 offset) ---
        // Matches renderer.py: inner radius = radius // 3, at (x-1, y-1), with glow at 0.8 factor
        if (r > 5) {
            const innerR = Math.max(1, Math.floor(r / 3));
            const highlightColor = lighten(baseColor, 0.8);
            ctx.beginPath();
            ctx.arc(p.x - 1, p.y - 1, innerR, 0, Math.PI * 2);
            ctx.fillStyle = hexToRgba(highlightColor, 0.7);
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
