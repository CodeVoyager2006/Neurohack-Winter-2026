// ═══════════════════════════════════════════════════════════════
// Landmark indices — Eye Aspect Ratio (EAR) system
//
// EAR = mean(v1, v2, v3) / eye_width
//   v1..v3 = three vertical distances between matched upper/lower lid pairs
//   eye_width = horizontal corner-to-corner distance (normalization)
//
// Because the brow directly pulls the upper eyelid up/down, EAR captures:
//   • Brow RAISED  → upper lid lifts   → EAR increases → LOOKING_UP
//   • Brow LOWERED → upper lid presses → EAR decreases → LOOKING_DOWN
//
// This gives far better dynamic range for downward detection than raw brow distance.
//
// Left eye  upper/lower pairs: (160,144) (159,145) (158,153)   corners: 33, 133
// Right eye upper/lower pairs: (387,373) (386,374) (385,380)   corners: 263, 362
// ═══════════════════════════════════════════════════════════════
const L_EAR_UPPER   = [160, 159, 158];
const L_EAR_LOWER   = [144, 145, 153];
const L_EYE_CORNERS = [33,  133];
const R_EAR_UPPER   = [387, 386, 385];
const R_EAR_LOWER   = [373, 374, 380];
const R_EYE_CORNERS = [263, 362];

const LOOKING_UP   = 'LOOKING_UP';
const LOOKING_DOWN = 'LOOKING_DOWN';
const NEUTRAL      = 'NEUTRAL';

// ═══════════════════════════════════════════════════════════════
// App state
// ═══════════════════════════════════════════════════════════════
const state = {
  eyeSelection: 'both',
  // Thresholds are now 0–1 iris ratios:
  //   ratio < lower → LOOKING_UP (iris high in eye)
  //   ratio > upper → LOOKING_DOWN (iris low in eye)
  // EAR defaults — typical neutral ~0.25-0.30, open ~0.35+, narrow ~0.15-0.20
  upper: { left: 0.32, right: 0.32 },   // above → LOOKING_UP  (eyes wide / brow raised)
  lower: { left: 0.20, right: 0.20 },   // below → LOOKING_DOWN (eyes narrow / brow lowered)
  calibrating: false,
  calibSamples: { left: [], right: [] },
  calibTimer: null,
  cameraOk: false,
  bciConnected: false,
  bciPort: null,
  bciReader: null,
  recording: false,
  recordingStart: 0,
  rows: [],
  latestCamera: {
    leftRatio: null,
    rightRatio: null,
    actionLeft: NEUTRAL,
    actionRight: NEUTRAL,
  },
};

// ═══════════════════════════════════════════════════════════════
// Status helpers
// ═══════════════════════════════════════════════════════════════
function setStatus(msg, isError = false) {
  const bar = document.getElementById('statusBar');
  bar.textContent = msg;
  bar.className = isError ? 'error' : '';
}

// ═══════════════════════════════════════════════════════════════
// Eye Aspect Ratio (EAR)
// EAR = mean(v1, v2, v3) / eye_width
//   High EAR (~0.30–0.40) = eye wide open  → brow raised   → LOOKING_UP
//   Low  EAR (~0.10–0.20) = eye narrowed   → brow lowered  → LOOKING_DOWN
//   Returns null if eye width is degenerate (< 5 px)
// ═══════════════════════════════════════════════════════════════
function computeEAR(lm, upperIds, lowerIds, cornerIds, W, H) {
  // Three vertical distances between matched upper/lower pairs
  let vertSum = 0;
  for (let i = 0; i < upperIds.length; i++) {
    vertSum += Math.abs(lm[upperIds[i]].y * H - lm[lowerIds[i]].y * H);
  }
  const meanVert = vertSum / upperIds.length;

  // Horizontal eye width (corner to corner)
  const c0 = cornerIds[0], c1 = cornerIds[1];
  const dx = (lm[c0].x - lm[c1].x) * W;
  const dy = (lm[c0].y - lm[c1].y) * H;
  const eyeWidth = Math.sqrt(dx * dx + dy * dy);
  if (eyeWidth < 5) return null;

  return meanVert / eyeWidth;
}

// ═══════════════════════════════════════════════════════════════
// Classification
// brow_lift > upper → brow raised   → LOOKING_UP
// brow_lift < lower → brow lowered  → LOOKING_DOWN
// ═══════════════════════════════════════════════════════════════
function classify(ratio, upper, lower) {
  if (ratio === null)  return NEUTRAL;
  if (ratio > upper)   return LOOKING_UP;
  if (ratio < lower)   return LOOKING_DOWN;
  return NEUTRAL;
}

// ═══════════════════════════════════════════════════════════════
// MediaPipe Face Mesh
// ═══════════════════════════════════════════════════════════════
const canvas = document.getElementById('videoCanvas');
const ctx    = canvas.getContext('2d');
const video  = document.getElementById('hiddenVideo');

const faceMesh = new FaceMesh({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,       // required for iris landmarks 468–477
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

faceMesh.onResults(onFaceMeshResults);

function onFaceMeshResults(results) {
  // Draw mirrored video frame
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  let leftRatio  = null;
  let rightRatio = null;

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const lm = results.multiFaceLandmarks[0];
    const W  = canvas.width;
    const H  = canvas.height;
    const sel = state.eyeSelection;

    if (sel === 'left' || sel === 'both') {
      leftRatio = computeEAR(lm, L_EAR_UPPER, L_EAR_LOWER, L_EYE_CORNERS, W, H);
      if (leftRatio !== null) {
        drawEAROverlay(lm, L_EAR_UPPER, L_EAR_LOWER, L_EYE_CORNERS, W, H,
                       leftRatio, state.upper.left, state.lower.left);
      }
    }

    if (sel === 'right' || sel === 'both') {
      rightRatio = computeEAR(lm, R_EAR_UPPER, R_EAR_LOWER, R_EYE_CORNERS, W, H);
      if (rightRatio !== null) {
        drawEAROverlay(lm, R_EAR_UPPER, R_EAR_LOWER, R_EYE_CORNERS, W, H,
                       rightRatio, state.upper.right, state.lower.right);
      }
    }

    if (state.calibrating) {
      if (leftRatio  !== null) state.calibSamples.left.push(leftRatio);
      if (rightRatio !== null) state.calibSamples.right.push(rightRatio);
    }
  }

  const actionLeft  = classify(leftRatio,  state.upper.left,  state.lower.left);
  const actionRight = classify(rightRatio, state.upper.right, state.lower.right);

  drawHUD(leftRatio, rightRatio, actionLeft, actionRight);

  state.latestCamera = { leftRatio, rightRatio, actionLeft, actionRight };
  updateReadout(leftRatio, rightRatio, actionLeft, actionRight);

  if (state.recording && !state.bciConnected) {
    pushRow(leftRatio, rightRatio, actionLeft, actionRight, null);
  }
}

// Draw EAR landmarks and a live gauge bar for one eye
function drawEAROverlay(lm, upperIds, lowerIds, cornerIds, W, H, ear, upper, lower) {
  function mx(idx) { return (1 - lm[idx].x) * W; }
  function my(idx) { return lm[idx].y * H; }

  const upperPts = upperIds.map(i => ({ x: mx(i), y: my(i) }));
  const lowerPts = lowerIds.map(i => ({ x: mx(i), y: my(i) }));
  const c0 = { x: mx(cornerIds[0]), y: my(cornerIds[0]) };
  const c1 = { x: mx(cornerIds[1]), y: my(cornerIds[1]) };

  const eyeWidth = Math.sqrt((c0.x - c1.x) ** 2 + (c0.y - c1.y) ** 2);
  const eyeCentX = (c0.x + c1.x) / 2;
  const eyeCentY = (c0.y + c1.y) / 2;

  // Corner dots (blue)
  drawDot(c0.x, c0.y, '#5599ff', 3);
  drawDot(c1.x, c1.y, '#5599ff', 3);

  // Three vertical measurement lines between matched pairs
  for (let i = 0; i < upperIds.length; i++) {
    const u = upperPts[i], l = lowerPts[i];
    // Upper lid dot (green), lower lid dot (red)
    drawDot(u.x, u.y, '#00ff88', 3);
    drawDot(l.x, l.y, '#ff4466', 3);
    // Vertical measurement line (white semi-transparent)
    ctx.beginPath();
    ctx.moveTo(u.x, u.y);
    ctx.lineTo(l.x, l.y);
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Horizontal eye-width line (blue semi-transparent)
  ctx.beginPath();
  ctx.moveTo(c0.x, c0.y);
  ctx.lineTo(c1.x, c1.y);
  ctx.strokeStyle = 'rgba(85,153,255,0.5)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Vertical EAR gauge bar (placed just outside the eye)
  const barH   = eyeWidth * 0.9;
  const barX   = eyeCentX + eyeWidth * 0.58;
  const barTop = eyeCentY - barH / 2;
  const maxEAR = 0.45;   // upper display limit

  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fillRect(barX, barTop, 5, barH);

  // Neutral band (green)
  const bandTop    = barTop + (1 - upper / maxEAR) * barH;
  const bandBottom = barTop + (1 - lower / maxEAR) * barH;
  ctx.fillStyle = 'rgba(100,220,100,0.35)';
  ctx.fillRect(barX, bandTop, 5, bandBottom - bandTop);

  // EAR indicator (yellow tick)
  const clamped = Math.max(0, Math.min(maxEAR, ear));
  const tickY   = barTop + (1 - clamped / maxEAR) * barH;
  ctx.fillStyle = '#ffe066';
  ctx.fillRect(barX - 2, tickY - 2, 9, 4);
}

function drawDot(x, y, color, r = 4) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawHUD(lRatio, rRatio, aLeft, aRight) {
  const colorMap = { LOOKING_UP: '#00ff88', LOOKING_DOWN: '#ff4466', NEUTRAL: '#aaaaaa' };
  const sel = state.eyeSelection;
  let y = 24;
  ctx.font = '600 13px Consolas, monospace';

  if (sel === 'left' || sel === 'both') {
    const rStr = lRatio !== null ? lRatio.toFixed(3) : 'N/A';
    ctx.fillStyle = colorMap[aLeft] || '#fff';
    ctx.fillText(`LEFT  EAR=${rStr}  ${aLeft}`, 10, y);
    y += 22;
  }
  if (sel === 'right' || sel === 'both') {
    const rStr = rRatio !== null ? rRatio.toFixed(3) : 'N/A';
    ctx.fillStyle = colorMap[aRight] || '#fff';
    ctx.fillText(`RIGHT EAR=${rStr}  ${aRight}`, 10, y);
    y += 22;
  }
  ctx.fillStyle = '#aaa';
  ctx.font = '11px Consolas, monospace';
  ctx.fillText(`Eye: ${sel.toUpperCase()}`, 10, y);

  if (state.calibrating) {
    const banner = '  CALIBRATING...  ';
    ctx.font = 'bold 16px Consolas, monospace';
    const tw = ctx.measureText(banner).width;
    const bx = (canvas.width - tw) / 2;
    const by = canvas.height - 24;
    ctx.fillStyle = 'rgba(0,140,255,0.55)';
    ctx.fillRect(bx - 8, by - 20, tw + 16, 30);
    ctx.fillStyle = '#fff';
    ctx.fillText(banner, bx, by);
  }
}

// ─── Camera start ──────────────────────────────────────────────
(function startCamera() {
  const camera = new Camera(video, {
    onFrame: async () => {
      await faceMesh.send({ image: video });
    },
    width: 640,
    height: 480,
  });

  camera.start()
    .then(() => {
      state.cameraOk = true;
      document.getElementById('camStatus').textContent = 'Camera active';
      document.getElementById('camStatus').className = 'cam-status ok';
      document.getElementById('headerDot').style.background = '#a6e3a1';
      setStatus('Camera: OK  |  OpenBCI: disconnected  |  Idle');
    })
    .catch((err) => {
      state.cameraOk = false;
      document.getElementById('camStatus').textContent = 'Camera error: ' + err.message;
      document.getElementById('camStatus').className = 'cam-status err';
      document.getElementById('headerDot').style.background = '#f38ba8';
      setStatus('ERROR: Camera not available — ' + err.message, true);
    });
})();

// ═══════════════════════════════════════════════════════════════
// Eye selection
// ═══════════════════════════════════════════════════════════════
function onEyeChange() {
  const val = document.querySelector('input[name="eyeSel"]:checked').value;
  state.eyeSelection = val;
}

// ═══════════════════════════════════════════════════════════════
// Calibration
// ═══════════════════════════════════════════════════════════════
function startCalibration() {
  if (state.calibrating) return;
  state.calibrating = true;
  state.calibSamples = { left: [], right: [] };

  const btn = document.getElementById('btnCalib');
  btn.disabled = true;
  btn.innerHTML = 'Calibrating… <span class="spinner"></span>';

  const result = document.getElementById('calibResult');
  result.textContent = 'Calibrating — look straight ahead…';
  result.className = '';

  state.calibTimer = setTimeout(() => {
    for (const eye of ['left', 'right']) {
      const samples = state.calibSamples[eye];
      if (samples.length >= 2) {
        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
        const std  = Math.sqrt(
          samples.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / samples.length
        );
        state.upper[eye] = Math.min(0.99, mean + 2.0 * std);
        state.lower[eye] = Math.max(0.01, mean - 2.0 * std);
      }
    }
    state.calibrating = false;
    btn.disabled = false;
    btn.innerHTML = 'Calibrate (3 s) — look straight ahead';

    const l = state.upper.left,  ll = state.lower.left;
    const r = state.upper.right, rl = state.lower.right;
    result.textContent =
      `L: down>${l.toFixed(3)}  up<${ll.toFixed(3)}  |  R: down>${r.toFixed(3)}  up<${rl.toFixed(3)}`;
    result.className = 'ok';
    setStatus('Calibration complete.');
  }, 3000);
}

// ═══════════════════════════════════════════════════════════════
// Manual threshold override
// ═══════════════════════════════════════════════════════════════
function applyThresholds() {
  const eye   = document.getElementById('threshEye').value;
  const upper = parseFloat(document.getElementById('threshUpper').value);
  const lower = parseFloat(document.getElementById('threshLower').value);

  if (isNaN(upper) || isNaN(lower)) {
    alert('Upper and Lower thresholds must be numbers.');
    return;
  }
  if (upper <= lower) {
    alert(`Upper (${upper}) must be greater than Lower (${lower}).`);
    return;
  }
  state.upper[eye] = upper;
  state.lower[eye] = lower;
  setStatus(`Manual thresholds applied — ${eye}: down>${upper.toFixed(3)} up<${lower.toFixed(3)}`);
}

// ═══════════════════════════════════════════════════════════════
// Live readout
// ═══════════════════════════════════════════════════════════════
function actionClass(action) {
  if (action === LOOKING_UP)   return 'action-up';
  if (action === LOOKING_DOWN) return 'action-down';
  return 'action-neutral';
}

function updateReadout(lRatio, rRatio, aLeft, aRight) {
  const lStr = lRatio !== null ? lRatio.toFixed(3) : '—';
  const rStr = rRatio !== null ? rRatio.toFixed(3) : '—';
  document.getElementById('readoutDist').innerHTML =
    `EAR — L: <span class="val">${lStr}</span>  R: <span class="val">${rStr}</span>`;
  document.getElementById('readoutAction').innerHTML =
    `Action — L: <span class="${actionClass(aLeft)}">${aLeft}</span>  ` +
    `R: <span class="${actionClass(aRight)}">${aRight}</span>`;
}

// ═══════════════════════════════════════════════════════════════
// OpenBCI — Web Serial API
// ═══════════════════════════════════════════════════════════════
async function refreshPorts() {
  if (!('serial' in navigator)) {
    document.getElementById('bciStatus').textContent =
      'Web Serial API not available. Use Chrome or Edge.';
    return;
  }
  try {
    const ports = await navigator.serial.getPorts();
    const sel   = document.getElementById('portSelect');
    sel.innerHTML = '<option value="">— select port —</option>';
    ports.forEach((p, i) => {
      const info = p.getInfo();
      const opt  = document.createElement('option');
      opt.value  = i;
      opt._port  = p;
      opt.textContent =
        `Serial Port ${i} (VID:${info.usbVendorId ?? '?'} PID:${info.usbProductId ?? '?'})`;
      sel.appendChild(opt);
    });
    if (ports.length === 0) {
      document.getElementById('bciStatus').textContent =
        'No paired ports. Click Refresh to grant access.';
      try {
        const newPort = await navigator.serial.requestPort();
        const opt = document.createElement('option');
        opt.value = 'new';
        opt._port = newPort;
        opt.textContent = 'Selected port';
        opt.selected = true;
        sel.appendChild(opt);
      } catch (_) { /* cancelled */ }
    }
  } catch (e) {
    document.getElementById('bciStatus').textContent = 'Error: ' + e.message;
  }
}

async function toggleOpenBCI() {
  const btn = document.getElementById('btnConnect');
  if (state.bciConnected) {
    await disconnectBCI();
    btn.textContent = 'Connect';
    btn.className   = 'btn-primary btn-full';
    document.getElementById('bciStatus').textContent = 'Disconnected.';
    state.bciConnected = false;
  } else {
    const sel = document.getElementById('portSelect');
    const opt = sel.options[sel.selectedIndex];
    if (!opt || !opt._port) {
      alert('Please select a COM port first. Click Refresh to pair one.');
      return;
    }
    try {
      const port = opt._port;
      await port.open({ baudRate: 115200 });
      state.bciPort      = port;
      state.bciConnected = true;
      btn.textContent    = 'Disconnect';
      btn.className      = 'btn-danger btn-full';
      document.getElementById('bciStatus').textContent = 'Connected — streaming.';
      setStatus('OpenBCI connected, streaming.');
      startBCIStream(port);
    } catch (e) {
      setStatus('OpenBCI connection failed: ' + e.message, true);
      document.getElementById('bciStatus').textContent = 'Connection failed: ' + e.message;
    }
  }
}

async function disconnectBCI() {
  try {
    if (state.bciReader) {
      await state.bciReader.cancel();
      state.bciReader = null;
    }
    if (state.bciPort && state.bciPort.readable) {
      await state.bciPort.close();
    }
  } catch (_) {}
  state.bciPort = null;
}

// Cyton packet: 33 bytes, start=0xA0, end=0xC0
// Channels 1–8: bytes 2–25, 3 bytes per channel (24-bit signed big-endian)
// Scale: 4.5 V / (2^23 - 1) / 24 gain * 1e6 → µV
const CYTON_SCALE       = (4.5 / (Math.pow(2, 23) - 1) / 24) * 1e6;
const CYTON_PACKET_SIZE = 33;

async function startBCIStream(port) {
  const reader = port.readable.getReader();
  state.bciReader = reader;

  const buf = new Uint8Array(CYTON_PACKET_SIZE * 4);
  let bufLen = 0;

  try {
    while (state.bciConnected) {
      const { value, done } = await reader.read();
      if (done) break;

      for (const byte of value) {
        buf[bufLen++] = byte;
        if (bufLen >= CYTON_PACKET_SIZE * 4) bufLen = 0;
      }

      let i = 0;
      while (i + CYTON_PACKET_SIZE <= bufLen) {
        if (buf[i] === 0xA0 && buf[i + CYTON_PACKET_SIZE - 1] === 0xC0) {
          const channels = [];
          for (let ch = 0; ch < 8; ch++) {
            const b0 = buf[i + 2 + ch * 3];
            const b1 = buf[i + 3 + ch * 3];
            const b2 = buf[i + 4 + ch * 3];
            let val = (b0 << 16) | (b1 << 8) | b2;
            if (val & 0x800000) val -= 0x1000000;
            channels.push(val * CYTON_SCALE);
          }
          const sample = {
            EMG_1: channels[0], EMG_2: channels[1],
            EMG_3: channels[2], EMG_4: channels[3],
            EEG_Frontal_1: channels[4], EEG_Frontal_2: channels[5],
            EEG_Behind_Left_Ear: channels[6], EEG_Behind_Right_Ear: channels[7],
          };
          if (state.recording) {
            const cam = state.latestCamera;
            pushRow(cam.leftRatio, cam.rightRatio, cam.actionLeft, cam.actionRight, sample);
          }
          i += CYTON_PACKET_SIZE;
        } else {
          i++;
        }
      }
      if (i > 0) {
        buf.copyWithin(0, i, bufLen);
        bufLen -= i;
      }
    }
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════
// Recording
// ═══════════════════════════════════════════════════════════════
function pushRow(lRatio, rRatio, aLeft, aRight, bci) {
  state.rows.push({
    timestamp:           (performance.now() - state.recordingStart) / 1000,
    ear_left:            lRatio !== null ? lRatio : NaN,
    ear_right:           rRatio !== null ? rRatio : NaN,
    action_left:          aLeft,
    action_right:         aRight,
    EMG_1:                bci ? bci.EMG_1                : NaN,
    EMG_2:                bci ? bci.EMG_2                : NaN,
    EMG_3:                bci ? bci.EMG_3                : NaN,
    EMG_4:                bci ? bci.EMG_4                : NaN,
    EEG_Frontal_1:        bci ? bci.EEG_Frontal_1        : NaN,
    EEG_Frontal_2:        bci ? bci.EEG_Frontal_2        : NaN,
    EEG_Behind_Left_Ear:  bci ? bci.EEG_Behind_Left_Ear  : NaN,
    EEG_Behind_Right_Ear: bci ? bci.EEG_Behind_Right_Ear : NaN,
  });
  document.getElementById('recIndicator').textContent = `Rows: ${state.rows.length}`;
}

function startRecording() {
  if (state.recording) return;
  state.rows        = [];
  state.recordingStart = performance.now();
  state.recording   = true;
  document.getElementById('btnStartRec').disabled = true;
  document.getElementById('btnStopRec').disabled  = false;
  const ind = document.getElementById('recIndicator');
  ind.textContent = 'Rows: 0';
  ind.className   = 'rec-indicator active';
  setStatus('Recording started.');
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  document.getElementById('btnStartRec').disabled = false;
  document.getElementById('btnStopRec').disabled  = true;
  const ind = document.getElementById('recIndicator');
  ind.className   = 'rec-indicator';

  const rowCount = state.rows.length;
  if (rowCount === 0) {
    ind.textContent = 'Rows: 0 (no data)';
    setStatus('Recording stopped — no data captured.');
    return;
  }
  ind.textContent = `Rows: ${rowCount} — saved`;
  exportCSV(state.rows);
}

function exportCSV(rows) {
  const COLS = [
    'timestamp', 'ear_left', 'ear_right',
    'action_left', 'action_right',
    'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4',
    'EEG_Frontal_1', 'EEG_Frontal_2',
    'EEG_Behind_Left_Ear', 'EEG_Behind_Right_Ear',
  ];

  const lines = [COLS.join(',')];
  for (const row of rows) {
    lines.push(COLS.map(c => {
      const v = row[c];
      if (v === undefined) return '';
      if (typeof v === 'number') return isNaN(v) ? 'NaN' : v.toString();
      return `"${String(v).replace(/"/g, '""')}"`;
    }).join(','));
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  const ts   = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.download = `bci_recording_${ts}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  setStatus(`Saved ${rows.length} rows → ${a.download}`);
}

// ═══════════════════════════════════════════════════════════════
// Periodic status bar refresh
// ═══════════════════════════════════════════════════════════════
setInterval(() => {
  const camStr = state.cameraOk ? 'Camera: OK' : 'Camera: ERROR';
  const bciStr = state.bciConnected ? 'OpenBCI: connected' : 'OpenBCI: disconnected';
  let recStr   = 'Idle';
  if (state.calibrating) recStr = 'CALIBRATING…';
  else if (state.recording) recStr = `Recording (${state.rows.length} rows)`;
  setStatus(`${camStr}  |  ${bciStr}  |  ${recStr}`);
}, 500);
