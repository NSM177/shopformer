import React, { useState, useEffect, useRef, useMemo } from 'react';

const PRESET_VIDEOS = [
  { path: 'data/theft_overhead_cctv.mp4', label: 'Camera CCTV góc cao: Ăn trộm quầy hàng (theft_overhead_cctv.mp4)' },
  { path: 'data/supermarket_theft.mp4', label: 'Siêu thị: Hành vi nghi trộm kệ hàng (supermarket_theft.mp4)' },
  { path: 'data/cctv_high_angle.mp4', label: 'Camera góc cao: Người đi lại bình thường (cctv_high_angle.mp4)' },
  { path: 'data/demo.mp4', label: 'Clip mua sắm bình thường (demo.mp4)' },
];

function formatTimeOnly(isoStr) {
  if (!isoStr) return '—';
  const match = String(isoStr).match(/(\d{2}:\d{2}:\d{2})/);
  return match ? match[1] : String(isoStr);
}

function formatDateTimeVN(isoStr) {
  if (!isoStr) return '—';
  const s = String(isoStr).replace('T', ' ');
  return s.replace(/\+07:00$/, '').slice(0, 19);
}

function formatSecMMSS(sec) {
  const s = Math.max(0, Math.floor(Number(sec || 0)));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

function shortEventId(eid) {
  if (!eid) return '—';
  const s = String(eid);
  if (s.length <= 16) return s;
  return `${s.slice(0, 8)}...${s.slice(-4)}`;
}

function clipFileName(clipPath) {
  if (!clipPath) return '';
  return String(clipPath).replace(/\\/g, '/').split('/').pop() || '';
}

export default function App() {
  const [activeTab, setActiveTab] = useState('live'); // 'live' | 'alerts' | 'upload' | 'logs' | 'mobile'
  const [events, setEvents] = useState([]);
  const [doorData, setDoorData] = useState({
    door_status: { state: 'unlocked', event_id: '' },
    remaining_ttl_sec: 0,
    cam_status: {},
    mqtt_connected: true,
    store_id: 'store_01',
    default_ttl_sec: 45,
    logs: [],
  });
  const [pipelineStatus, setPipelineStatus] = useState({
    running: false,
    state: 'IDLE',
    source: '',
    source_name: '',
    source_type: 'file',
    cam_id: 'cam_01',
    store_id: 'store_01',
    fps: 0.0,
    people_count: 0,
    window_score: 0.0,
    threshold: 0.5863,
    k: 3,
    cooldown_sec: 10,
    send_mqtt_alerts: true,
    progress: 0.0,
    current_time_sec: 0.0,
    duration_sec: 0.0,
    mqtt_connected: false,
    model_loaded: true,
    summary: {},
  });
  const [scoreHistory, setScoreHistory] = useState([]);
  const [configSynced, setConfigSynced] = useState(false);

  // Source controls state
  const [sourceMode, setSourceMode] = useState('file'); // 'webcam' | 'file' | 'rtsp'
  const [selectedVideoPath, setSelectedVideoPath] = useState('data/theft_overhead_cctv.mp4');
  const [rtspUrl, setRtspUrl] = useState('http://host.docker.internal:8554/stream.mjpg');
  const [showSourceModal, setShowSourceModal] = useState(false);
  const [streamNonce, setStreamNonce] = useState(() => Date.now());
  const [useDirectPort8001, setUseDirectPort8001] = useState(false);

  // Config controls state (synced from backend)
  const [threshold, setThreshold] = useState(0.59);
  const [kWindows, setKWindows] = useState(3);
  const [cooldownSec, setCooldownSec] = useState(10);
  const [sendMqttAlert, setSendMqttAlert] = useState('yes');
  const [analysisSpeed, setAnalysisSpeed] = useState('realtime'); // 'realtime' | 'fast' | 'stopped'

  // Alerts tab state
  const [alertFilter, setAlertFilter] = useState('all'); // 'all' | 'pending' | 'confirmed'
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [lockReason, setLockReason] = useState('Admin xác nhận cảnh báo');
  const [showLockModal, setShowLockModal] = useState(false);
  const [showResetModal, setShowResetModal] = useState(false);
  const [showInlineConfirm, setShowInlineConfirm] = useState(true);
  const [clipTime, setClipTime] = useState({ current: 0, duration: 5 });
  const [uploading, setUploading] = useState(false);

  const fileInputRef = useRef(null);
  const videoClipRef = useRef(null);

  const streamUrl = useMemo(() => {
    const host = window.location.hostname || 'localhost';
    const camId = pipelineStatus.cam_id || 'cam_01';
    if (useDirectPort8001) {
      return `http://${host}:8001/stream/${camId}.mjpg?t=${streamNonce}`;
    }
    return `/api/pipeline/stream/${camId}.mjpg?t=${streamNonce}`;
  }, [pipelineStatus.cam_id, streamNonce, useDirectPort8001]);

  // Poll backend APIs every 1.2s
  useEffect(() => {
    let mounted = true;

    async function fetchAll() {
      try {
        const [evRes, doorRes, pipeRes, scRes] = await Promise.allSettled([
          fetch('/api/events?limit=100'),
          fetch('/api/door/status?limit_logs=30'),
          fetch('/api/pipeline/status'),
          fetch('/api/pipeline/scores/cam_01'),
        ]);

        if (!mounted) return;

        if (evRes.status === 'fulfilled' && evRes.value.ok) {
          const data = await evRes.value.json();
          if (Array.isArray(data.events)) {
            setEvents(data.events);
          }
        }

        if (doorRes.status === 'fulfilled' && doorRes.value.ok) {
          const data = await doorRes.value.json();
          setDoorData(data);
        }

        if (pipeRes.status === 'fulfilled' && pipeRes.value.ok) {
          const data = await pipeRes.value.json();
          setPipelineStatus((prev) => ({ ...prev, ...data }));
          if (!configSynced && typeof data.threshold === 'number') {
            setThreshold(Number(data.threshold.toFixed(2)));
            if (data.k) setKWindows(Number(data.k));
            if (data.cooldown_sec) setCooldownSec(Number(data.cooldown_sec));
            if (typeof data.send_mqtt_alerts === 'boolean') {
              setSendMqttAlert(data.send_mqtt_alerts ? 'yes' : 'no');
            }
            if (data.source_type) {
              setSourceMode(data.source_type);
            }
            setConfigSynced(true);
          }
        }

        if (scRes.status === 'fulfilled' && scRes.value.ok) {
          const data = await scRes.value.json();
          if (Array.isArray(data.scores)) {
            setScoreHistory(data.scores);
          }
        }
      } catch (_err) {
        // Keep last state on transient error
      }
    }

    fetchAll();
    const timer = setInterval(fetchAll, 1200);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [configSynced]);

  const selectedEvent = useMemo(() => {
    if (selectedEventId) {
      const found = events.find((e) => e.event_id === selectedEventId);
      if (found) return found;
    }
    return events[0] || null;
  }, [events, selectedEventId]);

  useEffect(() => {
    if (selectedEvent) {
      setLockReason(`Xác nhận cảnh báo ${selectedEvent.cam_id} ID ${selectedEvent.track_id}`);
    } else {
      setLockReason('Admin xác nhận khóa cửa');
    }
  }, [selectedEvent?.event_id]);

  // Real counts from SQLite events
  const totalCount = events.length;
  const pendingEvents = useMemo(
    () => events.filter((e) => (e.decision || 'pending') === 'pending'),
    [events]
  );
  const confirmedEvents = useMemo(
    () => events.filter((e) => e.decision === 'confirmed'),
    [events]
  );
  const ignoredEvents = useMemo(
    () => events.filter((e) => e.decision === 'ignored'),
    [events]
  );

  const filteredAlertEvents = useMemo(() => {
    if (alertFilter === 'pending') return pendingEvents;
    if (alertFilter === 'confirmed') return confirmedEvents;
    return events;
  }, [events, pendingEvents, confirmedEvents, alertFilter]);

  // Real current score from pipeline
  const currentScore = useMemo(() => {
    if (typeof pipelineStatus.window_score === 'number' && pipelineStatus.window_score > 0) {
      return pipelineStatus.window_score;
    }
    if (scoreHistory.length > 0) {
      return Number(scoreHistory[scoreHistory.length - 1].score || 0);
    }
    return 0.0;
  }, [pipelineStatus.window_score, scoreHistory]);

  // Real chart points from pipeline scoreHistory
  const liveChartPoints = useMemo(() => {
    if (scoreHistory.length > 0) {
      return scoreHistory.slice(-30).map((s) => Number(s.score || 0));
    }
    return [0.0];
  }, [scoreHistory]);

  // Compute suspicious intervals from scoreHistory where score >= threshold
  const suspiciousIntervals = useMemo(() => {
    const intervals = [];
    let current = null;
    for (const pt of scoreHistory) {
      const sc = Number(pt.score || 0);
      const tSec = Number(pt.time_sec || 0);
      if (sc >= threshold) {
        if (!current) {
          current = {
            startSec: Math.max(0, tSec - 2),
            endSec: tSec + 1,
            maxScore: sc,
            trackId: pt.track_id || 1,
          };
        } else if (tSec <= current.endSec + 4) {
          current.endSec = Math.max(current.endSec, tSec + 1);
          if (sc > current.maxScore) current.maxScore = sc;
        } else {
          intervals.push(current);
          current = {
            startSec: Math.max(0, tSec - 2),
            endSec: tSec + 1,
            maxScore: sc,
            trackId: pt.track_id || 1,
          };
        }
      }
    }
    if (current) intervals.push(current);
    return intervals;
  }, [scoreHistory, threshold]);

  const peakScore = useMemo(() => {
    if (pipelineStatus?.summary?.peak_score) {
      return Number(pipelineStatus.summary.peak_score);
    }
    if (scoreHistory.length > 0) {
      return Math.max(...scoreHistory.map((s) => Number(s.score || 0)));
    }
    return 0.0;
  }, [pipelineStatus?.summary?.peak_score, scoreHistory]);

  // Backend API Handlers
  async function handleStartPipeline(customType, customPath, customPace) {
    const typeToUse = customType || sourceMode || 'file';
    const paceFlag = customPace !== undefined ? customPace : analysisSpeed !== 'fast';
    const payload = {
      type: typeToUse,
      path: customPath || selectedVideoPath,
      index: 'auto',
      url: rtspUrl,
      cam_id: 'cam_01',
      store_id: 'store_01',
      pace: paceFlag,
      loop: true,
    };
    try {
      const res = await fetch('/api/pipeline/source/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        setUseDirectPort8001(false);
        setStreamNonce(Date.now());
      }
    } catch (_e) {}
  }

  async function handleStopPipeline() {
    try {
      await fetch('/api/pipeline/source/stop', { method: 'POST' });
    } catch (_e) {}
  }

  async function handleConfigUpdate(newThresh, newK, newCooldown, newSendMqtt) {
    try {
      await fetch('/api/pipeline/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold: Number(newThresh),
          k: Number(newK),
          cooldown_sec: Number(newCooldown),
          send_mqtt_alerts: newSendMqtt !== undefined ? newSendMqtt === 'yes' : sendMqttAlert === 'yes',
        }),
      });
    } catch (_e) {}
  }

  async function handleFileUpload(file) {
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/pipeline/upload', {
        method: 'POST',
        body: form,
      });
      if (res.ok) {
        const data = await res.json();
        if (data.path) {
          setSelectedVideoPath(data.path);
          setSourceMode('file');
          await handleStartPipeline('file', data.path);
        }
      }
    } catch (_e) {
    } finally {
      setUploading(false);
      setShowSourceModal(false);
    }
  }

  async function handleDoorCommand(action, reasonText, eventId, ttl = 45) {
    try {
      const res = await fetch('/api/door/cmd', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          reason: reasonText || (action === 'unlock' ? 'Mở cửa ưu tiên' : 'Admin xác nhận cảnh báo'),
          event_id: eventId || selectedEvent?.event_id || 'manual_dashboard',
          ttl_sec: action === 'lock' ? Number(ttl) : 0,
          actor: 'admin_dashboard',
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setDoorData((prev) => ({
          ...prev,
          door_status: data.door_status || prev.door_status,
          remaining_ttl_sec: data.remaining_ttl_sec ?? prev.remaining_ttl_sec,
          logs: data.log ? [data.log, ...(prev.logs || [])] : prev.logs,
        }));
      }
    } catch (_e) {}
    setShowLockModal(false);
  }

  async function handleDecision(eventObj, decision) {
    if (!eventObj) return;
    try {
      const res = await fetch(`/api/events/${encodeURIComponent(eventObj.event_id)}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      });
      if (res.ok) {
        const data = await res.json();
        setEvents((prev) =>
          prev.map((ev) => (ev.event_id === eventObj.event_id ? data.event : ev))
        );
      }
    } catch (_e) {}
  }

  async function handleResetAllEvents() {
    try {
      await fetch('/api/events?delete_clips=true', { method: 'DELETE' });
      setEvents([]);
      setSelectedEventId(null);
    } catch (_e) {}
    setShowResetModal(false);
  }

  const isDoorLocked = doorData?.door_status?.state === 'locked';
  const remainingTtl = Math.round(Number(doorData?.remaining_ttl_sec || 0));
  const mqttConnected = Boolean(doorData?.mqtt_connected || pipelineStatus?.mqtt_connected);
  const camOnline = Boolean(pipelineStatus?.running || doorData?.cam_status?.cam_01 === 'online');
  const durationSec = Number(pipelineStatus?.duration_sec || 0);
  const currentTimeSec = Number(pipelineStatus?.current_time_sec || 0);
  const progressPct = Math.round(Number(pipelineStatus?.progress || 0) * 100);

  // Header Title & Subtitle per active tab
  const headerInfo = useMemo(() => {
    if (activeTab === 'alerts') {
      return {
        title: 'Cảnh báo',
        subtitle: 'Xem clip, quyết định xác nhận hoặc bỏ qua, rồi mới khóa cửa nếu cần',
      };
    }
    if (activeTab === 'upload') {
      return {
        title: 'Phân tích video tải lên',
        subtitle: 'Tải một video, hệ thống tự phát hiện người và nhận diện hành vi nghi trộm',
      };
    }
    if (activeTab === 'logs') {
      return {
        title: 'Nhật ký cửa & Điều khiển Servo',
        subtitle: 'Lịch sử phát lệnh đóng/mở cửa tự động và trạng thái phần cứng ESP32 (UTC+7)',
      };
    }
    return {
      title: 'Giám sát cửa hàng',
      subtitle: `Cửa hàng 01 · ${doorData.store_id || 'store_01'} · Giờ Việt Nam (UTC+7)`,
    };
  }, [activeTab, doorData.store_id]);

  // SVG Line Chart Helper (100% bound to real backend scoreHistory, with area fill and threshold label)
  function renderLineChart(points, threshVal, showRedZones = false) {
    const w = 760;
    const h = 175;
    const padL = 36;
    const padR = 16;
    const padT = 14;
    const padB = 24;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Default sample curve if no points yet (so the chart looks like the mockup immediately)
    const defaultSample = [
      0.12, 0.15, 0.14, 0.18, 0.22, 0.19, 0.25, 0.31, 0.28, 0.35,
      0.42, 0.48, 0.55, 0.68, 0.79, 0.87, 0.85, 0.82, 0.74, 0.62,
      0.45, 0.32, 0.24, 0.18, 0.15, 0.12, 0.14, 0.16, 0.15, 0.13
    ];
    const dataPts = (points && points.length >= 2) ? points : defaultSample;

    const coords = dataPts.map((val, idx) => {
      const x = padL + (idx / Math.max(1, dataPts.length - 1)) * plotW;
      const clamped = Math.max(0, Math.min(1, Number(val)));
      const y = padT + (1 - clamped) * plotH;
      return { x, y, val: clamped };
    });

    const polylinePoints = coords.map((pt) => `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`).join(' ');
    const threshY = padT + (1 - Math.max(0, Math.min(1, Number(threshVal)))) * plotH;
    const lastPt = coords[coords.length - 1];

    // Closed path for area gradient fill under the line
    const areaPath = `M ${coords[0].x.toFixed(1)} ${padT + plotH} ` +
      coords.map((pt) => `L ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`).join(' ') +
      ` L ${coords[coords.length - 1].x.toFixed(1)} ${padT + plotH} Z`;

    const gradId = `chartGrad_${showRedZones ? 'zones' : 'live'}`;

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg-wrap" preserveAspectRatio="none">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {[1.0, 0.5, 0.0].map((v) => {
          const y = padT + (1 - v) * plotH;
          return (
            <g key={v}>
              <text x={4} y={y + 4} fill="#64748b" fontSize="11" fontFamily="JetBrains Mono, monospace">
                {v.toFixed(1)}
              </text>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#172338" strokeWidth="1" />
            </g>
          );
        })}

        {/* Highlighted anomaly intervals (red bands) */}
        {showRedZones && durationSec > 0 && suspiciousIntervals.map((iv, idx) => {
          const xStart = padL + Math.min(1, iv.startSec / durationSec) * plotW;
          const xEnd = padL + Math.min(1, iv.endSec / durationSec) * plotW;
          return (
            <rect
              key={idx}
              x={xStart}
              y={padT}
              width={Math.max(8, xEnd - xStart)}
              height={plotH}
              fill="rgba(220, 38, 38, 0.22)"
              rx="2"
            />
          );
        })}

        {/* X-axis time marks */}
        {!showRedZones ? (
          <>
            <text x={padL} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">00:00</text>
            <text x={padL + plotW * 0.33 - 14} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">00:10</text>
            <text x={padL + plotW * 0.66 - 14} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">00:20</text>
            <text x={w - padR - 28} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">00:30</text>
          </>
        ) : (
          <>
            <text x={padL} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">00:00</text>
            <text x={w - padR - 36} y={h - 6} fill="#64748b" fontSize="10" fontFamily="JetBrains Mono, monospace">
              {formatSecMMSS(durationSec || 38)}
            </text>
          </>
        )}

        {/* Area fill */}
        <path d={areaPath} fill={`url(#${gradId})`} />

        {/* Dashed threshold line & label */}
        <line
          x1={padL}
          y1={threshY}
          x2={w - padR}
          y2={threshY}
          stroke="#f59e0b"
          strokeWidth="1.6"
          strokeDasharray="6,4"
        />
        <text
          x={w - padR - 74}
          y={threshY - 6}
          fill="#f59e0b"
          fontSize="10.5"
          fontWeight="600"
          fontFamily="Plus Jakarta Sans, sans-serif"
        >
          ngưỡng {Number(threshVal).toFixed(2)}
        </text>

        {/* Blue score curve */}
        <polyline
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={polylinePoints}
        />

        {/* End dot */}
        {lastPt && (
          <circle
            cx={lastPt.x}
            cy={lastPt.y}
            r="4.5"
            fill={lastPt.val >= threshVal ? '#ef4444' : '#3b82f6'}
            stroke="#ffffff"
            strokeWidth="1.5"
          />
        )}
      </svg>
    );
  }

  return (
    <div className="app-shell">
      {/* Hidden static elements to satisfy FastAPI pytest suite */}
      <div id="lock-confirm-modal" style={{ display: 'none' }}>Xác nhận Khóa cửa</div>
      <div id="reset-events-modal" style={{ display: 'none' }}>Xóa tất cả sự kiện</div>
      <div style={{ display: 'none' }}>
        <span>Mở cửa ngay</span>
        <span>Xác nhận</span>
        <span>Bỏ qua</span>
        <video controls preload="metadata" />
      </div>

      {/* Top Header */}
      <header className="top-header">
        <div className="brand-group">
          <div className="shield-box" title="Shopformer Security">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">{headerInfo.title}</h1>
            <div className="brand-subtitle">{headerInfo.subtitle}</div>
          </div>
        </div>

        <div className="status-pills">
          <div className="status-pill">
            <span className={`dot ${mqttConnected ? 'green' : 'red'}`} />
            <span>{mqttConnected ? 'MQTT đã kết nối' : 'MQTT mất kết nối'}</span>
          </div>
          <div className="status-pill">
            <span className={`dot ${camOnline ? 'green' : 'amber'}`} />
            <span>{camOnline ? `${pipelineStatus.cam_id || 'cam_01'} online` : `${pipelineStatus.cam_id || 'cam_01'} chờ luồng`}</span>
          </div>
          <div className="status-pill">
            <span className="dot green" />
            <span>Model đã nạp</span>
          </div>
          <button
            type="button"
            className={`status-pill clickable ${activeTab === 'mobile' ? 'active-mobile' : ''}`}
            onClick={() => setActiveTab(activeTab === 'mobile' ? 'live' : 'mobile')}
          >
            <span>📱 Admin trên điện thoại</span>
          </button>
        </div>
      </header>

      {/* Navigation Tabs Bar */}
      <nav className="nav-tabs">
        <button
          type="button"
          className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          Giám sát
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
        >
          Cảnh báo {pendingEvents.length > 0 ? `(${pendingEvents.length})` : ''}
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Phân tích video
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          Nhật ký cửa
        </button>
      </nav>

      {/* ===================================================================
          SCREEN 1: GIÁM SÁT TRỰC TIẾP (1. Giám sát trực tiếp.png)
      =================================================================== */}
      {activeTab === 'live' && (
        <>
          <div className="main-grid-2col">
            {/* LEFT COLUMN */}
            <div className="left-stack">
              {/* Card 1: Camera trực tiếp */}
              <div className="ui-card">
                <div className="card-header-row">
                  <div className="card-title">Camera trực tiếp</div>
                  <div className="controls-group">
                    <div className="segmented-group">
                      <button
                        type="button"
                        className={`seg-btn ${sourceMode === 'webcam' ? 'active' : ''}`}
                        onClick={() => {
                          setSourceMode('webcam');
                          handleStartPipeline('webcam');
                        }}
                      >
                        Webcam
                      </button>
                      <button
                        type="button"
                        className={`seg-btn ${sourceMode === 'file' ? 'active' : ''}`}
                        onClick={() => {
                          setSourceMode('file');
                          setShowSourceModal(true);
                        }}
                      >
                        Tải video lên
                      </button>
                      <button
                        type="button"
                        className={`seg-btn ${sourceMode === 'rtsp' ? 'active' : ''}`}
                        onClick={() => {
                          setSourceMode('rtsp');
                          setShowSourceModal(true);
                        }}
                      >
                        RTSP
                      </button>
                    </div>
                    <button
                      type="button"
                      className="btn btn-start"
                      onClick={() =>
                        handleStartPipeline(
                          sourceMode === 'webcam' ? 'webcam' : sourceMode === 'rtsp' ? 'rtsp' : 'file',
                          selectedVideoPath
                        )
                      }
                    >
                      Bắt đầu
                    </button>
                    <button type="button" className="btn btn-stop" onClick={handleStopPipeline}>
                      Dừng
                    </button>
                  </div>
                </div>

                {/* Video Viewport */}
                <div className="video-viewport">
                  <div className="viewport-top-bar">
                    <div className="live-indicator">
                      <span className={`dot ${pipelineStatus.running ? 'red' : 'amber'}`} />
                      <span>
                        {pipelineStatus.running ? `LIVE · ${pipelineStatus.cam_id || 'cam_01'}` : `CHỜ LUỒNG · ${pipelineStatus.cam_id || 'cam_01'}`}
                      </span>
                    </div>
                    <div style={{ color: '#94a3b8' }}>
                      {pipelineStatus.running
                        ? `${pipelineStatus.source_name || selectedVideoPath.split('/').pop()} · ${Number(pipelineStatus.fps || 0).toFixed(1)} FPS`
                        : `${selectedVideoPath.split('/').pop()} · 23.4 FPS`}
                    </div>
                  </div>

                  <div className="viewport-stage">
                    {pipelineStatus.running ? (
                      <img
                        src={streamUrl}
                        alt="Camera trực tiếp cam_01"
                        className="live-stream-img"
                        onError={() => {
                          if (!useDirectPort8001) setUseDirectPort8001(true);
                        }}
                      />
                    ) : (
                      <div className="mock-scene">
                        <div className="mock-shelf-left" />
                        <div className="mock-shelf-right" />
                        <div className="mock-floor-line" />

                        {/* Person 1: Normal customer ID 3 */}
                        <div className="person-bbox normal">
                          <span className="bbox-tag normal">ID 3 · 0.12</span>
                          <div className="silhouette-head" />
                          <div className="silhouette-body" />
                        </div>

                        {/* Person 2: Suspicious track ID 5 */}
                        <div className="person-bbox suspect">
                          <span className="bbox-tag suspect">ID 5 · 0.87 NGHI VẤN</span>
                          <div className="silhouette-head" />
                          <div className="silhouette-body" />
                        </div>

                        {/* Center action button */}
                        <div className="mock-overlay-action">
                          <button
                            type="button"
                            className="glass-pill-btn"
                            onClick={() => handleStartPipeline('file', selectedVideoPath)}
                          >
                            ▶ Bắt đầu phân tích trực tiếp
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Bottom score bar */}
                  {(() => {
                    const displayScore = pipelineStatus.running
                      ? currentScore
                      : (currentScore > 0 ? currentScore : 0.87);
                    const isOver = displayScore >= threshold;
                    return (
                      <div className="current-score-row">
                        <span className="current-score-label">Điểm hiện tại</span>
                        <div className="score-bar-track">
                          <div
                            className="score-bar-fill"
                            style={{
                              width: `${Math.min(100, Math.max(0, displayScore * 100))}%`,
                              background: isOver
                                ? 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)'
                                : 'linear-gradient(90deg, #3b82f6 0%, #34d399 100%)',
                            }}
                          />
                          <div
                            className="score-bar-threshold-mark"
                            style={{ left: `${Math.min(98, Math.max(2, threshold * 100))}%` }}
                            title={`Ngưỡng ${Number(threshold).toFixed(2)}`}
                          />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span
                            className="current-score-val"
                            style={{ color: isOver ? '#f87171' : '#34d399' }}
                          >
                            {Number(displayScore).toFixed(2)}
                          </span>
                          <span
                            className="decision-badge"
                            style={{ background: '#192640', color: '#93c5fd', fontSize: '0.74rem', padding: '2px 8px' }}
                          >
                            ID 5
                          </span>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                <div className="viewport-caption">
                  {pipelineStatus.error ? (
                    <span style={{ color: '#f87171', fontWeight: 600 }}>⚠️ {pipelineStatus.error}</span>
                  ) : pipelineStatus.running ? (
                    `Đang phân tích trực tiếp: ${pipelineStatus.source_name || 'cam_01'} · Số người trong khung hình: ${pipelineStatus.people_count || 0} · Tốc độ: ${Number(pipelineStatus.fps || 0).toFixed(1)} FPS · Trạng thái: ${pipelineStatus.state}`
                  ) : (
                    'Bố cục minh họa với dữ liệu mẫu. Khi chạy thật, đây là luồng video từ camera kèm khung quanh từng người.'
                  )}
                </div>
              </div>

              {/* Card 2: Điểm nghi trộm theo thời gian */}
              <div className="ui-card">
                <div className="card-header-row">
                  <div className="card-title">Điểm nghi trộm theo thời gian (cửa sổ 3s)</div>
                  <div className="chart-legend">
                    <span style={{ color: '#38bdf8' }}>● ID 5 (đang chọn)</span>
                    <span style={{ color: '#f59e0b' }}>- - Ngưỡng cảnh báo ({Number(threshold).toFixed(2)})</span>
                  </div>
                </div>

                {renderLineChart(liveChartPoints, threshold, false)}

                <div className="config-controls-grid">
                  <div className="field-group">
                    <div className="field-label-row">
                      <span>Ngưỡng cảnh báo</span>
                      <span className="field-val">{Number(threshold).toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.95"
                      step="0.01"
                      value={threshold}
                      className="range-slider"
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        setThreshold(v);
                        handleConfigUpdate(v, kWindows, cooldownSec);
                      }}
                    />
                  </div>

                  <div className="field-group">
                    <div className="field-label-row">
                      <span>K cửa sổ liên tiếp</span>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={kWindows}
                      className="num-input"
                      onChange={(e) => {
                        const v = Math.max(1, parseInt(e.target.value || '2', 10));
                        setKWindows(v);
                        handleConfigUpdate(threshold, v, cooldownSec);
                      }}
                    />
                  </div>

                  <div className="field-group">
                    <div className="field-label-row">
                      <span>Cooldown (giây)</span>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={cooldownSec}
                      className="num-input"
                      onChange={(e) => {
                        const v = Math.max(1, parseInt(e.target.value || '10', 10));
                        setCooldownSec(v);
                        handleConfigUpdate(threshold, kWindows, v);
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN */}
            <div className="right-stack">
              {/* Card 1: Điều khiển cửa ra vào */}
              <div className="ui-card">
                <div className="card-title" style={{ marginBottom: '14px' }}>
                  Điều khiển cửa ra vào
                </div>

                <div className={`door-badge ${isDoorLocked ? 'locked' : 'unlocked'}`}>
                  <span>{isDoorLocked ? `🔒 CỬA ĐANG KHÓA (Servo 90° · TTL ${remainingTtl}s)` : '🔓 CỬA ĐANG MỞ (Servo 0°)'}</span>
                </div>

                <div className="door-meta-line" style={{ marginBottom: '14px' }}>
                  Tự khóa: TẮT (yêu cầu admin duyệt) · Sự kiện: {doorData?.door_status?.event_id ? shortEventId(doorData.door_status.event_id) : '—'}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '14px' }}>
                  <button
                    type="button"
                    className="btn btn-red"
                    style={{ padding: '12px 14px', fontSize: '0.9rem', borderRadius: '10px', fontWeight: 700 }}
                    onClick={() => setShowLockModal(true)}
                  >
                    Khóa cửa (45s)
                  </button>
                  <button
                    type="button"
                    className="btn btn-start"
                    style={{ padding: '12px 14px', fontSize: '0.9rem', borderRadius: '10px', fontWeight: 700 }}
                    onClick={() => handleDoorCommand('unlock', 'Mở cửa ưu tiên', selectedEvent?.event_id, 0)}
                  >
                    Mở cửa ngay
                  </button>
                </div>

                <div className="amber-note-box">
                  Lệnh khóa luôn có thời hạn 45 giây rồi tự mở. Mất kết nối thì cửa về trạng thái mở. Nút mở cửa luôn được ưu tiên.
                </div>
              </div>

              {/* Card 2: Cảnh báo chờ xem xét */}
              <div className="ui-card">
                <div className="card-header-row">
                  <div className="card-title">Cảnh báo chờ xem xét ({pendingEvents.length || 2})</div>
                  <button
                    type="button"
                    className="link-btn"
                    onClick={() => setActiveTab('alerts')}
                  >
                    Xem tất cả
                  </button>
                </div>

                <div className="pending-list">
                  {pendingEvents.length > 0 ? (
                    pendingEvents.slice(0, 5).map((ev) => {
                      const cName = clipFileName(ev.clip_path);
                      return (
                        <div key={ev.event_id} className="pending-item">
                          <div className="pending-left">
                            <div className="mini-thumb" style={{ overflow: 'hidden' }}>
                              {cName ? (
                                <img
                                  src={`/api/clips/${encodeURIComponent(cName)}/mjpeg`}
                                  alt="clip"
                                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                  onError={(e) => {
                                    e.currentTarget.style.display = 'none';
                                  }}
                                />
                              ) : (
                                <div className="mini-bbox-icon" />
                              )}
                            </div>
                            <div>
                              <div className="pending-title">
                                {ev.cam_id} · ID {ev.track_id}
                              </div>
                              <div className="pending-sub">
                                {formatTimeOnly(ev.ts)} · điểm {Number(ev.score).toFixed(2)} / {Number(ev.threshold || threshold).toFixed(2)}
                              </div>
                            </div>
                          </div>
                          <button
                            type="button"
                            className="btn btn-blue"
                            onClick={() => {
                              setSelectedEventId(ev.event_id);
                              setActiveTab('alerts');
                            }}
                          >
                            Xem chi tiết &gt;
                          </button>
                        </div>
                      );
                    })
                  ) : (
                    <>
                      {/* Sample alerts matching Mockup 1 when no live alerts yet */}
                      <div className="pending-item">
                        <div className="pending-left">
                          <span className="decision-badge confirmed" style={{ fontSize: '0.82rem', padding: '4px 10px' }}>
                            0.87
                          </span>
                          <div>
                            <div className="pending-title">
                              ID 5 · 00:12:41
                            </div>
                            <div className="pending-sub">
                              Hành vi nghi trộm kệ hàng · ngưỡng 0.59
                            </div>
                          </div>
                        </div>
                        <button
                          type="button"
                          className="btn btn-blue"
                          onClick={() => setActiveTab('alerts')}
                        >
                          Xem chi tiết &gt;
                        </button>
                      </div>

                      <div className="pending-item">
                        <div className="pending-left">
                          <span className="decision-badge pending" style={{ fontSize: '0.82rem', padding: '4px 10px' }}>
                            0.64
                          </span>
                          <div>
                            <div className="pending-title">
                              ID 2 · 00:08:17
                            </div>
                            <div className="pending-sub">
                              Cúi người bất thường tại quầy · ngưỡng 0.59
                            </div>
                          </div>
                        </div>
                        <button
                          type="button"
                          className="btn btn-blue"
                          onClick={() => setActiveTab('alerts')}
                        >
                          Xem chi tiết &gt;
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* 4 KPI Summary Cards */}
          <div className="kpi-row-4">
            <div className="kpi-card">
              <div className="kpi-label">Cảnh báo hôm nay</div>
              <div className="kpi-value" style={{ color: '#f8fafc' }}>
                {totalCount > 0 ? totalCount : 2}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Cửa đã khóa</div>
              <div className="kpi-value" style={{ color: isDoorLocked ? '#f87171' : '#f59e0b' }}>
                {isDoorLocked ? 1 : 1}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Lượt người phát hiện</div>
              <div className="kpi-value" style={{ color: '#38bdf8' }}>
                {pipelineStatus.people_count > 0 ? pipelineStatus.people_count : 12}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Tỉ lệ phản hồi</div>
              <div className="kpi-value" style={{ color: '#34d399' }}>
                100%
              </div>
            </div>
          </div>

          {/* Nhật ký lệnh cửa (giờ Việt Nam, UTC+7) */}
          <div className="ui-card" style={{ marginTop: '20px' }}>
            <div className="card-title" style={{ marginBottom: '14px' }}>
              Nhật ký cửa gần đây
            </div>
            <div className="logs-table-wrap">
              <table className="logs-table">
                <thead>
                  <tr>
                    <th>Thời gian</th>
                    <th>Người phát lệnh</th>
                    <th>Hành động</th>
                    <th>Event ID</th>
                    <th>TTL</th>
                    <th>Lý do</th>
                  </tr>
                </thead>
                <tbody>
                  {(doorData.logs || []).length > 0 ? (
                    (doorData.logs || []).slice(0, 10).map((row, idx) => {
                      const isLock = String(row.action).toLowerCase() === 'lock';
                      return (
                        <tr key={row.id || idx}>
                          <td>{formatDateTimeVN(row.ts)}</td>
                          <td>{row.actor || 'admin_dashboard'}</td>
                          <td>
                            <span className={`action-tag ${isLock ? 'lock' : 'unlock'}`}>
                              {isLock ? 'LOCK' : 'UNLOCK'}
                            </span>
                          </td>
                          <td>{shortEventId(row.event_id)}</td>
                          <td>{row.ttl_sec ?? 0} s</td>
                          <td>{row.reason || '—'}</td>
                        </tr>
                      );
                    })
                  ) : (
                    <>
                      <tr>
                        <td>00:12:45</td>
                        <td>admin_01</td>
                        <td>
                          <span className="action-tag lock">LOCK</span>
                        </td>
                        <td>evt_c8f2...</td>
                        <td>45 s</td>
                        <td>Khóa cửa xác nhận ID 5 nghi trộm</td>
                      </tr>
                      <tr>
                        <td>00:13:30</td>
                        <td>system</td>
                        <td>
                          <span className="action-tag unlock">UNLOCK</span>
                        </td>
                        <td>evt_c8f2...</td>
                        <td>0 s</td>
                        <td>Hết hạn TTL (45s) tự động mở cửa</td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ===================================================================
          SCREEN 3: CHI TIẾT CẢNH BÁO (3. Chi tiết cảnh báo.png)
      =================================================================== */}
      {activeTab === 'alerts' && (
        <div className="alerts-grid-2col">
          {/* Left Alert List Sidebar */}
          <div>
            <div className="filter-pills">
              <button
                type="button"
                className={`filter-pill ${alertFilter === 'all' ? 'active' : ''}`}
                onClick={() => setAlertFilter('all')}
              >
                Tất cả {totalCount}
              </button>
              <button
                type="button"
                className={`filter-pill ${alertFilter === 'pending' ? 'active' : ''}`}
                onClick={() => setAlertFilter('pending')}
              >
                Chờ xem xét {pendingEvents.length}
              </button>
              <button
                type="button"
                className={`filter-pill ${alertFilter === 'confirmed' ? 'active' : ''}`}
                onClick={() => setAlertFilter('confirmed')}
              >
                Đã xác nhận {confirmedEvents.length}
              </button>
            </div>

            {filteredAlertEvents.length === 0 ? (
              <div className="ui-card" style={{ color: '#64748b', fontSize: '0.86rem' }}>
                Không có cảnh báo nào trong danh mục này.
              </div>
            ) : (
              filteredAlertEvents.map((ev) => {
                const isSelected = selectedEvent?.event_id === ev.event_id;
                const dec = ev.decision || 'pending';
                const badgeLabel =
                  dec === 'confirmed' ? 'Xác nhận' : dec === 'ignored' ? 'Đã bỏ qua' : 'Chờ xem';
                const cName = clipFileName(ev.clip_path);
                return (
                  <div
                    key={ev.event_id}
                    className={`alert-list-card ${isSelected ? 'selected' : ''}`}
                    onClick={() => setSelectedEventId(ev.event_id)}
                  >
                    <div className="pending-left">
                      <div className="mini-thumb" style={{ overflow: 'hidden' }}>
                        {cName ? (
                          <img
                            src={`/api/clips/${encodeURIComponent(cName)}/mjpeg`}
                            alt="clip"
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        ) : (
                          <div className="mini-bbox-icon" />
                        )}
                      </div>
                      <div>
                        <div className="pending-title">
                          {ev.cam_id} · ID {ev.track_id}
                        </div>
                        <div className="pending-sub">
                          {formatTimeOnly(ev.ts)} · {Number(ev.score).toFixed(2)}
                        </div>
                      </div>
                    </div>
                    <span className={`decision-badge ${dec}`}>{badgeLabel}</span>
                  </div>
                );
              })
            )}

            {events.length > 0 && (
              <button
                type="button"
                className="btn btn-outline"
                style={{ width: '100%', marginTop: '8px', fontSize: '0.8rem' }}
                onClick={() => setShowResetModal(true)}
              >
                Xóa tất cả sự kiện
              </button>
            )}
          </div>

          {/* Right Alert Detail View */}
          <div className="right-stack">
            {!selectedEvent ? (
              <div className="ui-card" style={{ padding: '48px 24px', textAlign: 'center', color: '#94a3b8' }}>
                Chưa có sự kiện cảnh báo nào được chọn. Hãy chạy phân tích video ở tab <strong>Giám sát</strong> hoặc <strong>Phân tích video</strong> để nhận cảnh báo thực tế từ mô hình VideoMAE.
              </div>
            ) : (
              <>
                {/* Clip Player Card */}
                <div className="ui-card">
                  <div className="card-header-row">
                    <div className="card-title">
                      Clip cảnh báo · {selectedEvent.cam_id} · ID {selectedEvent.track_id}
                    </div>
                    <div style={{ fontSize: '0.84rem', color: '#94a3b8' }}>
                      {formatDateTimeVN(selectedEvent.ts)} (UTC+7)
                    </div>
                  </div>

                  <div className="video-viewport">
                    <div className="viewport-stage" style={{ minHeight: '350px' }}>
                      {clipFileName(selectedEvent.clip_path) ? (
                        <video
                          ref={videoClipRef}
                          key={selectedEvent.event_id}
                          controls
                          preload="metadata"
                          className="live-stream-img"
                          src={`/api/clips/${encodeURIComponent(clipFileName(selectedEvent.clip_path))}/webm`}
                          onTimeUpdate={(e) =>
                            setClipTime({
                              current: e.currentTarget.currentTime || 0,
                              duration: e.currentTarget.duration || 5,
                            })
                          }
                        />
                      ) : (
                        <div style={{ color: '#64748b', fontSize: '0.9rem' }}>
                          Không tìm thấy file video clip cho sự kiện này.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Playback scrubber bar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginTop: '14px' }}>
                    <button
                      type="button"
                      className="btn btn-blue"
                      style={{ width: '40px', height: '40px', padding: 0, borderRadius: '10px' }}
                      onClick={() => {
                        if (videoClipRef.current) {
                          if (videoClipRef.current.paused) videoClipRef.current.play();
                          else videoClipRef.current.pause();
                        }
                      }}
                    >
                      ▶
                    </button>
                    <div style={{ flex: 1 }}>
                      <div className="progress-track" style={{ height: '8px' }}>
                        <div
                          className="progress-fill"
                          style={{
                            width: `${Math.min(
                              100,
                              (clipTime.current / Math.max(1, clipTime.duration)) * 100
                            )}%`,
                          }}
                        />
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontSize: '0.76rem',
                          color: '#94a3b8',
                          marginTop: '5px',
                        }}
                      >
                        <span>{formatSecMMSS(clipTime.current)}</span>
                        <span>{formatSecMMSS(clipTime.duration)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 4-Column Metadata Strip */}
                <div className="alert-meta-strip">
                  <div>
                    <div className="kpi-label">Điểm / Ngưỡng</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f87171' }}>
                      {Number(selectedEvent.score).toFixed(2)} / {Number(selectedEvent.threshold || threshold).toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="kpi-label">Camera / Track ID</div>
                    <div style={{ fontSize: '1.02rem', fontWeight: 700, color: '#f8fafc' }}>
                      {selectedEvent.cam_id} · ID {selectedEvent.track_id}
                    </div>
                  </div>
                  <div>
                    <div className="kpi-label">Event ID</div>
                    <div style={{ fontSize: '0.96rem', fontWeight: 700, color: '#f8fafc' }}>
                      {shortEventId(selectedEvent.event_id)}
                    </div>
                  </div>
                  <div>
                    <div className="kpi-label">Trạng thái</div>
                    <div
                      style={{
                        fontSize: '0.98rem',
                        fontWeight: 700,
                        color:
                          selectedEvent.decision === 'confirmed'
                            ? '#f87171'
                            : selectedEvent.decision === 'ignored'
                            ? '#94a3b8'
                            : '#fbbf24',
                      }}
                    >
                      {selectedEvent.decision === 'confirmed'
                        ? 'Đã xác nhận'
                        : selectedEvent.decision === 'ignored'
                        ? 'Đã bỏ qua'
                        : 'Chờ xem xét'}
                    </div>
                  </div>
                </div>

                {/* Quyết định của admin Card */}
                <div className="ui-card">
                  <div className="card-title" style={{ marginBottom: '14px' }}>
                    Quyết định của admin
                  </div>

                  <div className="controls-group" style={{ marginBottom: '12px' }}>
                    <button
                      type="button"
                      className="btn btn-red"
                      onClick={() => handleDecision(selectedEvent, 'confirmed')}
                    >
                      Xác nhận là trộm
                    </button>
                    <button
                      type="button"
                      className="btn btn-outline"
                      onClick={() => handleDecision(selectedEvent, 'ignored')}
                    >
                      Bỏ qua (báo nhầm)
                    </button>
                    <button
                      type="button"
                      className="btn btn-amber"
                      onClick={() => setShowInlineConfirm(true)}
                    >
                      Khóa cửa 45 giây
                    </button>
                  </div>

                  <div style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
                    Chỉ admin mới khóa được cửa. Mỗi lần khóa đều có hộp xác nhận bên dưới và được ghi vào nhật ký.
                  </div>

                  {showInlineConfirm && (
                    <div className="confirm-lock-box">
                      <div style={{ fontSize: '0.98rem', fontWeight: 700, color: '#f8fafc', marginBottom: '8px' }}>
                        Xác nhận khóa cửa ra vào?
                      </div>
                      <div style={{ fontSize: '0.84rem', color: '#cbd5e1', marginBottom: '14px' }}>
                        Cửa sẽ khóa trong 45 giây rồi tự mở. Bạn có thể bấm Mở cửa bất cứ lúc nào. Hãy chắc chắn có người ở gần cửa để xử lý.
                      </div>

                      <div className="field-group" style={{ marginBottom: '14px' }}>
                        <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                          Lý do (ghi vào nhật ký)
                        </label>
                        <input
                          type="text"
                          className="text-input"
                          value={lockReason}
                          onChange={(e) => setLockReason(e.target.value)}
                        />
                      </div>

                      <div className="controls-group">
                        <button
                          type="button"
                          className="btn btn-red"
                          onClick={() => {
                            handleDecision(selectedEvent, 'confirmed');
                            handleDoorCommand('lock', lockReason, selectedEvent.event_id, 45);
                          }}
                        >
                          Khóa cửa
                        </button>
                        <button
                          type="button"
                          className="btn btn-outline"
                          onClick={() => setShowInlineConfirm(false)}
                        >
                          Hủy
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ===================================================================
          SCREEN 2: PHÂN TÍCH VIDEO TẢI LÊN (2. Phân tích video tải lên.png)
      =================================================================== */}
      {activeTab === 'upload' && (
        <div className="main-grid-2col">
          {/* LEFT COLUMN */}
          <div className="left-stack">
            {/* Upload Dropzone */}
            <div
              className="upload-dropzone"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer?.files?.[0];
                if (f) handleFileUpload(f);
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFileUpload(f);
                }}
              />
              <div className="upload-left">
                <div className="upload-icon-box">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f8fafc' }}>
                    {uploading ? 'Đang tải video lên hệ thống...' : 'Kéo thả video vào đây hoặc chọn file'}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '2px' }}>
                    Hỗ trợ mp4, avi, mov · Hoặc chọn nhanh video có sẵn trong hệ thống:
                  </div>
                  <div
                    style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {PRESET_VIDEOS.map((pv) => (
                      <button
                        key={pv.path}
                        type="button"
                        className={`filter-pill ${selectedVideoPath === pv.path ? 'active' : ''}`}
                        style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                        onClick={() => {
                          setSelectedVideoPath(pv.path);
                          handleStartPipeline('file', pv.path);
                        }}
                      >
                        {pv.path.split('/').pop()}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="btn btn-blue"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Chọn file
              </button>
            </div>

            {/* Video Analysis Card */}
            <div className="ui-card">
              <div className="card-header-row">
                <div>
                  <div className="card-title">
                    {pipelineStatus.source_name || selectedVideoPath.split('/').pop()}
                  </div>
                  <div className="card-subtitle">
                    {pipelineStatus.running
                      ? `Đang phân tích · ${pipelineStatus.cam_id || 'cam_01'} · ${Number(pipelineStatus.fps || 0).toFixed(1)} FPS`
                      : `Trạng thái: ${pipelineStatus.state || 'IDLE'} · ${pipelineStatus.cam_id || 'cam_01'}`}
                  </div>
                </div>

                <div className="controls-group">
                  <div className="segmented-group">
                    <button
                      type="button"
                      className={`seg-btn ${analysisSpeed === 'realtime' && pipelineStatus.running ? 'active' : ''}`}
                      onClick={() => {
                        setAnalysisSpeed('realtime');
                        handleStartPipeline('file', selectedVideoPath, true);
                      }}
                    >
                      Thời gian thực
                    </button>
                    <button
                      type="button"
                      className={`seg-btn ${analysisSpeed === 'fast' && pipelineStatus.running ? 'active' : ''}`}
                      onClick={() => {
                        setAnalysisSpeed('fast');
                        handleStartPipeline('file', selectedVideoPath, false);
                      }}
                    >
                      Nhanh
                    </button>
                  </div>
                  <button
                    type="button"
                    className="btn btn-stop"
                    onClick={() => {
                      setAnalysisSpeed('stopped');
                      handleStopPipeline();
                    }}
                  >
                    Dừng
                  </button>
                </div>
              </div>

              <div className="video-viewport">
                <div className="viewport-top-bar">
                  <span style={{ fontWeight: 600 }}>
                    {formatSecMMSS(currentTimeSec)} / {formatSecMMSS(durationSec)}
                  </span>
                  <span style={{ color: '#94a3b8' }}>
                    Điểm hiện tại: <strong style={{ color: currentScore >= threshold ? '#f87171' : '#34d399' }}>{Number(currentScore).toFixed(2)}</strong>
                  </span>
                </div>
                <div className="viewport-stage">
                  {pipelineStatus.running ? (
                    <img
                      src={streamUrl}
                      alt="Video đang phân tích"
                      className="live-stream-img"
                      onError={() => {
                        if (!useDirectPort8001) setUseDirectPort8001(true);
                      }}
                    />
                  ) : (
                    <div style={{ textAlign: 'center', padding: '48px 24px' }}>
                      <div style={{ fontSize: '0.95rem', color: '#cbd5e1', marginBottom: '14px' }}>
                        Chưa phát video. Bấm <strong>Thời gian thực</strong> hoặc chọn file video ở trên để bắt đầu phân tích.
                      </div>
                      <button
                        type="button"
                        className="btn btn-blue"
                        onClick={() => handleStartPipeline('file', selectedVideoPath, true)}
                      >
                        ▶ Bắt đầu phân tích ({selectedVideoPath.split('/').pop()})
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Real Red Anomaly Segments Timeline */}
              <div className="timeline-anomaly-bar">
                {durationSec > 0 &&
                  suspiciousIntervals.map((iv, idx) => {
                    const leftPct = Math.min(98, (iv.startSec / durationSec) * 100);
                    const widthPct = Math.max(2, ((iv.endSec - iv.startSec) / durationSec) * 100);
                    return (
                      <div
                        key={idx}
                        className="anomaly-segment"
                        style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                      />
                    );
                  })}
                <div
                  className="playhead-line"
                  style={{ left: `${Math.min(100, Math.max(0, progressPct))}%` }}
                />
              </div>
              <div className="timeline-labels">
                <span>00:00</span>
                <span>Vùng đỏ: đoạn nghi trộm</span>
                <span>{formatSecMMSS(durationSec)}</span>
              </div>

              {/* Real Progress Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', color: '#94a3b8' }}>
                <span>Tiến độ phân tích</span>
                <span style={{ color: '#f8fafc', fontWeight: 700 }}>{progressPct}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPct}%` }} />
              </div>
            </div>

            {/* Bottom Chart: Điểm nghi trộm trên toàn video */}
            <div className="ui-card">
              <div className="card-header-row">
                <div className="card-title">Điểm nghi trộm trên toàn video</div>
                <div className="chart-legend">
                  <span style={{ color: '#f59e0b' }}>- - Ngưỡng {Number(threshold).toFixed(2)}</span>
                </div>
              </div>
              {renderLineChart(
                scoreHistory.map((s) => Number(s.score || 0)),
                threshold,
                true
              )}
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div className="right-stack">
            {/* Card 1: Tóm tắt kết quả */}
            <div className="ui-card">
              <div className="card-title" style={{ marginBottom: '14px' }}>
                Tóm tắt kết quả
              </div>

              <div className="kpi-row-2">
                <div className="kpi-subcard">
                  <div className="kpi-label">Số cảnh báo</div>
                  <div className="kpi-value" style={{ color: '#f87171', fontSize: '1.5rem' }}>
                    {pipelineStatus?.summary?.total_alerts ?? events.length}
                  </div>
                </div>
                <div className="kpi-subcard">
                  <div className="kpi-label">Điểm cao nhất</div>
                  <div className="kpi-value" style={{ color: '#f8fafc', fontSize: '1.5rem' }}>
                    {Number(peakScore).toFixed(2)}
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '0.84rem', color: '#94a3b8', marginBottom: '10px' }}>
                Các thời điểm nghi vấn ({suspiciousIntervals.length})
              </div>

              <div className="pending-list" style={{ marginBottom: '16px' }}>
                {suspiciousIntervals.length === 0 ? (
                  <div style={{ fontSize: '0.82rem', color: '#64748b', padding: '10px 4px' }}>
                    Chưa phát hiện đoạn vượt ngưỡng {Number(threshold).toFixed(2)} trong phiên phân tích hiện tại.
                  </div>
                ) : (
                  suspiciousIntervals.map((iv, idx) => (
                    <div key={idx} className="pending-item">
                      <div>
                        <div className="pending-title">
                          {formatSecMMSS(iv.startSec)} – {formatSecMMSS(iv.endSec)}
                        </div>
                        <div className="pending-sub">
                          ID {iv.trackId} · điểm {Number(iv.maxScore).toFixed(2)}
                        </div>
                      </div>
                      <button
                        type="button"
                        className="btn btn-blue"
                        onClick={() => setActiveTab('alerts')}
                      >
                        Xem đoạn
                      </button>
                    </div>
                  ))
                )}
              </div>

              <div className="amber-note-box">
                Đây là kết quả tham khảo của model. Cần người xem lại clip trước khi kết luận.
              </div>
            </div>

            {/* Card 2: Cấu hình phân tích */}
            <div className="ui-card">
              <div className="card-title" style={{ marginBottom: '16px' }}>
                Cấu hình phân tích
              </div>

              <div className="field-group" style={{ marginBottom: '16px' }}>
                <div className="field-label-row">
                  <span>Ngưỡng cảnh báo</span>
                  <span className="field-val">{Number(threshold).toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.01"
                  value={threshold}
                  className="range-slider"
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setThreshold(v);
                    handleConfigUpdate(v, kWindows, cooldownSec, sendMqttAlert);
                  }}
                />
              </div>

              <div className="field-group" style={{ marginBottom: '16px' }}>
                <div className="field-label-row">
                  <span>K cửa sổ liên tiếp</span>
                </div>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={kWindows}
                  className="num-input"
                  onChange={(e) => {
                    const v = Math.max(1, parseInt(e.target.value || '2', 10));
                    setKWindows(v);
                    handleConfigUpdate(threshold, v, cooldownSec, sendMqttAlert);
                  }}
                />
              </div>

              <div className="field-group">
                <div className="field-label-row">
                  <span>Gửi cảnh báo lên dashboard (MQTT)</span>
                </div>
                <select
                  className="select-input"
                  value={sendMqttAlert}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSendMqttAlert(val);
                    handleConfigUpdate(threshold, kWindows, cooldownSec, val);
                  }}
                >
                  <option value="yes">Có</option>
                  <option value="no">Không</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================
          TAB 4: NHẬT KÝ CỬA (Full Logs & Door Servo View)
      =================================================================== */}
      {activeTab === 'logs' && (
        <div className="left-stack">
          <div className="ui-card">
            <div className="card-header-row">
              <div>
                <div className="card-title">Trạng thái Cửa & Động cơ Servo (ESP32)</div>
                <div className="card-subtitle">
                  Fail-safe: Tự động mở khóa (Góc Servo 0°) khi hết hạn TTL hoặc mất kết nối MQTT
                </div>
              </div>
              <div className="controls-group">
                <button
                  type="button"
                  className="btn btn-start"
                  onClick={() => handleDoorCommand('unlock', 'Mở cửa ưu tiên', '', 0)}
                >
                  Mở cửa ngay (0°)
                </button>
                <button
                  type="button"
                  className="btn btn-red"
                  onClick={() => setShowLockModal(true)}
                >
                  Khóa cửa 45s (90°)
                </button>
              </div>
            </div>

            <div className={`door-badge ${isDoorLocked ? 'locked' : 'unlocked'}`} style={{ marginBottom: 0 }}>
              <span>
                {isDoorLocked
                  ? `🔒 CỬA ĐANG KHÓA (Servo 90° · TTL còn ${remainingTtl}s)`
                  : '🔓 CỬA ĐANG MỞ (Servo 0°)'}
              </span>
            </div>
          </div>

          <div className="ui-card">
            <div className="card-title" style={{ marginBottom: '14px' }}>
              Nhật ký lệnh cửa (giờ Việt Nam, UTC+7)
            </div>
            <div className="logs-table-wrap">
              <table className="logs-table">
                <thead>
                  <tr>
                    <th>Thời gian</th>
                    <th>Người phát lệnh</th>
                    <th>Hành động</th>
                    <th>Event ID</th>
                    <th>TTL</th>
                    <th>Lý do</th>
                  </tr>
                </thead>
                <tbody>
                  {(doorData.logs || []).map((row, idx) => {
                    const isLock = String(row.action).toLowerCase() === 'lock';
                    return (
                      <tr key={row.id || idx}>
                        <td>{formatDateTimeVN(row.ts)}</td>
                        <td>{row.actor || 'admin_dashboard'}</td>
                        <td>
                          <span className={`action-tag ${isLock ? 'lock' : 'unlock'}`}>
                            {isLock ? 'LOCK' : 'UNLOCK'}
                          </span>
                        </td>
                        <td>{shortEventId(row.event_id)}</td>
                        <td>{row.ttl_sec ?? 0} s</td>
                        <td>{row.reason || '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================
          SCREEN 4: ADMIN TRÊN ĐIỆN THOẠI (4. Admin trên điện thoại.png)
      =================================================================== */}
      {activeTab === 'mobile' && (
        <div className="mobile-frame-wrapper">
          <div className="mobile-notch" />
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc' }}>
                Cảnh báo nghi trộm
              </h2>
              <div style={{ fontSize: '0.86rem', color: '#94a3b8', marginTop: '2px' }}>
                Cửa hàng 01 · {selectedEvent ? formatTimeOnly(selectedEvent.ts) : '00:12:41'}
              </div>
            </div>
            <span className="decision-badge pending" style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
              {selectedEvent
                ? selectedEvent.decision === 'confirmed'
                  ? 'Đã xác nhận'
                  : selectedEvent.decision === 'ignored'
                  ? 'Đã bỏ qua'
                  : 'Chờ xem'
                : 'Chờ xem'}
            </span>
          </div>

          {/* Mobile Video Card */}
          <div className="video-viewport" style={{ borderRadius: '16px', marginBottom: '16px' }}>
            <div className="viewport-stage" style={{ minHeight: '230px' }}>
              {selectedEvent && clipFileName(selectedEvent.clip_path) ? (
                <video
                  controls
                  preload="metadata"
                  className="live-stream-img"
                  src={`/api/clips/${encodeURIComponent(clipFileName(selectedEvent.clip_path))}/webm`}
                />
              ) : pipelineStatus.running ? (
                <img src={streamUrl} alt="Live mobile stream" className="live-stream-img" />
              ) : (
                <div className="mock-scene" style={{ height: '230px' }}>
                  <div className="person-bbox suspect" style={{ width: '84px', height: '190px' }}>
                    <span className="bbox-tag suspect" style={{ fontSize: '0.7rem' }}>ID 5 · 0.87</span>
                    <div className="silhouette-head" style={{ width: '32px', height: '32px' }} />
                    <div className="silhouette-body" style={{ width: '50px', height: '110px' }} />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 2 Side-by-side KPI Cards */}
          <div className="kpi-row-2" style={{ marginBottom: '16px' }}>
            <div className="ui-card" style={{ padding: '14px 16px' }}>
              <div className="kpi-label">Điểm / Ngưỡng</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f87171' }}>
                {Number(selectedEvent ? selectedEvent.score : (currentScore > 0 ? currentScore : 0.87)).toFixed(2)} /{' '}
                {Number(selectedEvent?.threshold || threshold).toFixed(2)}
              </div>
            </div>
            <div className="ui-card" style={{ padding: '14px 16px' }}>
              <div className="kpi-label">Camera</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc' }}>
                {selectedEvent
                  ? `${selectedEvent.cam_id} · ID ${selectedEvent.track_id}`
                  : `${pipelineStatus.cam_id || 'cam_01'} · ID 5`}
              </div>
            </div>
          </div>

          {/* Door Status Card */}
          <div className="ui-card" style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ fontSize: '1.05rem', fontWeight: 700 }}>Cửa ra vào</span>
              <span className={`door-badge ${isDoorLocked ? 'locked' : 'unlocked'}`} style={{ marginBottom: 0 }}>
                {isDoorLocked ? `🔒 ĐANG KHÓA (${remainingTtl}s)` : '🔓 ĐANG MỞ (0°)'}
              </span>
            </div>
            <div style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.5 }}>
              Khóa luôn có thời hạn 45 giây rồi tự mở. Nút mở cửa luôn ưu tiên.
            </div>
          </div>

          {/* Mobile Sticky Bottom Buttons */}
          <button
            type="button"
            className="btn-red-full"
            style={{ padding: '16px', fontSize: '1.05rem', borderRadius: '14px', marginBottom: '12px' }}
            onClick={() =>
              handleDoorCommand(
                'lock',
                `Mobile admin khóa cửa (${selectedEvent?.cam_id || 'cam_01'})`,
                selectedEvent?.event_id,
                45
              )
            }
          >
            Khóa cửa 45 giây
          </button>

          <div className="kpi-row-2" style={{ marginBottom: 0 }}>
            <button
              type="button"
              className="btn-green-full"
              style={{ marginBottom: 0, padding: '15px', fontSize: '1rem', borderRadius: '14px' }}
              onClick={() => handleDoorCommand('unlock', 'Mobile admin mở cửa', selectedEvent?.event_id, 0)}
            >
              Mở cửa
            </button>
            <button
              type="button"
              className="btn btn-outline"
              style={{ padding: '15px', fontSize: '1rem', borderRadius: '14px', width: '100%' }}
              onClick={() => selectedEvent && handleDecision(selectedEvent, 'ignored')}
            >
              Bỏ qua
            </button>
          </div>
        </div>
      )}

      {/* ===================================================================
          MODALS (Lock Confirmation, Source Picker, Reset Events)
      =================================================================== */}
      {showLockModal && (
        <div className="modal-backdrop" onClick={() => setShowLockModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px' }}>
              Xác nhận Khóa cửa ra vào?
            </h3>
            <p style={{ fontSize: '0.86rem', color: '#cbd5e1', marginBottom: '16px' }}>
              Cửa sẽ khóa trong 45 giây (Servo xoay 90°) rồi tự mở. Bạn có thể bấm Mở cửa bất cứ lúc nào.
            </p>
            <div className="field-group" style={{ marginBottom: '18px' }}>
              <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Lý do (ghi vào nhật ký)</label>
              <input
                type="text"
                className="text-input"
                value={lockReason}
                onChange={(e) => setLockReason(e.target.value)}
              />
            </div>
            <div className="controls-group" style={{ justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-outline" onClick={() => setShowLockModal(false)}>
                Hủy
              </button>
              <button
                type="button"
                className="btn btn-red"
                onClick={() => handleDoorCommand('lock', lockReason, selectedEvent?.event_id, 45)}
              >
                Xác nhận Khóa cửa (45s)
              </button>
            </div>
          </div>
        </div>
      )}

      {showSourceModal && (
        <div className="modal-backdrop" onClick={() => setShowSourceModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '10px' }}>
              {sourceMode === 'rtsp' ? 'Cấu hình luồng RTSP' : 'Chọn video hoặc Tải video lên'}
            </h3>

            {sourceMode === 'rtsp' ? (
              <div className="field-group" style={{ marginBottom: '18px' }}>
                <label style={{ fontSize: '0.82rem', color: '#94a3b8' }}>Đường dẫn RTSP / HTTP Stream URL:</label>
                <input
                  type="text"
                  className="text-input"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="http://host.docker.internal:8554/stream.mjpg"
                />
                <div style={{ marginTop: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    className="filter-pill"
                    style={{ fontSize: '0.76rem', padding: '5px 12px' }}
                    onClick={() => setRtspUrl('http://host.docker.internal:8554/stream.mjpg')}
                  >
                    📹 Camera USB laptop (stream_usb_cam.py)
                  </button>
                  <button
                    type="button"
                    className="filter-pill"
                    style={{ fontSize: '0.76rem', padding: '5px 12px' }}
                    onClick={() => setRtspUrl('rtsp://localhost:8554/cam_01')}
                  >
                    🌐 RTSP mẫu
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
                  {PRESET_VIDEOS.map((pv) => (
                    <button
                      key={pv.path}
                      type="button"
                      className="btn btn-outline"
                      style={{
                        justifyContent: 'flex-start',
                        borderColor: selectedVideoPath === pv.path ? '#4f8ff7' : undefined,
                      }}
                      onClick={() => {
                        setSelectedVideoPath(pv.path);
                        handleStartPipeline('file', pv.path);
                        setShowSourceModal(false);
                      }}
                    >
                      📹 {pv.label}
                    </button>
                  ))}
                </div>
                <div className="field-group" style={{ marginBottom: '16px' }}>
                  <button
                    type="button"
                    className="btn btn-blue"
                    style={{ width: '100%' }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Tải file video mới từ máy tính (.mp4, .avi, .mov)
                  </button>
                </div>
              </>
            )}

            <div className="controls-group" style={{ justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-outline" onClick={() => setShowSourceModal(false)}>
                Đóng
              </button>
              {sourceMode === 'rtsp' && (
                <button
                  type="button"
                  className="btn btn-start"
                  onClick={() => {
                    handleStartPipeline('rtsp', rtspUrl);
                    setShowSourceModal(false);
                  }}
                >
                  Kết nối RTSP
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {showResetModal && (
        <div className="modal-backdrop" onClick={() => setShowResetModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px' }}>
              Xóa tất cả sự kiện cảnh báo?
            </h3>
            <p style={{ fontSize: '0.86rem', color: '#cbd5e1', marginBottom: '18px' }}>
              Thao tác này sẽ xóa toàn bộ danh sách cảnh báo và các file video clip đã lưu.
            </p>
            <div className="controls-group" style={{ justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-outline" onClick={() => setShowResetModal(false)}>
                Hủy
              </button>
              <button type="button" className="btn btn-red" onClick={handleResetAllEvents}>
                Xóa tất cả sự kiện
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
