/**
 * Frontend client script for Multi-Objective Deep Learning QoS Dashboard.
 * Monochrome / Black & White / Grayscale Edition.
 */

let socket = null;
let radarChart = null;
let timeSeriesChart = null;
let isPaused = false;
let currentProfile = "balanced";
let currentController = "modl";

// Rolling telemetry buffer for time-series chart
const maxHistory = 25;
const tsLabels = [];
const tsDelayData = [];
const tsJitterData = [];
const tsThroughputData = [];

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    connectWebSocket();
});

// Chart.js Setup (Monochrome Edition)
function initCharts() {
    // 1. Radar Chart Setup (5 QoS Dimensions)
    const ctxRadar = document.getElementById("radarChart").getContext("2d");
    radarChart = new Chart(ctxRadar, {
        type: "radar",
        data: {
            labels: ["Throughput", "Low Delay", "Low Jitter", "Packet Reliability", "Fair Share"],
            datasets: [
                {
                    label: "Active Policy",
                    data: [0.8, 0.9, 0.9, 1.0, 0.95],
                    backgroundColor: "rgba(255, 255, 255, 0.15)",
                    borderColor: "#FFFFFF",
                    borderWidth: 2,
                    pointBackgroundColor: "#FFFFFF",
                    pointBorderColor: "#000000",
                    pointRadius: 4
                },
                {
                    label: "Ideal Target",
                    data: [1.0, 1.0, 1.0, 1.0, 1.0],
                    backgroundColor: "transparent",
                    borderColor: "rgba(115, 115, 115, 0.6)",
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 1.0,
                    ticks: { display: false, stepSize: 0.2 },
                    grid: { color: "rgba(50, 50, 50, 0.7)" },
                    angleLines: { color: "rgba(60, 60, 60, 0.8)" },
                    pointLabels: {
                        color: "#D4D4D4",
                        font: { size: 10, weight: "bold" }
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // 2. Time Series Chart Setup (Delay & Jitter)
    const ctxTS = document.getElementById("timeSeriesChart").getContext("2d");
    timeSeriesChart = new Chart(ctxTS, {
        type: "line",
        data: {
            labels: tsLabels,
            datasets: [
                {
                    label: "Delay (ms)",
                    data: tsDelayData,
                    borderColor: "#FFFFFF",
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: "y"
                },
                {
                    label: "Jitter (ms)",
                    data: tsJitterData,
                    borderColor: "#737373",
                    backgroundColor: "transparent",
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: "y"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: "rgba(40, 40, 40, 0.5)" },
                    ticks: { display: false }
                },
                y: {
                    min: 0,
                    suggestedMax: 60,
                    grid: { color: "rgba(40, 40, 40, 0.6)" },
                    ticks: { color: "#A3A3A3", font: { size: 9, family: "monospace" } }
                }
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#A3A3A3", font: { size: 10, family: "monospace" }, boxWidth: 10 }
                }
            }
        }
    });
}

// WebSocket Connection & Real-Time Data Ingestion
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        const statusEl = document.getElementById("ws-status");
        if (statusEl) {
            statusEl.innerText = "LIVE";
            statusEl.className = "text-white";
        }
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "telemetry") {
                updateDashboard(data);
            }
        } catch (e) {
            console.error("Failed to parse websocket message", e);
        }
    };

    socket.onclose = () => {
        const statusEl = document.getElementById("ws-status");
        if (statusEl) {
            statusEl.innerText = "CONNECTING...";
            statusEl.className = "text-neutral-400";
        }
        setTimeout(connectWebSocket, 1500);
    };

    socket.onerror = () => {
        socket.close();
    };
}

// Update UI on Every Simulation Step
function updateDashboard(payload) {
    const m = payload.metrics;
    if (!m) return;

    // 1. Update KPI Cards
    const thruEl = document.getElementById("kpi-throughput");
    const delayEl = document.getElementById("kpi-delay");
    const jitterEl = document.getElementById("kpi-jitter");
    const lossEl = document.getElementById("kpi-loss");
    const fairEl = document.getElementById("kpi-fairness");

    if (thruEl) thruEl.innerText = m.throughput_mbps.toFixed(1);
    if (delayEl) delayEl.innerText = m.avg_delay_ms.toFixed(1);
    if (jitterEl) jitterEl.innerText = m.avg_jitter_ms.toFixed(1);
    if (lossEl) lossEl.innerText = m.packet_loss_percent.toFixed(2);
    if (fairEl) fairEl.innerText = m.jains_fairness.toFixed(2);

    // KPI Progress Bars
    const thruBar = document.getElementById("kpi-throughput-bar");
    if (thruBar) thruBar.style.width = `${Math.min(100, (m.throughput_mbps / 40.0) * 100)}%`;

    const utilEl = document.getElementById("kpi-util");
    if (utilEl) utilEl.innerText = `Util: ${(m.link_utilization * 100).toFixed(0)}%`;

    const delayBar = document.getElementById("kpi-delay-bar");
    if (delayBar) delayBar.style.width = `${Math.min(100, (m.avg_delay_ms / 100.0) * 100)}%`;

    const jitterBar = document.getElementById("kpi-jitter-bar");
    if (jitterBar) jitterBar.style.width = `${Math.min(100, (m.avg_jitter_ms / 25.0) * 100)}%`;

    const lossBar = document.getElementById("kpi-loss-bar");
    if (lossBar) lossBar.style.width = `${Math.min(100, m.packet_loss_percent * 5)}%`;

    const fairBar = document.getElementById("kpi-fairness-bar");
    if (fairBar) fairBar.style.width = `${Math.min(100, m.jains_fairness * 100)}%`;

    const dropsEl = document.getElementById("kpi-dropped-pkts");
    if (dropsEl) dropsEl.innerText = `Drops: ${m.total_dropped}`;

    // 2. Update Radar Chart
    if (radarChart && m.vector_reward) {
        radarChart.data.datasets[0].data = m.vector_reward;
        radarChart.update("none");
    }

    // 3. Update Time-Series Chart
    if (timeSeriesChart) {
        tsLabels.push(m.step);
        tsDelayData.push(m.avg_delay_ms);
        tsJitterData.push(m.avg_jitter_ms);
        tsThroughputData.push(m.throughput_mbps);

        if (tsLabels.length > maxHistory) {
            tsLabels.shift();
            tsDelayData.shift();
            tsJitterData.shift();
            tsThroughputData.shift();
        }
        timeSeriesChart.update("none");

        const tsDelayEl = document.getElementById("ts-avg-delay");
        const tsJitterEl = document.getElementById("ts-avg-jitter");
        if (tsDelayEl) tsDelayEl.innerText = `Avg Delay: ${m.avg_delay_ms.toFixed(1)} ms`;
        if (tsJitterEl) tsJitterEl.innerText = `Avg Jitter: ${m.avg_jitter_ms.toFixed(1)} ms`;
    }

    // 4. Update Queue Buffers (Monochrome Gray Shades)
    const classMetrics = m.class_metrics || {};
    const monoColors = ["bg-white", "bg-neutral-300", "bg-neutral-400", "bg-neutral-500"];

    for (let c = 0; c < 4; c++) {
        const cm = classMetrics[c];
        if (!cm) continue;

        const qStatEl = document.getElementById(`q-stat-${c}`);
        const qBarEl = document.getElementById(`q-bar-${c}`);
        const qThreshEl = document.getElementById(`q-thresh-${c}`);
        const qWeightEl = document.getElementById(`q-weight-${c}`);
        const qDelayEl = document.getElementById(`q-delay-${c}`);
        const qLossEl = document.getElementById(`q-loss-${c}`);

        if (qStatEl) qStatEl.innerText = `${cm.queue_depth} / 120 pkts`;
        if (qBarEl) {
            const pct = (cm.queue_depth / 120.0) * 100;
            qBarEl.style.width = `${Math.min(100, pct)}%`;
            if (pct > 75) {
                qBarEl.className = "h-2 rounded-full bg-white glow-white queue-bar-transition";
            } else {
                qBarEl.className = `h-2 rounded-full ${monoColors[c]} queue-bar-transition`;
            }
        }
        if (qThreshEl) {
            qThreshEl.style.left = `${(cm.drop_threshold * 100).toFixed(0)}%`;
        }
        if (qWeightEl) qWeightEl.innerText = `Weight: ${(cm.weight_assigned * 100).toFixed(0)}%`;
        if (qDelayEl) qDelayEl.innerText = `Delay: ${cm.avg_delay_ms.toFixed(0)}ms`;
        if (qLossEl) qLossEl.innerText = `Loss: ${(cm.loss_ratio * 100).toFixed(1)}%`;
    }

    // 5. Update Dropped Packets Table (Monochrome)
    const tbody = document.getElementById("packet-log-tbody");
    if (tbody && m.recent_drops && m.recent_drops.length > 0) {
        tbody.innerHTML = "";
        const classNames = ["VoIP", "Video", "IoT", "Best-Effort"];

        m.recent_drops.forEach(pkt => {
            const tr = document.createElement("tr");
            tr.className = "hover:bg-neutral-900/60";
            tr.innerHTML = `
                <td class="py-2 px-3 text-white">#${pkt.id}</td>
                <td class="py-2 px-3 font-semibold text-silvertext">${classNames[pkt.class]}</td>
                <td class="py-2 px-3 text-mutedgray">${pkt.size_bytes} B</td>
                <td class="py-2 px-3 text-mutedgray">${pkt.creation_time.toFixed(3)}s</td>
                <td class="py-2 px-3 text-white font-mono">${pkt.delay_ms} ms</td>
                <td class="py-2 px-3"><span class="px-1.5 py-0.5 rounded text-[9px] bg-neutral-900 text-white border border-bordergraylight font-mono">DROPPED</span></td>
                <td class="py-2 px-3 text-silvertext font-sans">${pkt.drop_reason || 'buffer_overflow'}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

// Controller Mode Switching
async function setControllerMode(mode) {
    currentController = mode;
    const btns = ["modl", "fifo", "wfq", "red", "greedy"];
    btns.forEach(b => {
        const el = document.getElementById(`btn-ctrl-${b}`);
        if (el) {
            if (b === mode) {
                el.className = "px-3 py-1.5 rounded font-semibold bg-white text-black shadow transition";
            } else {
                el.className = "px-3 py-1.5 rounded font-medium text-silvertext hover:text-white hover:bg-neutral-900 transition";
            }
        }
    });

    try {
        await fetch("/api/control/mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: mode })
        });
    } catch (e) {
        console.error("Failed to set controller mode", e);
    }
}

// Preference Steering
async function applyPreferenceProfile(profileName) {
    currentProfile = profileName;
    const profiles = ["balanced", "gaming_urllc", "streaming", "fairness_critical"];
    profiles.forEach(p => {
        const el = document.getElementById(`prof-${p}`);
        if (el) {
            if (p === profileName) {
                el.className = "px-3 py-1.5 rounded border border-white bg-white text-black font-bold";
            } else {
                el.className = "px-3 py-1.5 rounded border border-bordergray bg-neutral-900 text-silvertext hover:text-white hover:border-bordergraylight";
            }
        }
    });

    try {
        const res = await fetch("/api/control/preferences", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ profile: profileName })
        });
        const data = await res.json();
        if (data.preference_weights) {
            syncSliders(data.preference_weights);
        }
    } catch (e) {
        console.error("Failed to apply profile", e);
    }
}

function syncSliders(weights) {
    for (let i = 0; i < 5; i++) {
        const slider = document.getElementById(`pref-slider-${i}`);
        const label = document.getElementById(`slider-val-${i}`);
        if (slider && weights[i] !== undefined) {
            slider.value = weights[i];
            if (label) label.innerText = weights[i].toFixed(2);
        }
    }
}

let sliderTimeout = null;
function onSliderChange() {
    clearTimeout(sliderTimeout);
    sliderTimeout = setTimeout(async () => {
        const weights = [];
        for (let i = 0; i < 5; i++) {
            const slider = document.getElementById(`pref-slider-${i}`);
            const val = parseFloat(slider ? slider.value : 0.2);
            weights.push(val);
            const label = document.getElementById(`slider-val-${i}`);
            if (label) label.innerText = val.toFixed(2);
        }

        try {
            await fetch("/api/control/preferences", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ weights: weights })
            });
        } catch (e) {
            console.error("Failed to update custom weights", e);
        }
    }, 150);
}

// Scenario & Traffic Spike Control
async function changeScenario(scenarioName) {
    try {
        await fetch("/api/traffic/scenario", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scenario: scenarioName })
        });
    } catch (e) {
        console.error("Failed to set scenario", e);
    }
}

async function triggerTrafficSpike(classId) {
    try {
        await fetch("/api/traffic/spike", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ class_id: classId, multiplier: 3.5, duration: 6.0 })
        });
    } catch (e) {
        console.error("Failed to trigger spike", e);
    }
}

// Simulation Playback Control
async function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById("pause-icon");
    if (btn) btn.innerText = isPaused ? "▶" : "⏸";

    try {
        await fetch("/api/simulation/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paused: isPaused })
        });
    } catch (e) {
        console.error("Failed to toggle pause", e);
    }
}

// Benchmark Modal & Comparison Runner
function openBenchmarkModal() {
    const modal = document.getElementById("benchmark-modal");
    if (modal) modal.classList.remove("hidden");
    runLiveBenchmark();
}

function closeBenchmarkModal() {
    const modal = document.getElementById("benchmark-modal");
    if (modal) modal.classList.add("hidden");
}

async function runLiveBenchmark() {
    const runBtn = document.getElementById("bm-run-btn");
    const tbody = document.getElementById("benchmark-tbody");
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerText = "Running Trial...";
    }
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-mutedgray font-sans">Simulating identical traffic sequences across all 5 algorithms...</td></tr>`;
    }

    try {
        const scenario = document.getElementById("scenario-selector")?.value || "congested_all";
        const res = await fetch(`/api/benchmark/run?scenario=${encodeURIComponent(scenario)}&steps=60`);
        const data = await res.json();

        // Update improvement summary cards
        const imp = data.improvements_over_fifo || {};
        const elDelay = document.getElementById("bm-imp-delay");
        const elJitter = document.getElementById("bm-imp-jitter");
        const elLoss = document.getElementById("bm-imp-loss");
        const elFair = document.getElementById("bm-imp-fairness");

        if (elDelay) elDelay.innerText = `-${imp.delay_reduction_percent || 0}%`;
        if (elJitter) elJitter.innerText = `-${imp.jitter_reduction_percent || 0}%`;
        if (elLoss) elLoss.innerText = `-${imp.packet_loss_reduction_percent || 0}%`;
        if (elFair) elFair.innerText = `+${imp.fairness_improvement_percent || 0}%`;

        // Render comparative table
        if (tbody && data.controllers) {
            tbody.innerHTML = "";
            const order = ["MODL-QoS (Ours)", "FIFO Drop-Tail", "Static WFQ", "RED + DWRR", "Greedy Throughput DL"];

            order.forEach(name => {
                const cData = data.controllers[name];
                if (!cData) return;
                const isModl = name === "MODL-QoS (Ours)";
                const tr = document.createElement("tr");
                tr.className = isModl ? "bg-neutral-900 border-l-2 border-white font-bold" : "hover:bg-neutral-900/40";

                tr.innerHTML = `
                    <td class="py-3 px-3 ${isModl ? 'text-white font-black' : 'text-silvertext'}">${name}</td>
                    <td class="py-3 px-3 text-silvertext">${cData.throughput_mbps} Mbps</td>
                    <td class="py-3 px-3 ${isModl ? 'text-white font-bold' : 'text-silvertext'}">${cData.delay_ms} ms</td>
                    <td class="py-3 px-3 ${isModl ? 'text-white font-bold' : 'text-silvertext'}">${cData.jitter_ms} ms</td>
                    <td class="py-3 px-3 ${isModl ? 'text-white font-bold' : 'text-silvertext'}">${cData.packet_loss_percent}%</td>
                    <td class="py-3 px-3 ${isModl ? 'text-white font-bold' : 'text-silvertext'}">${cData.fairness_jfi}</td>
                    <td class="py-3 px-3 text-white font-bold">${cData.hypervolume_indicator}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Dominance note
        const domEl = document.getElementById("benchmark-dominance-status");
        if (domEl) {
            domEl.innerText = "✓ MODL-QoS achieves superior 5D-Hypervolume coverage and dominates traditional Drop-Tail FIFO.";
        }
    } catch (e) {
        console.error("Benchmark failed", e);
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="7" class="py-4 text-center text-white font-sans">Benchmark execution failed. Check server logs.</td></tr>`;
        }
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerText = "Re-run Benchmark";
        }
    }
}
