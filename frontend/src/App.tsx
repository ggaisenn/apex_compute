import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { getSnapshot, formatNumber, sizes, snapshotFromImportedLog } from "./data";
import Icon from "./components/Icon";
import { ThreeScene } from "./components/ThreeScene";
import type { BenchmarkSnapshot, Kernel, MatrixSize, Page, RunMode } from "./types";

const navItems: Array<{ id: Page; label: string; detail: string; icon: "gauge" | "layers" | "cpu" | "users" }> = [
  { id: "home", label: "Overview", detail: "home", icon: "gauge" },
  { id: "execution", label: "Execution flow", detail: "01", icon: "layers" },
  { id: "simd", label: "SIMD kernels", detail: "02", icon: "cpu" },
  { id: "threads", label: "Thread pool", detail: "03", icon: "users" },
];

const kernels: Array<{ value: Kernel; label: string; note: string }> = [
  { value: "scalar", label: "Scalar", note: "fallback" },
  { value: "neon-4x4", label: "NEON 4×4", note: "arm64" },
  { value: "avx512", label: "AVX-512", note: "x86_64" },
];

const commandFor = (size: MatrixSize, kernel: Kernel, mode: RunMode) =>
  `python3 apex-bench.py --sizes ${size} --kernels ${kernel === "neon-4x4" ? "auto" : kernel} --variants ${mode === "naive" ? "naive,engine" : "engine,naive"}`;

const Metric = ({ label, value, unit, accent = "cyan", footnote }: { label: string; value: string; unit?: string; accent?: string; footnote?: string }) => (
  <div className={`metric metric-${accent}`}>
    <div className="metric-label">{label}</div>
    <div className="metric-value">{value}<span>{unit}</span></div>
    {footnote && <div className="metric-footnote">{footnote}</div>}
  </div>
);

const PageTitle = ({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) => (
  <div className="page-title">
    <div>
      <div className="eyebrow"><span className="eyebrow-dot" />{eyebrow}</div>
      <h1>{title}</h1>
    </div>
    {children && <div className="page-title-actions">{children}</div>}
  </div>
);

const PanelHeading = ({ label, action }: { label: string; action?: ReactNode }) => (
  <div className="panel-heading"><span>{label}</span>{action}</div>
);

const StatusPill = ({ children, tone = "green" }: { children: ReactNode; tone?: "green" | "amber" | "muted" }) => (
  <span className={`status-pill status-${tone}`}><span className="status-dot" />{children}</span>
);

const Sparkline = ({ color = "cyan" }: { color?: "cyan" | "amber" | "violet" }) => (
  <svg aria-hidden="true" className={`sparkline sparkline-${color}`} preserveAspectRatio="none" viewBox="0 0 240 48">
    <path d="M0 40 C18 39 18 32 34 34S52 25 68 28s16-11 31-9 15 12 31 6 16-15 30-8 17-4 28-10 29-1 52-7" />
  </svg>
);

const RegisterMap = ({ kernel }: { kernel: Kernel }) => {
  const count = kernel === "avx512" ? 16 : kernel === "neon-4x4" ? 8 : 4;
  return (
    <div className={`register-map register-${kernel}`}>
      {Array.from({ length: count }, (_, index) => (
        <div className={`register-cell ${index % 4 === 0 ? "register-cell-hot" : ""}`} key={index}>
          <span>{kernel === "scalar" ? `a${index}` : `v${index}`}</span>
          <b>{((index * 7 + 3) % 19 / 10 + 0.2).toFixed(1)}</b>
        </div>
      ))}
    </div>
  );
};

const ThreadGrid = ({ selectedThread }: { selectedThread: number }) => (
  <div className="thread-grid" aria-label="Matrix C divided between workers">
    {Array.from({ length: 48 }, (_, index) => {
      const thread = index % 4;
      return <div className={`thread-cell thread-${thread} ${thread === selectedThread ? "thread-selected" : ""}`} key={index} />;
    })}
  </div>
);

const App = () => {
  const [page, setPage] = useState<Page>("home");
  const [size, setSize] = useState<MatrixSize>(256);
  const [kernel, setKernel] = useState<Kernel>("neon-4x4");
  const [mode, setMode] = useState<RunMode>("engine");
  const [playing, setPlaying] = useState(true);
  const [step, setStep] = useState(3);
  const [speed, setSpeed] = useState(64);
  const [snapshot, setSnapshot] = useState<BenchmarkSnapshot>(() => getSnapshot(256, "neon-4x4"));
  const [notice, setNotice] = useState("Demo data loaded from the latest local benchmark export");
  const [selectedThread, setSelectedThread] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setStep((current) => (current + 1) % 12), Math.max(120, 950 - speed * 8));
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  const updateSize = useCallback((value: number) => {
    const nextSize = value as MatrixSize;
    setSize(nextSize);
    setStep(0);
    setSnapshot(getSnapshot(nextSize, kernel));
    setNotice(`Preview updated for ${nextSize} × ${nextSize} × ${nextSize}`);
  }, [kernel]);

  const updateKernel = useCallback((value: Kernel) => {
    setKernel(value);
    setSnapshot(getSnapshot(size, value));
    setNotice(`${value} micro-kernel selected`);
  }, [size]);

  const handleRun = () => {
    setPlaying(true);
    setStep(0);
    setNotice(`Streaming visual simulation for ${size} × ${size} × ${size}`);
    window.setTimeout(() => setNotice("Simulation complete · metrics are ready to inspect"), 1800);
  };

  const handleNextTile = () => {
    setPlaying(false);
    setStep((value) => (value + 1) % 12);
    setNotice("Paused on the next accumulator tile");
  };

  const handleReset = () => {
    setPlaying(false);
    setStep(0);
    setNotice("Animation reset to tile 01");
  };

  const handleImport = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = snapshotFromImportedLog(JSON.parse(await file.text()));
      if (!parsed) throw new Error("No benchmark rows found");
      setSize(parsed.size);
      setKernel(parsed.kernel);
      setSnapshot(parsed);
      setNotice(`Imported ${file.name} · showing measured runner facts`);
    } catch {
      setNotice("Could not read that file. Choose an apex-bench JSON export.");
    } finally {
      event.target.value = "";
    }
  };

  const command = useMemo(() => commandFor(size, kernel, mode), [kernel, mode, size]);
  const activeTile = (step % 12) + 1;

  const renderHome = () => (
    <>
      <PageTitle eyebrow="APEX COMPUTE / CONTROL ROOM" title="See every cycle." >
        <StatusPill>engine online</StatusPill>
        <button className="button button-ghost" onClick={() => fileInputRef.current?.click()}><Icon name="upload" size={15} /> Import run</button>
        <input accept=".json,application/json" className="visually-hidden" onChange={handleImport} ref={fileInputRef} type="file" />
      </PageTitle>

      <section className="hero-grid">
        <div className="hero-copy">
          <div className="hero-kicker"><Icon name="spark" size={14} /> Interactive execution visualizer</div>
          <h2>From a <em>matrix</em><br />to a finished tile.</h2>
          <p>Trace how APEX Compute turns a dense GEMM into cache-friendly blocks, SIMD lanes, and disjoint work for every core.</p>
          <div className="hero-actions">
            <button className="button button-primary" onClick={() => { setPage("execution"); setPlaying(true); }}><Icon name="play" size={15} /> Watch execution</button>
            <button className="text-button" onClick={() => setPage("simd")}>Explore the kernel <Icon name="arrow-up-right" size={15} /></button>
          </div>
          <div className="hero-facts"><span><i /> real benchmark schema</span><span><i /> three execution layers</span></div>
        </div>
        <div className="hero-visual panel panel-glow">
          <div className="visual-topline"><span>LIVE ALLOCATION / TILE {String(activeTile).padStart(2, "0")}</span><span className="visual-rec"><i /> 60 FPS</span></div>
          <ThreeScene compact size={size} kernel={kernel} mode={mode} playing={playing} speed={speed} step={step} />
          <div className="visual-legend"><span><i className="legend-cyan" /> A panel</span><span><i className="legend-violet" /> B panel</span><span><i className="legend-amber" /> C accumulator</span></div>
        </div>
      </section>

      <section className="command-panel panel">
        <PanelHeading label="Run a controlled experiment" action={<span className="panel-meta"><Icon name="circle-info" size={14} /> uses local benchmark shape</span>} />
        <div className="command-row">
          <div className="terminal-command"><span className="terminal-prompt">$</span><code>{command}</code><span className="terminal-cursor" /></div>
          <button className="button button-primary button-run" onClick={handleRun}><Icon name="bolt" size={15} /> Run simulation</button>
        </div>
        <div className="control-row">
          <label>Matrix shape <select value={size} onChange={(event) => updateSize(Number(event.target.value))}>{sizes.map((item) => <option key={item} value={item}>{item} × {item} × {item}</option>)}</select></label>
          <label>Kernel <select value={kernel} onChange={(event) => updateKernel(event.target.value as Kernel)}>{kernels.map((item) => <option key={item.value} value={item.value}>{item.label} · {item.note}</option>)}</select></label>
          <div className="mode-toggle"><span>path</span><button className={mode === "naive" ? "is-active" : ""} onClick={() => setMode("naive")}>naive</button><button className={mode === "engine" ? "is-active" : ""} onClick={() => setMode("engine")}>engine</button></div>
          <span className="run-notice">{notice}</span>
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading"><div><div className="eyebrow">THE MECHANISM</div><h3>One multiply. Three layers.</h3></div><button className="text-button" onClick={() => setPage("execution")}>Open full walkthrough <Icon name="arrow-up-right" size={15} /></button></div>
        <div className="mechanism-grid">
          {[
            { number: "01", title: "Cache tiling", text: "Split M × N × K into Mc × Kc × Nc panels that stay warm in L1 and L2.", icon: "layers" as const, color: "cyan", page: "execution" as Page },
            { number: "02", title: "SIMD FMA", text: "Load a register tile, broadcast a scalar, and accumulate multiple lanes per tick.", icon: "cpu" as const, color: "violet", page: "simd" as Page },
            { number: "03", title: "Work partitioning", text: "Give each worker disjoint output tiles so threads never fight over a cache line.", icon: "users" as const, color: "amber", page: "threads" as Page },
          ].map((item) => <button className="mechanism-card" key={item.number} onClick={() => setPage(item.page)}><span className={`mechanism-icon mechanism-${item.color}`}><Icon name={item.icon} size={19} /></span><span className="mechanism-number">{item.number}</span><strong>{item.title}</strong><span className="mechanism-text">{item.text}</span><span className="mechanism-link">inspect mechanism <Icon name="chevron-right" size={14} /></span></button>)}
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel metric-panel"><PanelHeading label="Latest measured run" action={<StatusPill tone="muted">{snapshot.source === "imported" ? "imported" : "local sample"}</StatusPill>} /><div className="metric-grid"><Metric label="Throughput" value={formatNumber(snapshot.gflops, 2)} unit=" GFLOPS" accent="cyan" footnote="engine path" /><Metric label="Speedup" value={`${formatNumber(snapshot.speedup, 2)}×`} accent="violet" footnote="vs naive baseline" /><Metric label="Latency" value={formatNumber(snapshot.latencyMs, 2)} unit=" ms" accent="amber" footnote={`${snapshot.size}³ shape`} /></div><Sparkline /></div>
        <div className="panel facts-panel"><PanelHeading label="Execution facts" action={<Icon name="file" size={16} />} /><div className="facts-list"><div><span>active kernel</span><strong>{kernel}</strong></div><div><span>cache tile</span><strong>{snapshot.tile.mc} × {snapshot.tile.kc} × {snapshot.tile.nc}</strong></div><div><span>threads used</span><strong>{snapshot.threads} workers</strong></div><div><span>input source</span><strong>{snapshot.source === "imported" ? "apex-bench JSON" : "local snapshot"}</strong></div></div></div>
      </section>
    </>
  );

  const renderExecution = () => (
    <>
      <PageTitle eyebrow="MECHANISM 01 / CACHE TILING" title="Execution flow.">
        <button aria-pressed={playing} className={`button button-ghost ${playing ? "is-active" : ""}`} onClick={() => setPlaying((value) => !value)}><Icon name={playing ? "pause" : "play"} size={15} /> {playing ? "Pause" : "Play"}</button>
        <button className="button button-primary" onClick={handleNextTile}><Icon name="chevron-right" size={15} /> Next tile</button>
      </PageTitle>
      <section className="execution-layout">
        <div className="panel execution-visual"><div className="visual-topline"><span>GEMM PIPELINE / {size}³</span><StatusPill tone={mode === "naive" ? "amber" : "green"}>{mode} path</StatusPill></div><ThreeScene size={size} speed={speed} kernel={kernel} mode={mode} playing={playing} step={step} /><div className="timeline"><div className="timeline-label"><span>tile {String(activeTile).padStart(2, "0")} / 12</span><span>{Math.round((activeTile / 12) * 100)}%</span></div><div className="timeline-track"><span style={{ width: `${(activeTile / 12) * 100}%` }} /></div><div className="timeline-steps"><span>A load</span><span>B load</span><span>FMA</span><span>C store</span></div></div></div>
        <aside className="execution-aside"><div className="panel control-panel"><PanelHeading label="Playback controls" /><label className="range-label"><span>animation speed</span><b>{speed}%</b><input max="100" min="10" onChange={(event) => setSpeed(Number(event.target.value))} type="range" value={speed} /></label><div className="select-stack"><span>execution path</span><div className="segmented"><button className={mode === "naive" ? "is-active" : ""} onClick={() => setMode("naive")}>naive</button><button className={mode === "engine" ? "is-active" : ""} onClick={() => setMode("engine")}>engine</button></div></div><div className="select-stack"><span>matrix dimension</span><div className="size-pills">{sizes.map((item) => <button className={size === item ? "is-active" : ""} key={item} onClick={() => updateSize(item)}>{item}</button>)}</div></div><button className="reset-button" onClick={handleReset}><Icon name="refresh" size={14} /> Reset animation</button></div><div className="panel explanation-panel"><div className="explanation-icon"><Icon name="spark" size={18} /></div><span className="eyebrow">WHAT IS HAPPENING</span><h3>{mode === "naive" ? "The memory walk is scattered." : "The working set stays hot."}</h3><p>{mode === "naive" ? "A naive traversal jumps across rows. The visualized red pulses represent cache lines fetched without reuse." : `Tile ${activeTile} packs A and B panels into contiguous scratch buffers before the micro-kernel accumulates into C.`}</p><div className="explanation-detail"><span>current operation</span><strong>{mode === "naive" ? "cache miss / reload" : activeTile % 3 === 0 ? "fused multiply-add" : "packed panel load"}</strong></div></div></aside>
      </section>
      <section className="metrics-strip panel"><Metric label="L1 hit ratio" value={formatNumber(snapshot.l1Hit)} unit="%" accent="cyan" /><Metric label="L2 hit ratio" value={formatNumber(snapshot.l2Hit)} unit="%" accent="violet" /><Metric label="memory bandwidth" value={formatNumber(snapshot.bandwidth)} unit=" GB/s" accent="amber" /><Metric label="active accumulator" value={`C${activeTile}`} accent="green" /></section>
    </>
  );

  const renderSimd = () => (
    <>
      <PageTitle eyebrow="MECHANISM 02 / REGISTER TILES" title="SIMD kernels."><div className="kernel-tabs">{kernels.map((item) => <button className={kernel === item.value ? "is-active" : ""} key={item.value} onClick={() => updateKernel(item.value)}>{item.label}</button>)}</div></PageTitle>
      <section className="simd-layout"><div className="panel register-panel"><div className="register-heading"><div><div className="eyebrow">REGISTER FILE / {kernel}</div><h3>{kernel === "scalar" ? "One value per tick." : kernel === "neon-4x4" ? "Four lanes, one instruction." : "Sixteen lanes in parallel."}</h3></div><StatusPill tone={kernel === "scalar" ? "amber" : "green"}>{kernel === "scalar" ? "fallback" : "vectorized"}</StatusPill></div><RegisterMap kernel={kernel} /><div className="register-caption"><span><i className="lane-dot lane-cyan" /> load</span><span><i className="lane-dot lane-amber" /> broadcast</span><span><i className="lane-dot lane-violet" /> FMA result</span></div></div><div className="simd-side"><div className="panel instruction-panel"><PanelHeading label="Instruction trace" action={<span className="mono-tag">cycle 0042</span>} /><div className="instruction-list"><div className="instruction-done"><span>01</span><code>ld1 {`{v0.4s}`}, [A]</code><Icon name="check" size={15} /></div><div className="instruction-active"><span>02</span><code>dup v1.4s, [B]</code><span className="pulse-dot" /></div><div><span>03</span><code>fmla v2.4s, v0.4s, v1.4s</code></div><div><span>04</span><code>st1 {`{v2.4s}`}, [C]</code></div></div></div><div className="panel comparison-panel"><PanelHeading label="Kernel comparison" /><div className="comparison-row"><div><span>scalar</span><strong>1</strong><small>lane / tick</small></div><div className="comparison-line"><span style={{ width: "9%" }} /></div></div><div className="comparison-row"><div><span>neon-4x4</span><strong>4</strong><small>lanes / tick</small></div><div className="comparison-line comparison-cyan"><span style={{ width: "34%" }} /></div></div><div className="comparison-row"><div><span>avx512</span><strong>16</strong><small>lanes / tick</small></div><div className="comparison-line comparison-violet"><span style={{ width: "100%" }} /></div></div></div></div></section>
      <section className="formula-row"><div className="formula-chip"><span className="formula-step">A</span><span>load vector</span><Icon name="arrow-up-right" size={14} /></div><div className="formula-operator">×</div><div className="formula-chip"><span className="formula-step formula-violet">B</span><span>broadcast scalar</span><Icon name="arrow-up-right" size={14} /></div><div className="formula-operator">+</div><div className="formula-chip"><span className="formula-step formula-amber">C</span><span>accumulate</span><Icon name="arrow-up-right" size={14} /></div><div className="formula-operator">=</div><div className="formula-result">one finished register tile</div></section>
    </>
  );

  const renderThreads = () => (
    <>
      <PageTitle eyebrow="MECHANISM 03 / EXECUTION ORCHESTRATION" title="Thread pool."><StatusPill>4 workers active</StatusPill></PageTitle>
      <section className="thread-layout"><div className="panel thread-visual"><div className="thread-visual-header"><div><div className="eyebrow">OUTPUT OWNERSHIP / MATRIX C</div><h3>Every tile has one owner.</h3></div><span className="panel-meta"><Icon name="circle-info" size={14} /> no false sharing</span></div><ThreadGrid selectedThread={selectedThread} /><div className="thread-legend">{["cyan", "violet", "amber", "green"].map((color, index) => <button className={selectedThread === index ? "is-active" : ""} key={color} onClick={() => setSelectedThread(index)}><i className={`thread-legend-${color}`} /> worker {index}<span>{index === selectedThread ? "selected" : ""}</span></button>)}</div></div><aside className="thread-aside"><div className="panel worker-panel"><PanelHeading label="Worker queue" action={<span className="mono-tag">lock-free</span>} />{[0, 1, 2, 3].map((worker) => <button className={`worker-row ${selectedThread === worker ? "is-active" : ""}`} key={worker} onClick={() => setSelectedThread(worker)}><span className={`worker-avatar worker-${worker}`}>T{worker}</span><span className="worker-name"><b>worker_{worker}</b><small>scratch buffer · C{worker + 1}</small></span><span className="worker-state">{selectedThread === worker ? "running" : "ready"}</span></button>)}</div><div className="panel thread-note"><Icon name="target" size={18} /><div><span className="eyebrow">WHY IT MATTERS</span><p>Work is partitioned on output tiles, not shared rows. That keeps writes disjoint and lets each worker reuse its packed panels privately.</p></div></div></aside></section>
      <section className="metrics-strip panel"><Metric label="threads requested" value="4" accent="cyan" /><Metric label="threads used" value="4" accent="violet" /><Metric label="private scratch" value="64 KB" accent="amber" /><Metric label="overlap" value="0" unit=" tiles" accent="green" /></section>
    </>
  );

  return (
    <div className="app-shell">
      <aside className="sidebar"><div className="brand"><span className="brand-mark"><span /><span /><span /></span><span>APEX<span className="brand-muted">/</span>COMPUTE</span></div><div className="sidebar-section-label">Visualizer</div><nav>{navItems.map((item) => <button className={`nav-item ${page === item.id ? "is-active" : ""}`} key={item.id} onClick={() => setPage(item.id)}><span className="nav-icon"><Icon name={item.icon} size={17} /></span><span>{item.label}</span><small>{item.detail}</small></button>)}</nav><div className="sidebar-bottom"><div className="system-card"><div className="system-card-head"><span className="system-live"><i /> local runner</span><Icon name="server" size={15} /></div><strong>macOS · arm64</strong><span>NEON available · 4 MB L2</span></div><button className="sidebar-link" onClick={() => setPage("home")}><Icon name="book" size={15} /> Read the architecture <Icon name="arrow-up-right" size={13} /></button><div className="version">APEX VISUALIZER <span>v0.1.0</span></div></div></aside>
      <main className="main-content"><header className="topbar"><div className="breadcrumbs"><span>workspace</span><Icon name="chevron-right" size={13} /><b>{page === "home" ? "overview" : navItems.find((item) => item.id === page)?.label.toLowerCase()}</b></div><div className="topbar-actions"><span className="topbar-status"><i /> synced to local schema</span><button className="icon-button" title="Refresh visualization" onClick={handleReset}><Icon name="refresh" size={16} /></button><button className="avatar-button">GG</button></div></header><div className="content-inner">{page === "home" && renderHome()}{page === "execution" && renderExecution()}{page === "simd" && renderSimd()}{page === "threads" && renderThreads()}</div></main>
    </div>
  );
};

export default App;
