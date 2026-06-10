import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { formatDuration, isActiveRun, statusLabel } from "../activity";
import type {
  Bootstrap,
  ConnectionState,
  Generation,
  RunSummary,
  Trace,
  TrainingConfig,
} from "../types";
import { Heatmap } from "./Heatmap";
import { LossChart } from "./LossChart";
import { MetricCard } from "./MetricCard";
import { VectorStrip } from "./VectorStrip";

function defaultConfig(bootstrap: Bootstrap): TrainingConfig {
  return {
    name: `실험 ${new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}`,
    preset: "balanced",
    steps: 500,
    batch_size: 32,
    learning_rate: 0.0003,
    weight_decay: 0.01,
    grad_clip: 1,
    seed: 42,
    model: {
      vocab_size: bootstrap.model.vocab_size,
      context_length: bootstrap.model.context_length,
      d_model: bootstrap.model.d_model,
      n_layers: bootstrap.model.n_layers,
      n_heads: bootstrap.model.n_heads,
      d_ff: bootstrap.model.d_ff,
      dropout: bootstrap.model.dropout,
    },
  };
}

export function Lab({
  bootstrap,
  run,
  connection,
  events,
  onRunChange,
  onRunFinished,
}: {
  bootstrap: Bootstrap;
  run: RunSummary | null;
  connection: ConnectionState;
  events: Array<Record<string, unknown>>;
  onRunChange: (run: RunSummary | null) => void;
  onRunFinished: (run: RunSummary) => void;
}) {
  const [config, setConfig] = useState(() => defaultConfig(bootstrap));
  const [trace, setTrace] = useState<Trace | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [activePanel, setActivePanel] = useState<"attention" | "vectors" | "logits">("attention");
  const [error, setError] = useState("");
  const [prompt, setPrompt] = useState("Once upon a time");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(10);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [selectedToken, setSelectedToken] = useState(-1);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [generating, setGenerating] = useState(false);
  const finishedRef = useRef<string | null>(null);
  const live = isActiveRun(run);

  useEffect(() => {
    if (live && run) setConfig(run.config);
  }, [live, run?.id]);

  useEffect(() => {
    const event = [...events].reverse().find((item) => item.trace);
    if (event?.trace) setTrace(event.trace as Trace);
  }, [events]);

  useEffect(() => {
    if (
      run &&
      ["completed", "stopped", "failed"].includes(run.status) &&
      finishedRef.current !== run.id
    ) {
      finishedRef.current = run.id;
      onRunFinished(run);
    }
  }, [onRunFinished, run]);

  const selectPreset = (key: string) => {
    const preset = bootstrap.presets[key];
    setConfig((current) => ({
      ...current,
      preset: key as TrainingConfig["preset"],
      steps: preset.steps,
      batch_size: preset.batch_size,
      learning_rate: preset.learning_rate,
    }));
  };

  const startRun = async () => {
    setError("");
    try {
      const created = await api.createRun(config);
      onRunChange(created);
      finishedRef.current = null;
      setTrace(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "학습을 시작하지 못했습니다.");
    }
  };

  const control = async (action: string) => {
    if (!run) return;
    try {
      onRunChange(await api.controlRun(run.id, action));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "제어 요청에 실패했습니다.");
    }
  };

  const generate = async () => {
    setGenerating(true);
    setError("");
    try {
      const result = await api.generate({
        prompt,
        max_new_tokens: 24,
        temperature,
        top_k: topK,
        seed: 42,
        run_id: run?.id,
        layer,
        head,
      });
      setGeneration(result);
      if (result.trace) setTrace(result.trace);
      setSelectedToken(result.trace?.attention?.token_index ?? -1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "문장을 생성하지 못했습니다.");
    } finally {
      setGenerating(false);
    }
  };

  const inspectToken = async (tokenIndex: number) => {
    if (!trace) return;
    setSelectedToken(tokenIndex);
    try {
      const result = await api.inspect({
        token_ids: trace.token_ids,
        run_id: run?.id,
        layer,
        head,
        token_index: tokenIndex,
      });
      setTrace(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "토큰 계산을 불러오지 못했습니다.");
    }
  };

  const metrics = run?.last_metrics ?? {};
  const lossPoints = run?.metric_series ?? run?.losses ?? [];
  const progress = Math.min(
    100,
    (run?.overall_progress ??
      (run ? run.step / Math.max(1, run.total_steps) : 0)) * 100,
  );
  const selectedAttention = trace?.attention;
  const probabilities = useMemo(
    () => trace?.top_logits ?? [],
    [trace],
  );

  return (
    <main className="lab-layout">
      <aside className="control-panel panel">
        <div className="control-heading">
          <div><div className="eyebrow">TRAINING SETUP</div><h2>실험 설계</h2></div>
          <span className={`live-status ${run?.status ?? "ready"}`}>{statusLabel(run?.status)}</span>
        </div>

        <label className="field">
          <span>실험 이름</span>
          <input
            disabled={live}
            value={config.name}
            maxLength={50}
            onChange={(event) => setConfig({ ...config, name: event.target.value })}
          />
        </label>

        <div className="field">
          <span>학습 프리셋</span>
          <div className="preset-list">
            {Object.entries(bootstrap.presets).map(([key, preset]) => (
              <button
                key={key}
                disabled={live}
                className={config.preset === key ? "active" : ""}
                onClick={() => selectPreset(key)}
              >
                <strong>{preset.label}</strong>
                <small>{preset.steps} steps · batch {preset.batch_size}</small>
              </button>
            ))}
          </div>
        </div>

        <div className="two-fields">
          <label className="field">
            <span>Steps</span>
            <input
              disabled={live}
              type="number"
              min={1}
              max={3000}
              value={config.steps}
              onChange={(event) => setConfig({ ...config, steps: Number(event.target.value) })}
            />
          </label>
          <label className="field">
            <span>Batch</span>
            <input
              disabled={live}
              type="number"
              min={1}
              max={64}
              value={config.batch_size}
              onChange={(event) =>
                setConfig({ ...config, batch_size: Number(event.target.value) })
              }
            />
          </label>
        </div>
        <label className="field">
          <span>Learning rate <b>{config.learning_rate}</b></span>
          <input
            disabled={live}
            type="range"
            min={0.00005}
            max={0.001}
            step={0.00005}
            value={config.learning_rate}
            onChange={(event) =>
              setConfig({ ...config, learning_rate: Number(event.target.value) })
            }
          />
        </label>

        <button disabled={live} className="advanced-toggle" onClick={() => setAdvanced(!advanced)}>
          <span>심화 설정</span><b>{advanced ? "−" : "+"}</b>
        </button>
        {advanced && (
          <div className="advanced-fields">
            <div className="two-fields">
              <label className="field"><span>레이어</span>
                <select
                  disabled={live}
                  value={config.model.n_layers}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      model: { ...config.model, n_layers: Number(event.target.value) },
                    })
                  }
                >
                  {[1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
              <label className="field"><span>헤드</span>
                <select
                  disabled={live}
                  value={config.model.n_heads}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      model: { ...config.model, n_heads: Number(event.target.value) },
                    })
                  }
                >
                  {[1, 2, 4, 8].map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
            </div>
            <div className="two-fields">
              <label className="field"><span>d_model</span>
                <select
                  disabled={live}
                  value={config.model.d_model}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      model: { ...config.model, d_model: Number(event.target.value) },
                    })
                  }
                >
                  {[64, 128, 256].map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
              <label className="field"><span>d_ff</span>
                <select
                  disabled={live}
                  value={config.model.d_ff}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      model: { ...config.model, d_ff: Number(event.target.value) },
                    })
                  }
                >
                  {[256, 512, 1024].map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
            </div>
          </div>
        )}

        <div className="run-controls">
          {!live ? (
            <button className="primary-button wide" onClick={startRun}>학습 시작</button>
          ) : (
            <>
              <button className="primary-button" onClick={() => control(run?.status === "paused" ? "resume" : "pause")}>
                {run?.status === "paused" ? "계속" : "일시정지"}
              </button>
              <button className="ghost-button" onClick={() => control("step")}>1 step</button>
              <button className="danger-button" onClick={() => control("stop")}>정지</button>
            </>
          )}
        </div>
        <div className="progress-block">
          <div><span>{run?.step ?? 0} / {run?.total_steps ?? config.steps} steps</span><b>{progress.toFixed(0)}%</b></div>
          <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
        </div>
        <div className={`connection-strip ${connection}`}>
          <i />
          <span>
            {live
              ? connection === "connected"
                ? "실시간 연결"
                : connection === "stale"
                  ? "상태 확인 필요"
                  : "재연결 중"
              : run
                ? "최근 실행 결과"
                : "실행 대기"}
          </span>
          <small>
            {run?.updated_at
              ? new Date(run.updated_at * 1000).toLocaleTimeString("ko-KR")
              : "—"}
          </small>
        </div>
        <div className="model-spec">
          <span>현재 모델</span>
          <strong>{(run?.parameter_count ?? bootstrap.model.parameter_count).toLocaleString()} params</strong>
          <small>{config.model.n_layers} layers · {config.model.n_heads} heads · {config.model.d_model} dim</small>
        </div>
      </aside>

      <section className="lab-main">
        {(error || run?.error_message) && (
          <div className="error-banner">{error || run?.error_message}</div>
        )}
        <section className="panel microscope-operations">
          <div>
            <span>{live ? "실시간 실행" : run ? "최근 실행" : "대기"}</span>
            <strong>{statusLabel(run?.status)}</strong>
          </div>
          <div><span>진행률</span><strong>{progress.toFixed(0)}%</strong></div>
          <div><span>경과 시간</span><strong>{formatDuration(run?.elapsed_seconds)}</strong></div>
          <div><span>남은 시간</span><strong>{formatDuration(run?.eta_seconds)}</strong></div>
          <div><span>처리 속도</span><strong>{run?.steps_per_second ? `${run.steps_per_second.toFixed(2)} step/s` : "계산 중"}</strong></div>
        </section>
        <div className="metrics-grid">
          <MetricCard label="Cross Entropy" value={metrics.loss?.toFixed(4) ?? "—"} detail="낮을수록 정답에 가까워요" tone="coral" />
          <MetricCard label="Perplexity" value={metrics.perplexity?.toFixed(2) ?? "—"} detail="모델이 고민하는 후보 수" tone="gold" />
          <MetricCard label="Gradient norm" value={metrics.gradient_norm?.toFixed(3) ?? "—"} detail="업데이트 방향의 크기" tone="blue" />
          <MetricCard label="처리 속도" value={metrics.steps_per_second ? `${metrics.steps_per_second.toFixed(1)}/s` : "—"} detail="CPU 학습 step / 초" />
        </div>

        <section className="panel chart-panel">
          <div className="section-heading">
            <div><div className="eyebrow">LEARNING CURVE</div><h2>모델이 틀리는 정도</h2></div>
            <div className="delta-pill">Δθ {metrics.parameter_delta_norm?.toExponential(2) ?? "—"}</div>
          </div>
          <LossChart points={lossPoints} />
        </section>

        <section className="panel inspector">
          <div className="section-heading inspector-heading">
            <div><div className="eyebrow">MODEL MICROSCOPE</div><h2>한 토큰의 계산을 해부하기</h2></div>
            <div className="inspector-tabs">
              <button className={activePanel === "attention" ? "active" : ""} onClick={() => setActivePanel("attention")}>Attention</button>
              <button className={activePanel === "vectors" ? "active" : ""} onClick={() => setActivePanel("vectors")}>Q · K · V</button>
              <button className={activePanel === "logits" ? "active" : ""} onClick={() => setActivePanel("logits")}>Logits</button>
            </div>
          </div>
          {!trace ? (
            <div className="empty-state tall">
              <strong>아직 관찰할 계산이 없어요.</strong>
              <span>학습을 한 step 실행하거나 아래에서 문장을 생성해 보세요.</span>
            </div>
          ) : (
            <>
              <div className="token-ribbon">
                {trace.tokens.map((token, index) => (
                  <button
                    key={`${token}-${index}`}
                    className={index === selectedToken || index === selectedAttention?.token_index ? "selected" : ""}
                    onClick={() => inspectToken(index)}
                    title={`${index}번 토큰의 계산 보기`}
                  >
                    <small>{index}</small>{token}
                  </button>
                ))}
              </div>
              {activePanel === "attention" && selectedAttention && (
                <div className="attention-view">
                  <div>
                    <div className="matrix-title">
                      <strong>Layer {selectedAttention.layer + 1} · Head {selectedAttention.head + 1}</strong>
                      <code>[{selectedAttention.attention.length} × {selectedAttention.attention.length}]</code>
                    </div>
                    <Heatmap matrix={selectedAttention.attention} tokens={trace.tokens} />
                  </div>
                  <div className="calculation-card">
                    <div className="calculation-step"><span>1</span><div><strong>내적</strong><code>QKᵀ</code></div></div>
                    <div className="number-strip">
                      {selectedAttention.raw_scores.slice(0, 12).map((value, index) => <i key={index}>{value.toFixed(2)}</i>)}
                    </div>
                    <div className="calculation-step"><span>2</span><div><strong>크기 보정 + mask</strong><code>/ √{selectedAttention.q.length}</code></div></div>
                    <div className="number-strip">
                      {selectedAttention.scaled_scores.slice(0, 12).map((value, index) => <i key={index}>{value === null ? "−∞" : value.toFixed(2)}</i>)}
                    </div>
                    <div className="calculation-step"><span>3</span><div><strong>Softmax</strong><code>Σ weight = {selectedAttention.weights.reduce((sum, value) => sum + value, 0).toFixed(4)}</code></div></div>
                    <div className="weight-bars">
                      {selectedAttention.weights.slice(0, 12).map((value, index) => <i key={index} style={{ height: `${Math.max(3, value * 100)}%` }} title={value.toFixed(4)} />)}
                    </div>
                  </div>
                </div>
              )}
              {activePanel === "vectors" && selectedAttention && (
                <div className="vectors-view">
                  <div className="formula-inline"><code>q = xW<sub>Q</sub></code><code>k = xW<sub>K</sub></code><code>v = xW<sub>V</sub></code></div>
                  <VectorStrip label="Query" values={selectedAttention.q} color="coral" />
                  <VectorStrip label="Key[선택]" values={selectedAttention.keys[selectedAttention.token_index]} color="blue" />
                  <VectorStrip label="Value[선택]" values={selectedAttention.values[selectedAttention.token_index]} color="gold" />
                  <VectorStrip label="Σ αV" values={selectedAttention.weighted_value} color="coral" />
                  <div className="layer-stats">
                    {trace.layer_stats.map((stat) => (
                      <div key={stat.layer}><span>Layer {stat.layer + 1}</span><strong>norm {stat.norm.toFixed(3)}</strong><small>μ {stat.mean.toFixed(3)} · σ {stat.std.toFixed(3)}</small></div>
                    ))}
                  </div>
                </div>
              )}
              {activePanel === "logits" && (
                <div className="logits-view">
                  <div className="logits-formula"><code>softmax(logits) → p(next token)</code><span>마지막 위치의 상위 후보</span></div>
                  {probabilities.length ? probabilities.map((item) => (
                    <div className="probability-row" key={item.token_id}>
                      <strong>{item.token}</strong>
                      <div><i style={{ width: `${Math.max(1, item.probability * 100)}%` }} /></div>
                      <span>{(item.probability * 100).toFixed(2)}%</span>
                      <code>{item.logit.toFixed(3)}</code>
                    </div>
                  )) : <p>학습 trace에서 다음 토큰 후보를 불러오는 중이에요.</p>}
                </div>
              )}
            </>
          )}
        </section>

        <section className="panel generator-panel">
          <div className="section-heading">
            <div><div className="eyebrow">AUTOREGRESSIVE GENERATION</div><h2>모델에게 이야기의 첫 문장을 건네기</h2></div>
          </div>
          <div className="generation-controls">
            <label className="prompt-field"><span>Prompt</span><input value={prompt} onChange={(event) => setPrompt(event.target.value)} /></label>
            <label><span>Temperature <b>{temperature}</b></span><input type="range" min={0.1} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
            <label><span>Top-k <b>{topK}</b></span><input type="range" min={1} max={50} value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
            <label><span>Layer</span><select value={layer} onChange={(event) => setLayer(Number(event.target.value))}>{Array.from({ length: config.model.n_layers }, (_, value) => <option value={value} key={value}>{value + 1}</option>)}</select></label>
            <label><span>Head</span><select value={head} onChange={(event) => setHead(Number(event.target.value))}>{Array.from({ length: config.model.n_heads }, (_, value) => <option value={value} key={value}>{value + 1}</option>)}</select></label>
            <button className="primary-button" disabled={generating} onClick={generate}>{generating ? "생성 중…" : "생성하기"}</button>
          </div>
          {generation ? (
            <div className="generation-result">
              <blockquote>{generation.text}</blockquote>
              <div className="generated-tokens">
                {generation.steps.map((step, index) => (
                  <span key={`${step.token}-${index}`} style={{ animationDelay: `${index * 35}ms` }} title={step.candidates.slice(0, 3).map((item) => `${item.token}: ${(item.probability * 100).toFixed(1)}%`).join("\n")}>
                    {step.token}
                  </span>
                ))}
              </div>
              <small>각 토큰 위에 마우스를 올리면 당시의 상위 후보를 볼 수 있어요.</small>
            </div>
          ) : (
            <div className="sample-preview">{run?.sample || "준비된 체크포인트 또는 현재 학습 모델로 바로 생성할 수 있어요."}</div>
          )}
        </section>
      </section>
    </main>
  );
}
