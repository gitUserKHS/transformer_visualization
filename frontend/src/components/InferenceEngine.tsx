import { useMemo, useState } from "react";
import { api } from "../api";
import type { Bootstrap, KVBenchmark } from "../types";

export function InferenceEngine({ bootstrap }: { bootstrap: Bootstrap }) {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [result, setResult] = useState<KVBenchmark | null>(null);
  const [error, setError] = useState("");
  const [scale, setScale] = useState<Record<string, unknown> | null>(null);
  const [billions, setBillions] = useState(7);
  const model = bootstrap.factory.model;

  const cacheModes = useMemo(() => {
    const headDim = model.d_model / model.n_heads;
    return [
      ["MHA", model.n_heads],
      ["GQA", model.n_kv_heads],
      ["MQA", 1],
    ].map(([mode, heads]) => ({
      mode,
      heads: Number(heads),
      bytes: 2 * model.n_layers * Number(heads) * headDim * 128 * 2,
    }));
  }, [model]);

  const benchmark = async () => {
    setError("");
    try {
      setResult(await api.kvBenchmark(prompt, 24));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "benchmark에 실패했습니다.");
    }
  };
  const estimate = async () => {
    setScale(await api.scaleEstimate({
      parameters_billions: billions,
      tokens_billions: billions * 20,
      context_length: 4096,
      precision_bytes: 2,
      gpu_memory_gb: 80,
      gpu_tflops: 312,
      utilization: 0.4,
    }));
  };

  return (
    <main className="inference-page">
      <header className="page-heading">
        <div><div className="eyebrow">INFERENCE SYSTEMS</div><h1>한 토큰을 빠르게 내보내는 법</h1><p>prefill과 decode를 나누고, 과거 K/V를 재사용해 반복 계산을 줄입니다.</p></div>
        <span className="real-badge">실측 + 시뮬레이션</span>
      </header>
      <section className="kv-explainer panel">
        <div className="cache-equation">
          <code>Kcache ← concat(Kpast, kₜ)</code>
          <code>Vcache ← concat(Vpast, vₜ)</code>
          <span>[batch, kv_heads, sequence, head_dim]</span>
        </div>
        <div className="cache-layers">
          {Array.from({ length: model.n_layers }, (_, layer) => (
            <div key={layer}><span>L{layer + 1}</span>{Array.from({ length: 12 }, (_, token) => <i key={token} style={{ opacity: 0.25 + token / 16 }} />)}</div>
          ))}
        </div>
      </section>
      <div className="inference-grid">
        <section className="panel benchmark-card">
          <div className="section-heading"><div><div className="eyebrow">REAL GPU BENCHMARK</div><h2>Cache ON / OFF</h2></div></div>
          <label><span>Prompt</span><input value={prompt} onChange={(event) => setPrompt(event.target.value)} /></label>
          <button className="primary-button wide" onClick={benchmark}>동일 greedy 생성 비교</button>
          {error && <div className="error-banner">{error}<code>{bootstrap.system.setup_command}</code></div>}
          {result && (
            <div className="benchmark-result">
              <div><span>Cache ON</span><strong>{result.cached.tokens_per_second.toFixed(1)} tok/s</strong><small>{result.cached.elapsed_ms.toFixed(1)} ms</small></div>
              <div><span>Cache OFF</span><strong>{result.uncached.tokens_per_second.toFixed(1)} tok/s</strong><small>{result.uncached.elapsed_ms.toFixed(1)} ms</small></div>
              <footer><span>출력 일치 {result.tokens_match ? "✓" : "×"}</span><strong>{result.speedup.toFixed(2)}× speedup</strong></footer>
              <p>{result.note}</p>
            </div>
          )}
        </section>
        <section className="panel cache-compare">
          <div className="section-heading"><div><div className="eyebrow">CACHE FOOTPRINT</div><h2>MHA · GQA · MQA</h2></div></div>
          {cacheModes.map((item) => (
            <div key={String(item.mode)}>
              <strong>{item.mode}</strong><span>{item.heads} KV heads</span>
              <i style={{ width: `${item.bytes / cacheModes[0].bytes * 100}%` }} />
              <code>{(item.bytes / 1024).toFixed(1)} KB</code>
            </div>
          ))}
          <p>ProductionMiniLM은 GQA로 MHA 대비 KV cache를 75% 줄입니다.</p>
        </section>
      </div>
      <section className="panel batching-card">
        <div className="section-heading"><div><div className="eyebrow">CONTINUOUS BATCHING</div><h2>끝난 요청은 내보내고 새 요청은 즉시 합류</h2></div><span className="sim-badge">scheduler simulation</span></div>
        <div className="batch-timeline">
          {(result?.continuous_batching ?? Array.from({ length: 9 }, (_, tick) => ({ tick, prefill: tick < 4 ? [`req-${tick + 1}`] : [], decode: [`req-1`, `req-${Math.min(4, tick + 1)}`], batch_size: 2 }))).slice(0, 9).map((item) => (
            <div key={item.tick}><small>T{item.tick}</small><span className={item.prefill.length ? "prefill" : ""}>{item.prefill.length ? `P ${item.prefill.join(",")}` : "·"}</span><span>D {item.decode.join(",")}</span></div>
          ))}
        </div>
      </section>
      <section className="scale-card panel">
        <div><div className="eyebrow">COMMERCIAL SCALE ESTIMATOR</div><h2>미니 모델을 7B·70B로 키우면?</h2><p>이 영역은 실제 학습이 아닌 공개식 기반 규모 추정입니다.</p></div>
        <label><span>Parameters</span><input type="range" min={1} max={70} value={billions} onChange={(event) => setBillions(Number(event.target.value))} /><b>{billions}B</b></label>
        <button className="ghost-button" onClick={estimate}>80GB GPU 기준 계산</button>
        {scale && <div className="scale-results"><div><span>Training state</span><strong>{Number(scale.training_state_gb).toFixed(0)} GB</strong></div><div><span>최소 training GPU</span><strong>{String(scale.minimum_training_gpus)}</strong></div><div><span>예상 GPU-days</span><strong>{Number(scale.estimated_gpu_days).toFixed(1)}</strong></div></div>}
      </section>
    </main>
  );
}
