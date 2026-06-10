import { useMemo, useState, type CSSProperties } from "react";
import type {
  Bootstrap,
  LossSurfaceTrace,
  NeuralBackwardStats,
  NeuralTensorStats,
  NeuralTrace,
} from "../types";

function demoTrace(bootstrap: Bootstrap): NeuralTrace {
  const model = bootstrap.factory.model;
  const forward: NeuralTensorStats[] = [
    {
      name: "Token embedding",
      shape: [16, 128, model.d_model],
      sampled_values: 65_536,
      mean: 0.002,
      std: 0.31,
      rms: 0.31,
      max_abs: 1.42,
    },
    ...Array.from({ length: model.n_layers }, (_, index) => ({
      name: `Transformer block ${index + 1}`,
      shape: [16, 128, model.d_model],
      sampled_values: 65_536,
      mean: 0.004 + index * 0.002,
      std: 0.42 + index * 0.035,
      rms: 0.43 + index * 0.037,
      max_abs: 1.9 + index * 0.22,
    })),
    {
      name: "Final RMSNorm",
      shape: [16, 128, model.d_model],
      sampled_values: 65_536,
      mean: 0.001,
      std: 1,
      rms: 1,
      max_abs: 3.4,
    },
    {
      name: "Vocabulary logits",
      shape: [16, 128, model.vocab_size],
      sampled_values: 65_536,
      mean: -0.08,
      std: 1.24,
      rms: 1.25,
      max_abs: 7.8,
    },
  ];
  const backward: NeuralBackwardStats[] = Array.from(
    { length: model.n_layers },
    (_, index) => ({
      name: `Transformer block ${index + 1}`,
      gradient_norm: 0.18 + (model.n_layers - index) * 0.065,
      mean_abs_gradient: 0.00012 * (model.n_layers - index),
      max_abs_gradient: 0.012 * (model.n_layers - index),
      parameter_tensors: 9,
      elements: 1_400_000,
      sampled_parameters: 2_304,
      update_rms: 0.000018 * (model.n_layers - index),
      update_max_abs: 0.00011 * (model.n_layers - index),
    }),
  );
  const axis = [-1, -0.5, 0, 0.5, 1];
  const losses = axis.map((vertical) =>
    axis.map(
      (horizontal) =>
        2.24 +
        (horizontal + 0.28) ** 2 * 0.34 +
        (vertical - 0.18) ** 2 * 0.22 +
        horizontal * vertical * 0.04,
    ),
  );
  return {
    stage: "pretrain",
    step: 1200,
    loss: 2.2016,
    batch_shape: [16, 128],
    tokens: [
      "<bos>",
      "Once",
      "upon",
      "a",
      "time",
      ",",
      "Mia",
      "found",
      "a",
      "key",
      ".",
    ],
    forward,
    attention: {
      q_shape: [16, 8, 128, 48],
      kv_shape: [16, 2, 128, 48],
      preview: Array.from({ length: 12 }, (_, row) =>
        Array.from({ length: 12 }, (_, column) =>
          column <= row ? Math.exp(column - row) * 0.54 : 0,
        ),
      ),
    },
    backward,
    optimizer: {
      name: "AdamW",
      learning_rate: 0.0003,
      pre_clip_gradient_norm: 1.92,
      clip_limit: 1,
      clip_scale: 0.52,
    },
    surface: {
      axis,
      losses,
      center_loss: losses[2][2],
      minimum_loss: Math.min(...losses.flat()),
      minimum_at: [-0.5, 0],
      direction_scale: 0.015,
      method: "현재 trainable weight 주변의 filter-normalized 2D slice",
    },
  };
}

function scientific(value: number | undefined) {
  if (value === undefined) return "—";
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function shapeLabel(shape: number[]) {
  return `[${shape.map((value) => value.toLocaleString()).join(" × ")}]`;
}

function LossSurface({ surface }: { surface: LossSurfaceTrace }) {
  const values = surface.losses.flat();
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = Math.max(1e-8, maximum - minimum);
  const project = (row: number, column: number) => {
    const normalized = (surface.losses[row][column] - minimum) / range;
    return {
      x: 250 + (column - row) * 52,
      y: 54 + (column + row) * 25 - normalized * 92,
    };
  };
  const rowLines = surface.losses.map((_, row) =>
    surface.axis.map((__, column) => project(row, column)),
  );
  const columnLines = surface.axis.map((_, column) =>
    surface.axis.map((__, row) => project(row, column)),
  );
  const centerIndex = Math.floor(surface.axis.length / 2);
  const center = project(centerIndex, centerIndex);
  const minimumColumn = Math.max(
    0,
    surface.axis.findIndex((value) => value === surface.minimum_at[0]),
  );
  const minimumRow = Math.max(
    0,
    surface.axis.findIndex((value) => value === surface.minimum_at[1]),
  );
  const target = project(minimumRow, minimumColumn);

  return (
    <div className="loss-surface">
      <svg viewBox="0 0 500 270" aria-label="2차원 학습 손실 곡면">
        <defs>
          <linearGradient id="surface-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#d7645d" stopOpacity=".35" />
            <stop offset="1" stopColor="#527d89" stopOpacity=".04" />
          </linearGradient>
          <marker id="descent-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#fff7e8" />
          </marker>
        </defs>
        <path d="M42 222 L250 260 L458 222 L250 184 Z" fill="url(#surface-fill)" />
        {[...rowLines, ...columnLines].map((line, index) => (
          <polyline
            key={index}
            points={line.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke={index < rowLines.length ? "#d88974" : "#6f9ca3"}
            strokeWidth="2"
            opacity=".82"
          />
        ))}
        <line
          x1={center.x}
          y1={center.y}
          x2={target.x}
          y2={target.y}
          stroke="#fff7e8"
          strokeWidth="3"
          strokeDasharray="7 5"
          markerEnd="url(#descent-arrow)"
        />
        <circle cx={center.x} cy={center.y} r="8" fill="#fff7e8" stroke="#a94645" strokeWidth="3" />
        <circle cx={target.x} cy={target.y} r="6" fill="#477b64" stroke="#fff" strokeWidth="2" />
        <text x="18" y="245">direction 2</text>
        <text x="398" y="245">direction 1</text>
        <text x="18" y="24">loss</text>
      </svg>
      <div className="surface-heatmap">
        {surface.losses.flatMap((row, rowIndex) =>
          row.map((loss, columnIndex) => {
            const normalized = (loss - minimum) / range;
            const style = {
              "--surface-color": `hsl(${190 - normalized * 175} 43% ${72 - normalized * 25}%)`,
            } as CSSProperties;
            return (
              <span
                key={`${rowIndex}-${columnIndex}`}
                className={rowIndex === centerIndex && columnIndex === centerIndex ? "current" : ""}
                style={style}
                title={`loss ${loss.toFixed(4)}`}
              />
            );
          }),
        )}
      </div>
    </div>
  );
}

export function NeuralFlowLab({
  bootstrap,
  trace,
}: {
  bootstrap: Bootstrap;
  trace: NeuralTrace | null;
}) {
  const fallback = useMemo(() => demoTrace(bootstrap), [bootstrap]);
  const data = trace?.forward?.length ? trace : fallback;
  const isActual = Boolean(trace?.forward?.length);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selected = data.forward[Math.min(selectedIndex, data.forward.length - 1)];
  const maximumRms = Math.max(...data.forward.map((item) => item.rms), 1e-8);
  const maximumGradient = Math.max(
    ...data.backward.map((item) => item.gradient_norm),
    1e-8,
  );

  return (
    <section className="neural-lab panel">
      <div className="section-heading neural-heading">
        <div>
          <div className="eyebrow">NEURAL COMPUTE GRAPH</div>
          <h2>데이터가 흐르고, 오차가 거꾸로 돌아오는 길</h2>
          <p>실제 GPU tensor 통계와 optimizer 전후 파라미터 표본으로 한 학습 스텝을 펼칩니다.</p>
        </div>
        <span className={isActual ? "real-badge" : "sim-badge"}>
          {isActual ? `${data.stage} step ${data.step} 실제 trace` : "저장 trace 대기 · 개념 데모"}
        </span>
      </div>

      <div className="learning-equations">
        <code>x → Embedding → Attention + SwiGLU → logits → CE loss</code>
        <code>∂L/∂θ ← 역전파</code>
        <code>θₜ₊₁ = AdamW(θₜ, ∇θL)</code>
      </div>

      <div className="token-conveyor">
        <span>batch {shapeLabel(data.batch_shape)}</span>
        <div>
          {data.tokens.map((token, index) => (
            <i key={`${token}-${index}`} style={{ animationDelay: `${index * 70}ms` }}>
              {token.replace("Ġ", "▁")}
            </i>
          ))}
        </div>
      </div>

      <div className="forward-network">
        {data.forward.map((item, index) => (
          <button
            key={item.name}
            className={selectedIndex === index ? "active" : ""}
            onClick={() => setSelectedIndex(index)}
          >
            <small>{String(index + 1).padStart(2, "0")}</small>
            <strong>{item.name.replace("Transformer ", "")}</strong>
            <span>{shapeLabel(item.shape)}</span>
            <i style={{ height: `${24 + (item.rms / maximumRms) * 38}px` }} />
          </button>
        ))}
        <div className="loss-node">
          <small>LOSS</small>
          <strong>{data.loss.toFixed(4)}</strong>
          <span>next-token CE</span>
        </div>
      </div>

      <div className="neural-dashboard">
        <article className="signal-inspector">
          <div className="card-title">
            <div><small>FORWARD SIGNAL</small><h3>{selected.name}</h3></div>
            <span>{shapeLabel(selected.shape)}</span>
          </div>
          <div className="signal-orbit">
            <i style={{ transform: `scale(${0.75 + Math.min(selected.rms, 2) * 0.12})` }} />
            <i />
            <strong>{selected.rms.toFixed(3)}</strong>
            <span>activation RMS</span>
          </div>
          <div className="stat-quads">
            <div><span>mean</span><b>{scientific(selected.mean)}</b></div>
            <div><span>std</span><b>{scientific(selected.std)}</b></div>
            <div><span>max |x|</span><b>{scientific(selected.max_abs)}</b></div>
            <div><span>sample</span><b>{selected.sampled_values.toLocaleString()}</b></div>
          </div>
          <p>RMS가 갑자기 커지거나 0에 가까워지면 exploding·vanishing activation을 의심할 수 있어요.</p>
        </article>

        <article className="backprop-inspector">
          <div className="card-title">
            <div><small>BACKPROPAGATION</small><h3>레이어별 gradient</h3></div>
            <code>∂L/∂θ</code>
          </div>
          <div className="gradient-rails">
            {[...data.backward].reverse().map((item, index) => (
              <div key={item.name}>
                <span>{item.name.replace("Transformer ", "")}</span>
                <i style={{ width: `${Math.max(3, item.gradient_norm / maximumGradient * 100)}%` }} />
                <b>{scientific(item.gradient_norm)}</b>
                <em style={{ animationDelay: `${index * 90}ms` }} />
              </div>
            ))}
          </div>
          <p>밝은 점은 loss에서 입력 방향으로 이동하는 gradient 신호예요. 막대가 너무 작으면 앞단이 배우지 못합니다.</p>
        </article>

        <article className="optimizer-inspector">
          <div className="card-title">
            <div><small>GRADIENT DESCENT</small><h3>{data.optimizer.name} update</h3></div>
            <span>η {data.optimizer.learning_rate.toExponential(1)}</span>
          </div>
          <div className="optimizer-step">
            <div><span>gradient norm</span><strong>{data.optimizer.pre_clip_gradient_norm.toFixed(3)}</strong></div>
            <i>× {data.optimizer.clip_scale.toFixed(3)}</i>
            <div><span>clip 이후</span><strong>{Math.min(data.optimizer.pre_clip_gradient_norm, data.optimizer.clip_limit).toFixed(3)}</strong></div>
          </div>
          <div className="update-list">
            {data.backward.slice(0, 5).map((item) => (
              <div key={item.name}>
                <span>{item.name.replace("Transformer ", "")}</span>
                <b>Δθ RMS {scientific(item.update_rms)}</b>
              </div>
            ))}
          </div>
          <p>gradient를 1.0으로 자른 뒤 Adam의 momentum·분산 보정과 weight decay가 실제 파라미터 이동을 결정합니다.</p>
        </article>

        <article className="surface-inspector">
          <div className="card-title">
            <div><small>LOCAL LOSS LANDSCAPE</small><h3>현재 weight 주변의 학습 곡면</h3></div>
            <span>{data.surface?.method ?? "곡면 trace 대기"}</span>
          </div>
          {data.surface ? (
            <>
              <LossSurface surface={data.surface} />
              <div className="surface-stats">
                <span>현재 loss <b>{data.surface.center_loss.toFixed(4)}</b></span>
                <span>slice 최저점 <b>{data.surface.minimum_loss.toFixed(4)}</b></span>
                <span>방향 scale <b>{(data.surface.direction_scale * 100).toFixed(1)}%</b></span>
              </div>
            </>
          ) : (
            <div className="surface-empty">첫 pretrain 또는 SFT step에서 25번의 실제 forward로 곡면을 측정합니다.</div>
          )}
        </article>

        <article className="attention-inspector">
          <div className="card-title">
            <div><small>ATTENTION ROUTING</small><h3>Q가 과거 K를 읽는 비율</h3></div>
            <span>{data.attention?.q_shape ? shapeLabel(data.attention.q_shape) : "—"}</span>
          </div>
          <div className="attention-mini">
            {(data.attention?.preview ?? []).flatMap((row, rowIndex) =>
              row.map((value, columnIndex) => (
                <i
                  key={`${rowIndex}-${columnIndex}`}
                  style={{ opacity: Math.max(0.04, Math.min(1, value * 2.2)) }}
                  title={value.toFixed(4)}
                />
              )),
            )}
          </div>
          <p>causal mask 때문에 위쪽 삼각형은 닫혀 있고, 각 행의 softmax 합은 1입니다.</p>
        </article>
      </div>
    </section>
  );
}
