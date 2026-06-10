import {
  type Dispatch,
  type SetStateAction,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api, openFactorySocket } from "../api";
import type {
  Bootstrap,
  FactoryRunConfig,
  FactoryRunSummary,
} from "../types";
import { LossChart } from "./LossChart";

const stageMeta: Record<string, [string, string]> = {
  data: ["데이터 정제", "중복·품질·언어·반복·PII"],
  pretrain: ["사전학습", "다음 토큰 예측"],
  sft: ["SFT", "지시문과 모범 응답"],
  reward: ["보상모델", "사람의 선호를 점수로"],
  dpo: ["DPO", "선호쌍을 직접 최적화"],
  ppo: ["PPO", "reward와 KL로 정책 업데이트"],
  grpo: ["GRPO", "그룹 상대 advantage"],
  evaluate: ["평가", "정렬 방식 나란히 비교"],
  quantize: ["양자화", "BF16·INT8·INT4"],
};

function defaults(stages: string[]): FactoryRunConfig {
  return {
    name: "ProductionMiniLM 전체 공정",
    stages,
    pretrain_steps: 1200,
    sft_steps: 200,
    reward_steps: 150,
    dpo_steps: 100,
    ppo_steps: 30,
    grpo_steps: 30,
    micro_batch_size: 16,
    gradient_accumulation: 2,
    learning_rate: 0.0003,
    lora_rank: 8,
    response_tokens: 32,
    group_size: 4,
    seed: 42,
  };
}

export function Factory({
  bootstrap,
  run,
  onRunChange,
}: {
  bootstrap: Bootstrap;
  run: FactoryRunSummary | null;
  onRunChange: Dispatch<SetStateAction<FactoryRunSummary | null>>;
}) {
  const [config, setConfig] = useState(() => defaults(bootstrap.factory.stages));
  const [error, setError] = useState("");
  const socket = useRef<WebSocket | null>(null);
  const [events, setEvents] = useState<Array<Record<string, unknown>>>([]);

  useEffect(() => () => socket.current?.close(), []);

  const start = async () => {
    setError("");
    setEvents([]);
    try {
      const created = await api.createFactoryRun(config);
      onRunChange(created);
      socket.current?.close();
      socket.current = openFactorySocket(created.id, (event) => {
        setEvents((current) => [...current.slice(-199), event]);
        if (event.summary) {
          onRunChange(event.summary as FactoryRunSummary);
          return;
        }
        onRunChange((current) => {
          const base = current ?? created;
          return {
            ...base,
            status: (event.status as string) ?? base.status,
            stage: (event.stage as string) ?? base.stage,
            stage_step: Number(event.step ?? base.stage_step),
            metrics: {
              ...base.metrics,
              [event.type as string]: event,
            },
          };
        });
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "모델 공장을 시작하지 못했습니다.");
    }
  };

  const control = async (action: "pause" | "resume" | "stop" | "step") => {
    if (!run) return;
    setError("");
    try {
      onRunChange(await api.controlFactoryRun(run.id, action));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "공장 제어에 실패했습니다.");
    }
  };

  const toggleStage = (stage: string) => {
    setConfig((current) => ({
      ...current,
      stages: current.stages.includes(stage)
        ? current.stages.filter((item) => item !== stage)
        : bootstrap.factory.stages.filter(
            (item) => current.stages.includes(item) || item === stage,
          ),
    }));
  };

  const lossPoints = useMemo(
    () =>
      events
        .filter((event) => typeof event.loss === "number")
        .map((event, index) => ({
          step: index + 1,
          loss: event.loss as number,
        })),
    [events],
  );
  const progress =
    run && run.stage_total ? Math.min(100, (run.stage_step / run.stage_total) * 100) : 0;

  return (
    <main className="factory-page">
      <header className="factory-hero">
        <div>
          <div className="eyebrow">PUBLIC-RESEARCH MODEL PIPELINE</div>
          <h1>작은 모델로 끝까지,<br />상용 LLM 제작 공장</h1>
          <p>
            1,554만 파라미터 LLaMA형 모델이 데이터에서 시작해 정렬·평가·서빙까지
            이동하는 전 과정을 실제 GPU 연산으로 관찰합니다.
          </p>
          <div className="hero-tags">
            <span>RMSNorm</span><span>RoPE</span><span>SwiGLU</span>
            <span>GQA 8Q / 2KV</span><span>LoRA r=8</span>
          </div>
        </div>
        <div className={`gpu-console ${bootstrap.system.production_ready ? "ready" : "blocked"}`}>
          <div className="console-head">
            <span className="status-dot" />
            <strong>{bootstrap.system.production_ready ? "GPU READY" : "GPU SETUP REQUIRED"}</strong>
          </div>
          <h3>{bootstrap.system.gpu?.name ?? "NVIDIA GPU를 찾지 못했어요"}</h3>
          <div className="gpu-specs">
            <span>PyTorch <b>{bootstrap.system.torch}</b></span>
            <span>CUDA <b>{bootstrap.system.torch_cuda ?? "없음"}</b></span>
            <span>Precision <b>{bootstrap.system.precision.toUpperCase()}</b></span>
          </div>
          <p>{bootstrap.system.message}</p>
          {!bootstrap.system.production_ready && <code>{bootstrap.system.setup_command}</code>}
        </div>
      </header>

      <section className="pipeline-board panel">
        <div className="section-heading">
          <div><div className="eyebrow">FACTORY LINE</div><h2>체크포인트가 이동하는 길</h2></div>
          <span className="real-badge">실제 연산</span>
        </div>
        <div className="factory-stages">
          {bootstrap.factory.stages.map((stage, index) => {
            const meta = stageMeta[stage];
            const selected = config.stages.includes(stage);
            const active = run?.stage === stage;
            return (
              <button
                key={stage}
                className={`${selected ? "selected" : ""} ${active ? "running" : ""}`}
                onClick={() => toggleStage(stage)}
              >
                <small>{String(index + 1).padStart(2, "0")}</small>
                <strong>{meta[0]}</strong>
                <span>{meta[1]}</span>
                <i>{active ? "RUNNING" : selected ? "INCLUDED" : "SKIP"}</i>
              </button>
            );
          })}
        </div>
      </section>

      <div className="factory-grid">
        <section className="panel factory-settings">
          <div className="section-heading"><div><div className="eyebrow">RUN CONFIG</div><h2>공장 설정</h2></div></div>
          <label><span>실행 이름</span><input value={config.name} onChange={(event) => setConfig({ ...config, name: event.target.value })} /></label>
          <div className="setting-grid">
            {[
              ["pretrain_steps", "Pretrain"],
              ["sft_steps", "SFT"],
              ["reward_steps", "Reward"],
              ["dpo_steps", "DPO"],
              ["ppo_steps", "PPO"],
              ["grpo_steps", "GRPO"],
            ].map(([key, label]) => (
              <label key={key}><span>{label} steps</span><input type="number" value={config[key as keyof FactoryRunConfig] as number} onChange={(event) => setConfig({ ...config, [key]: Number(event.target.value) })} /></label>
            ))}
          </div>
          <div className="setting-grid advanced-settings">
            {[
              ["micro_batch_size", "Microbatch", 1, 16, 1],
              ["gradient_accumulation", "Grad accumulation", 1, 32, 1],
              ["learning_rate", "Learning rate", 0.000001, 0.003, 0.00001],
              ["lora_rank", "LoRA rank", 2, 32, 2],
              ["response_tokens", "Response tokens", 4, 64, 1],
              ["group_size", "GRPO group", 2, 8, 1],
            ].map(([key, label, min, max, step]) => (
              <label key={String(key)}>
                <span>{label}</span>
                <input
                  type="number"
                  min={Number(min)}
                  max={Number(max)}
                  step={Number(step)}
                  value={config[key as keyof FactoryRunConfig] as number}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      [key as string]: Number(event.target.value),
                    })
                  }
                />
              </label>
            ))}
          </div>
          <div className="factory-summary">
            <div><span>모델</span><strong>{(bootstrap.factory.model.parameter_count / 1e6).toFixed(2)}M</strong></div>
            <div><span>학습 파라미터</span><strong>{bootstrap.factory.model.lora.trainable_percent.toFixed(2)}%</strong></div>
            <div><span>Effective batch</span><strong>{config.micro_batch_size * config.gradient_accumulation}</strong></div>
          </div>
          {error && <div className="error-banner">{error}</div>}
          <button className="primary-button wide factory-start" disabled={!config.stages.length} onClick={start}>
            {bootstrap.system.production_ready ? "모델 공장 실행" : "GPU 진단 확인 후 실행"}
          </button>
          {run && !["completed", "failed", "stopped"].includes(run.status) && (
            <div className="factory-controls">
              <button onClick={() => control(run.status === "paused" ? "resume" : "pause")}>
                {run.status === "paused" ? "재개" : "일시정지"}
              </button>
              <button onClick={() => control("step")}>한 스텝</button>
              <button className="danger" onClick={() => control("stop")}>중지</button>
            </div>
          )}
        </section>

        <section className="panel factory-live">
          <div className="section-heading">
            <div><div className="eyebrow">LIVE TELEMETRY</div><h2>{run ? stageMeta[run.stage]?.[0] ?? run.stage : "공장 대기 중"}</h2></div>
            <span className={`live-status ${run?.status ?? "ready"}`}>{run?.status ?? "ready"}</span>
          </div>
          <div className="factory-progress"><i style={{ width: `${progress}%` }} /></div>
          <div className="factory-metrics">
            <div><span>Stage step</span><strong>{run?.stage_step ?? 0} / {run?.stage_total ?? 0}</strong></div>
            <div><span>Peak VRAM</span><strong>{run ? `${(run.peak_vram_bytes / 1024 ** 3).toFixed(2)} GB` : "—"}</strong></div>
            <div><span>Checkpoints</span><strong>{Object.keys(run?.checkpoints ?? {}).length}</strong></div>
          </div>
          <LossChart points={lossPoints} />
          <div className="event-stream">
            {events.slice(-5).reverse().map((event, index) => (
              <div key={`${event.timestamp}-${index}`}>
                <code>{String(event.type).toUpperCase()}</code>
                <span>{typeof event.loss === "number" ? `loss ${(event.loss as number).toFixed(4)}` : String(event.status ?? event.stage ?? "event")}</span>
              </div>
            ))}
            {!events.length && <p>실행하면 단계별 tensor와 학습 지표가 여기에 도착해요.</p>}
          </div>
        </section>
      </div>

      <section className="data-line panel">
        <div className="section-heading"><div><div className="eyebrow">DATA REFINERY</div><h2>원문이 학습 데이터가 되기까지</h2></div></div>
        <div className="data-funnel">
          {bootstrap.factory.data.cleaning_trace.map((item) => (
            <div key={item.stage}>
              <strong>{item.count.toLocaleString()}</strong>
              <span>{item.stage}</span>
              <i style={{ width: `${item.retention * 100}%` }} />
              <small>{(item.retention * 100).toFixed(1)}% retained</small>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
