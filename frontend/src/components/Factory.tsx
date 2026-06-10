import {
  useEffect,
  useMemo,
  useState,
} from "react";
import { api } from "../api";
import {
  eventDescription,
  formatDuration,
  stageLabel,
  statusLabel,
} from "../activity";
import type {
  Bootstrap,
  ConnectionState,
  DeviceTelemetry,
  FactoryRunConfig,
  FactoryRunSummary,
  NeuralTrace,
} from "../types";
import { LossChart } from "./LossChart";
import { NeuralFlowLab } from "./NeuralFlowLab";

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
  isLive,
  connection,
  telemetry,
  events,
  onRunChange,
}: {
  bootstrap: Bootstrap;
  run: FactoryRunSummary | null;
  isLive: boolean;
  connection: ConnectionState;
  telemetry: DeviceTelemetry | null;
  events: Array<Record<string, unknown>>;
  onRunChange: (run: FactoryRunSummary | null) => void;
}) {
  const [config, setConfig] = useState(() => defaults(bootstrap.factory.stages));
  const [error, setError] = useState("");

  useEffect(() => {
    if (isLive && run) setConfig(run.config);
  }, [isLive, run?.id]);

  const start = async () => {
    setError("");
    try {
      const created = await api.createFactoryRun(config);
      onRunChange(created);
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
      (run?.metric_series ?? []).map((point, index) => ({
          step: index + 1,
          loss: point.loss,
        })),
    [run?.metric_series],
  );
  const stageProgress = Math.min(
    100,
    (run?.stage_progress ??
      (run?.stage_total ? run.stage_step / run.stage_total : 0)) * 100,
  );
  const overallProgress = Math.min(100, (run?.overall_progress ?? 0) * 100);
  const neuralTrace =
    (run?.metrics.neural_trace as unknown as NeuralTrace | undefined) ?? null;
  const currentMetric = (run ? run.metrics[run.stage] : undefined) as
      | Record<string, unknown>
      | undefined;
  const currentLoss =
    typeof currentMetric?.loss === "number" ? currentMetric.loss : null;
  const connectionLabel =
    connection === "connected"
      ? "실시간 연결"
      : connection === "stale"
        ? "상태 확인 필요"
        : isLive
          ? "재연결 중"
          : "기록 보기";
  const memoryUsed = telemetry?.memory_used_bytes;
  const memoryTotal = telemetry?.memory_total_bytes;

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
            const currentStageIndex = bootstrap.factory.stages.indexOf(
              run?.stage ?? "",
            );
            const state =
              run?.stage_states?.[stage] ??
              (run?.stage === stage && isLive
                ? "running"
                : run?.status === "completed" && selected
                  ? "completed"
                  : run && index < currentStageIndex && selected
                    ? "completed"
                : selected
                  ? "pending"
                  : "skipped");
            return (
              <button
                key={stage}
                className={`${selected ? "selected" : ""} stage-${state}`}
                onClick={() => toggleStage(stage)}
                disabled={isLive}
              >
                <small>{String(index + 1).padStart(2, "0")}</small>
                <strong>{meta[0]}</strong>
                <span>{meta[1]}</span>
                <i>
                  {state === "running"
                    ? "RUNNING"
                    : state === "completed"
                      ? "DONE"
                      : state === "skipped"
                        ? "SKIP"
                        : "PENDING"}
                </i>
              </button>
            );
          })}
        </div>
      </section>

      <div className="factory-grid">
        <section className="panel factory-settings">
          <div className="section-heading"><div><div className="eyebrow">RUN CONFIG</div><h2>공장 설정</h2></div></div>
          <label><span>실행 이름</span><input disabled={isLive} value={config.name} onChange={(event) => setConfig({ ...config, name: event.target.value })} /></label>
          <div className="setting-grid">
            {[
              ["pretrain_steps", "Pretrain"],
              ["sft_steps", "SFT"],
              ["reward_steps", "Reward"],
              ["dpo_steps", "DPO"],
              ["ppo_steps", "PPO"],
              ["grpo_steps", "GRPO"],
            ].map(([key, label]) => (
              <label key={key}><span>{label} steps</span><input disabled={isLive} type="number" value={config[key as keyof FactoryRunConfig] as number} onChange={(event) => setConfig({ ...config, [key]: Number(event.target.value) })} /></label>
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
                  disabled={isLive}
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
          {(error || run?.error_message) && (
            <div className="error-banner">{error || run?.error_message}</div>
          )}
          <button
            className="primary-button wide factory-start"
            disabled={
              !config.stages.length ||
              !bootstrap.system.production_ready ||
              isLive
            }
            onClick={start}
          >
            {isLive
              ? `${stageLabel(run?.stage)} ${statusLabel(run?.status)}`
              : bootstrap.system.production_ready
                ? "모델 공장 실행"
                : "GPU 진단 확인 후 실행"}
          </button>
          {run && isLive && (
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
            <div className="live-heading-meta">
              <span className={isLive ? "real-badge" : "history-badge"}>
                {isLive ? "실시간 실행" : run ? "최근 완료 실행" : "대기"}
              </span>
              <span className={`live-status ${run?.status ?? "ready"}`}>
                {statusLabel(run?.status)}
              </span>
            </div>
          </div>
          <div className={`connection-strip ${connection}`}>
            <i />
            <span>{connectionLabel}</span>
            <small>
              마지막 갱신{" "}
              {run?.updated_at
                ? new Date(run.updated_at * 1000).toLocaleTimeString("ko-KR")
                : "—"}
            </small>
          </div>
          <div className="progress-caption">
            <span>전체 공정</span><b>{overallProgress.toFixed(0)}%</b>
          </div>
          <div className="factory-progress overall"><i style={{ width: `${overallProgress}%` }} /></div>
          <div className="progress-caption">
            <span>{stageLabel(run?.stage)} 단계</span><b>{stageProgress.toFixed(0)}%</b>
          </div>
          <div className="factory-progress"><i style={{ width: `${stageProgress}%` }} /></div>
          <div className="factory-metrics operations">
            <div><span>Stage step</span><strong>{run?.stage_step ?? 0} / {run?.stage_total ?? 0}</strong></div>
            <div><span>현재 loss</span><strong>{currentLoss?.toFixed(4) ?? "—"}</strong></div>
            <div><span>처리 속도</span><strong>{run?.steps_per_second ? `${run.steps_per_second.toFixed(2)} step/s` : "계산 중"}</strong></div>
            <div><span>경과 시간</span><strong>{formatDuration(run?.elapsed_seconds)}</strong></div>
            <div><span>남은 시간</span><strong>{formatDuration(run?.eta_seconds)}</strong></div>
            <div><span>Checkpoints</span><strong>{Object.keys(run?.checkpoints ?? {}).length}</strong></div>
          </div>
          <div className="gpu-telemetry-grid">
            <div><span>GPU 사용률</span><strong>{telemetry?.utilization_percent !== undefined ? `${telemetry.utilization_percent.toFixed(0)}%` : "—"}</strong></div>
            <div><span>VRAM</span><strong>{memoryUsed !== undefined && memoryTotal ? `${(memoryUsed / 1024 ** 3).toFixed(2)} / ${(memoryTotal / 1024 ** 3).toFixed(1)} GB` : run ? `${(run.peak_vram_bytes / 1024 ** 3).toFixed(2)} GB peak` : "—"}</strong></div>
            <div><span>온도</span><strong>{telemetry?.temperature_c !== undefined ? `${telemetry.temperature_c.toFixed(0)}°C` : "—"}</strong></div>
            <div><span>전력</span><strong>{telemetry?.power_w !== undefined ? `${telemetry.power_w.toFixed(0)} W` : "—"}</strong></div>
          </div>
          <LossChart points={lossPoints} />
          <div className="event-stream">
            {events.slice(-5).reverse().map((event, index) => (
              <div key={`${event.timestamp}-${index}`}>
                <time>
                  {typeof event.timestamp === "number"
                    ? new Date(event.timestamp * 1000).toLocaleTimeString("ko-KR")
                    : "—"}
                </time>
                <code>{String(event.type).toUpperCase()}</code>
                <span>{eventDescription(event)}</span>
              </div>
            ))}
            {!events.length && <p>{isLive ? "실시간 이벤트를 기다리고 있어요." : "새 실행을 시작하면 단계별 이벤트가 여기에 표시돼요."}</p>}
          </div>
        </section>
      </div>

      <NeuralFlowLab bootstrap={bootstrap} trace={neuralTrace} />

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
