import type {
  FactoryRunSummary,
  RunSummary,
} from "./types";

export const activeStatuses = new Set(["running", "paused", "stopping"]);

export function isActiveRun(
  run: FactoryRunSummary | RunSummary | null,
): boolean {
  return Boolean(run && activeStatuses.has(run.status));
}

export function statusLabel(status?: string): string {
  return {
    running: "학습 중",
    paused: "일시정지",
    stopping: "정지 중",
    stopped: "정지됨",
    completed: "완료",
    failed: "실패",
  }[status ?? ""] ?? "준비";
}

export function stageLabel(stage?: string): string {
  return {
    queued: "준비",
    training: "학습",
    data: "데이터 정제",
    pretrain: "사전학습",
    sft: "SFT",
    reward: "보상모델",
    dpo: "DPO",
    ppo: "PPO",
    grpo: "GRPO",
    evaluate: "평가",
    quantize: "양자화",
  }[stage ?? ""] ?? stage ?? "준비";
}

export function activityLabel(
  run: FactoryRunSummary | RunSummary,
): string {
  const stage = stageLabel(run.stage);
  if (run.status === "running") return `${stage} 중`;
  return `${stage} ${statusLabel(run.status)}`;
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "계산 중";
  }
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    return `${hours}시간 ${minutes % 60}분`;
  }
  return minutes ? `${minutes}분 ${remainder}초` : `${remainder}초`;
}

export function eventDescription(event: Record<string, unknown>): string {
  if (event.type === "stage") {
    return `${stageLabel(String(event.stage))} ${
      event.stage_status === "completed" ? "완료" : "시작"
    }`;
  }
  if (event.type === "status") {
    return statusLabel(String(event.status));
  }
  if (event.type === "heartbeat") return "상태 동기화";
  if (event.type === "error") return String(event.message ?? "오류 발생");
  if (typeof event.loss === "number") {
    return `${stageLabel(String(event.type))} loss ${event.loss.toFixed(4)}`;
  }
  return stageLabel(String(event.type ?? "event"));
}
