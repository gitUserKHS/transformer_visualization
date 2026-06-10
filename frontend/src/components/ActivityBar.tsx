import { activityLabel, formatDuration } from "../activity";
import type {
  ConnectionState,
  FactoryRunSummary,
  RunSummary,
  Tab,
} from "../types";

export function ActivityBar({
  items,
  onNavigate,
}: {
  items: Array<{
    track: "factory" | "microscope";
    run: FactoryRunSummary | RunSummary;
    connection: ConnectionState;
  }>;
  onNavigate: (tab: Tab) => void;
}) {
  if (!items.length) return null;
  return (
    <aside className="activity-bar" aria-label="현재 학습 관제">
      <strong>LIVE</strong>
      {items.map(({ track, run, connection }) => {
        const progress = Math.round((run.overall_progress ?? 0) * 100);
        return (
          <button
            key={run.id}
            onClick={() =>
              onNavigate(track === "factory" ? "factory" : "microscope")
            }
          >
            <i className={run.status} />
            <span>
              <b>{track === "factory" ? "모델 공장" : "학습 현미경"}</b>
              <small>{activityLabel(run)}</small>
            </span>
            <em><u style={{ width: `${progress}%` }} /></em>
            <code>{progress}%</code>
            <small>ETA {formatDuration(run.eta_seconds)}</small>
            {connection !== "connected" && (
              <mark>
                {connection === "stale" ? "상태 확인 필요" : "재연결 중"}
              </mark>
            )}
          </button>
        );
      })}
    </aside>
  );
}
