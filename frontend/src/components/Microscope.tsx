import { useEffect, useState } from "react";
import { isActiveRun } from "../activity";
import type {
  Bootstrap,
  ConnectionState,
  RunSummary,
} from "../types";
import { Guide } from "./Guide";
import { Lab } from "./Lab";

export function Microscope({
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
  const [mode, setMode] = useState<"guide" | "lab">("guide");
  useEffect(() => {
    if (isActiveRun(run)) setMode("lab");
  }, [run?.id, run?.status]);
  return (
    <div>
      <div className="subnav">
        <button className={mode === "guide" ? "active" : ""} onClick={() => setMode("guide")}>
          원리 가이드
        </button>
        <button className={mode === "lab" ? "active" : ""} onClick={() => setMode("lab")}>
          93만 파라미터 현미경
        </button>
        <span>CPU에서도 실행 가능</span>
      </div>
      {mode === "guide" ? (
        <Guide bootstrap={bootstrap} onOpenLab={() => setMode("lab")} />
      ) : (
        <Lab
          bootstrap={bootstrap}
          run={run}
          connection={connection}
          events={events}
          onRunChange={onRunChange}
          onRunFinished={onRunFinished}
        />
      )}
    </div>
  );
}
