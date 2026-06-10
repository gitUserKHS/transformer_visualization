import { useState } from "react";
import type { Bootstrap, RunSummary } from "../types";
import { Guide } from "./Guide";
import { Lab } from "./Lab";

export function Microscope({
  bootstrap,
  onRunFinished,
}: {
  bootstrap: Bootstrap;
  onRunFinished: (run: RunSummary) => void;
}) {
  const [mode, setMode] = useState<"guide" | "lab">("guide");
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
        <Lab bootstrap={bootstrap} onRunFinished={onRunFinished} />
      )}
    </div>
  );
}

