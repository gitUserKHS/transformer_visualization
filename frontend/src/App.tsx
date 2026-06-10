import { useEffect, useState } from "react";
import { api } from "./api";
import { AlignmentStudio } from "./components/AlignmentStudio";
import { EvaluationHub } from "./components/EvaluationHub";
import { Factory } from "./components/Factory";
import { InferenceEngine } from "./components/InferenceEngine";
import { Microscope } from "./components/Microscope";
import { PreferenceArena } from "./components/PreferenceArena";
import type { Bootstrap, FactoryRunSummary, RunSummary, Tab } from "./types";

function App() {
  const [tab, setTab] = useState<Tab>("factory");
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [error, setError] = useState("");
  const [factoryRun, setFactoryRun] = useState<FactoryRunSummary | null>(null);

  useEffect(() => {
    api.bootstrap()
      .then((data) => {
        setBootstrap(data);
        setHistory(data.history);
        setFactoryRun(data.factory.history[0] ?? null);
      })
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : "서버에 연결하지 못했습니다."),
      );
  }, []);

  const recordRun = (run: RunSummary) => {
    setHistory((current) => [run, ...current.filter((item) => item.id !== run.id)].slice(0, 5));
  };

  if (error) {
    return (
      <div className="fatal-screen">
        <div className="brand-mark">T</div>
        <h1>학습 서버를 깨우지 못했어요</h1>
        <p>{error}</p>
        <code>python -m uvicorn backend.app.main:app --reload</code>
      </div>
    );
  }
  if (!bootstrap) {
    return (
      <div className="loading-screen">
        <div className="loading-rings"><i /><i /><i /></div>
        <strong>모델과 데이터셋을 펼치는 중…</strong>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setTab("factory")}>
          <span className="brand-mark">T</span>
          <div><strong>Transformer</strong><small>Learning Studio</small></div>
        </button>
        <nav aria-label="주 메뉴">
          <button className={tab === "factory" ? "active" : ""} onClick={() => setTab("factory")}>모델 공장</button>
          <button className={tab === "microscope" ? "active" : ""} onClick={() => setTab("microscope")}>학습 현미경</button>
          <button className={tab === "alignment" ? "active" : ""} onClick={() => setTab("alignment")}>사후학습</button>
          <button className={tab === "arena" ? "active" : ""} onClick={() => setTab("arena")}>피드백 아레나</button>
          <button className={tab === "inference" ? "active" : ""} onClick={() => setTab("inference")}>추론 엔진</button>
          <button className={tab === "evaluation" ? "active" : ""} onClick={() => setTab("evaluation")}>평가·비교</button>
        </nav>
        <div className="top-meta">
          <span className={bootstrap.model.checkpoint_ready ? "ready" : "random"}>
            {bootstrap.model.checkpoint_ready ? "체크포인트 준비됨" : "초기 모델"}
          </span>
          <b>{bootstrap.model.parameter_count.toLocaleString()}</b>
          <small>parameters</small>
        </div>
      </header>
      {tab === "factory" && <Factory bootstrap={bootstrap} run={factoryRun} onRunChange={setFactoryRun} />}
      {tab === "microscope" && <Microscope bootstrap={bootstrap} onRunFinished={recordRun} />}
      {tab === "alignment" && <AlignmentStudio run={factoryRun} />}
      {tab === "arena" && <PreferenceArena initialVotes={bootstrap.factory.data.user_vote_count} />}
      {tab === "inference" && <InferenceEngine bootstrap={bootstrap} />}
      {tab === "evaluation" && <EvaluationHub factoryRun={factoryRun} microscopeRuns={history} />}
      <footer>
        <span>Transformer Learning Studio</span>
        <p>
          Dataset: <a href={bootstrap.dataset.source} target="_blank" rel="noreferrer">TinyStories</a>
          {" · "}{bootstrap.dataset.license} · 교육용 고정 부분집합
        </p>
      </footer>
    </div>
  );
}

export default App;
