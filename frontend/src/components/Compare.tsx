import type { RunSummary } from "../types";
import { LossChart } from "./LossChart";

export function Compare({ runs }: { runs: RunSummary[] }) {
  if (!runs.length) {
    return (
      <main className="compare-empty panel">
        <div className="empty-orbit">∿</div>
        <h1>아직 비교할 실험이 없어요</h1>
        <p>실험실에서 학습을 마치면 최근 5개의 설정과 결과가 이곳에 저장됩니다.</p>
      </main>
    );
  }
  return (
    <main className="compare-page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">EXPERIMENT HISTORY</div>
          <h1>같은 데이터, 다른 학습</h1>
          <p>설정의 차이가 손실과 생성 결과를 어떻게 바꾸었는지 나란히 살펴보세요.</p>
        </div>
      </header>
      <div className="compare-grid">
        {runs.slice(0, 5).map((run, index) => (
          <article className="run-card panel" key={run.id}>
            <div className="run-card-head">
              <span className={`run-swatch swatch-${index}`} />
              <div><strong>{run.name}</strong><small>{run.id}</small></div>
              <b>{run.status}</b>
            </div>
            <div className="run-numbers">
              <div><span>최종 Loss</span><strong>{run.last_metrics.loss?.toFixed(3) ?? "—"}</strong></div>
              <div><span>Steps</span><strong>{run.step}</strong></div>
              <div><span>LR</span><strong>{run.config.learning_rate}</strong></div>
            </div>
            <LossChart points={run.losses} />
            <blockquote>{run.sample || "생성 샘플이 기록되지 않았습니다."}</blockquote>
          </article>
        ))}
      </div>
      <div className="comparison-table panel">
        <div className="table-row table-head">
          <span>실험</span><span>레이어 / 헤드</span><span>배치</span><span>Loss</span><span>Perplexity</span>
        </div>
        {runs.slice(0, 5).map((run) => (
          <div className="table-row" key={run.id}>
            <strong>{run.name}</strong>
            <span>{run.config.model.n_layers} / {run.config.model.n_heads}</span>
            <span>{run.config.batch_size}</span>
            <span>{run.last_metrics.loss?.toFixed(4) ?? "—"}</span>
            <span>{run.last_metrics.perplexity?.toFixed(2) ?? "—"}</span>
          </div>
        ))}
      </div>
    </main>
  );
}

