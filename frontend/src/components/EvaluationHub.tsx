import type { FactoryRunSummary, RunSummary } from "../types";
import { Compare } from "./Compare";

export function EvaluationHub({
  factoryRun,
  microscopeRuns,
}: {
  factoryRun: FactoryRunSummary | null;
  microscopeRuns: RunSummary[];
}) {
  const models = (factoryRun?.comparisons?.models ?? {}) as Record<string, {
    response: string; reward: number; repetition: number; safety: number;
  }>;
  const quantize = factoryRun?.metrics.quantize as Record<string, Record<string, number | string>> | undefined;
  return (
    <main className="evaluation-page">
      <header className="page-heading"><div><div className="eyebrow">EVALUATION & RELEASE</div><h1>좋아진 모델인지, 숫자와 샘플로 확인하기</h1><p>reward 하나가 아니라 KL·반복률·안전성과 압축 오차를 함께 봅니다.</p></div></header>
      {Object.keys(models).length ? (
        <div className="model-scorecards">
          {Object.entries(models).map(([name, item]) => (
            <article className="panel" key={name}>
              <div><strong>{name.toUpperCase()}</strong><span>aligned checkpoint</span></div>
              <blockquote>{item.response}</blockquote>
              <footer><span>Reward <b>{item.reward.toFixed(2)}</b></span><span>Repeat <b>{(item.repetition * 100).toFixed(1)}%</b></span><span>Safety <b>{item.safety.toFixed(1)}</b></span></footer>
            </article>
          ))}
        </div>
      ) : <div className="evaluation-empty panel"><h2>공장 평가 단계의 결과가 아직 없어요</h2><p>모델 공장을 실행하면 SFT·DPO·PPO·GRPO를 같은 prompt에서 비교합니다.</p></div>}
      <section className="panel quantization-table">
        <div className="section-heading"><div><div className="eyebrow">QUANTIZATION</div><h2>정밀도와 크기의 교환</h2></div><span className="real-badge">실제 weight quantization</span></div>
        <div className="table-row table-head"><span>Format</span><span>Size</span><span>MSE</span><span>Max error</span><span>비고</span></div>
        {["bf16", "int8", "int4"].map((mode) => {
          const item = quantize?.[mode];
          return <div className="table-row" key={mode}><strong>{mode.toUpperCase()}</strong><span>{item ? `${(Number(item.bytes) / 1024 ** 2).toFixed(1)} MB` : "—"}</span><span>{item ? Number(item.mse).toExponential(2) : "—"}</span><span>{item?.max_error ? Number(item.max_error).toFixed(4) : "—"}</span><span>{String(item?.note ?? (mode === "bf16" ? "기준" : "fused kernel 없음"))}</span></div>;
        })}
      </section>
      <div className="legacy-comparison"><Compare runs={microscopeRuns} /></div>
    </main>
  );
}

