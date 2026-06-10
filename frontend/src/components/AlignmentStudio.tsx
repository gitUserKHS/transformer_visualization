import { useState } from "react";
import type { FactoryRunSummary } from "../types";

const methods = [
  {
    id: "sft",
    label: "SFT",
    title: "정답 구간만 교사 강요",
    formula: "L_SFT = −Σ log πθ(yₜ | x, y<ₜ)",
    description: "prompt는 −100으로 가려 gradient를 만들지 않고, 모범 response 토큰만 학습합니다.",
  },
  {
    id: "reward",
    label: "Reward Model",
    title: "사람의 비교를 하나의 점수로",
    formula: "L_RM = −log σ(r(x,y⁺) − r(x,y⁻))",
    description: "chosen의 점수가 rejected보다 높아지도록 Bradley–Terry pairwise loss를 최적화합니다.",
  },
  {
    id: "ppo",
    label: "PPO",
    title: "좋은 행동은 늘리되 너무 멀리 가지 않게",
    formula: "min(rₜAₜ, clip(rₜ,1−ε,1+ε)Aₜ) − βKL",
    description: "reward, value, advantage와 old/new policy ratio를 사용하고 reference KL로 급격한 변화를 막습니다.",
  },
  {
    id: "dpo",
    label: "DPO",
    title: "보상모델 없이 선호 margin을 직접",
    formula: "−log σ(β[(logπy⁺−logπy⁻) − (logπref y⁺−logπref y⁻)])",
    description: "chosen과 rejected의 상대 log-probability를 reference 모델과 비교해 직접 업데이트합니다.",
  },
  {
    id: "grpo",
    label: "GRPO",
    title: "같은 질문의 답들끼리 상대 평가",
    formula: "Aᵢ = (rᵢ − mean(r_group)) / (std(r_group)+ε)",
    description: "별도 value model 대신 같은 prompt에서 생성한 응답 그룹의 평균과 표준편차를 baseline으로 씁니다.",
  },
];

function number(metric: Record<string, unknown> | undefined, key: string, fallback: number) {
  const value = metric?.[key];
  return typeof value === "number" ? value : fallback;
}

export function AlignmentStudio({ run }: { run: FactoryRunSummary | null }) {
  const [selected, setSelected] = useState("ppo");
  const method = methods.find((item) => item.id === selected)!;
  const metric = run?.metrics[selected];
  const ppoRatio = number(metric, "ratio", 1.16);
  const advantage = number(metric, "advantage", 0.84);
  const rewardGap = number(run?.metrics.reward, "reward_gap", 1.32);
  const dpoMargin = number(run?.metrics.dpo, "implicit_reward_margin", 0.48);
  const group = (run?.metrics.grpo?.ranking as Array<Record<string, unknown>> | undefined) ?? [
    { response: "The child shared the map with a friend.", reward: 1.4, advantage: 1.2 },
    { response: "They followed the bright path home.", reward: 0.8, advantage: 0.3 },
    { response: "The map was a map and map.", reward: -0.4, advantage: -0.7 },
    { response: "Nothing happened.", reward: -0.9, advantage: -1.1 },
  ];

  return (
    <main className="alignment-page">
      <header className="page-heading">
        <div><div className="eyebrow">POST-TRAINING LAB</div><h1>같은 SFT 모델, 다섯 가지 학습 신호</h1><p>점수 하나가 어떤 확률과 gradient로 바뀌는지 실제 공장 지표와 함께 해부합니다.</p></div>
        <span className="real-badge">GPU gradient</span>
      </header>
      <div className="method-tabs">
        {methods.map((item) => <button key={item.id} className={selected === item.id ? "active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
      </div>
      <section className="method-hero panel">
        <div><div className="eyebrow">{method.label}</div><h2>{method.title}</h2><p>{method.description}</p></div>
        <code>{method.formula}</code>
      </section>

      {selected === "sft" && (
        <section className="alignment-grid">
          <article className="panel token-mask-card">
            <h3>Label masking</h3>
            <div className="mask-tokens">
              {["<user>", "Continue", "this", "story", "<assistant>", "The", "child", "smiled", "."].map((token, index) => (
                <span className={index < 5 ? "masked" : "learned"} key={token}><small>{index < 5 ? "−100" : "target"}</small>{token}</span>
              ))}
            </div>
            <p>회색 prompt는 문맥으로 사용하지만 loss에서는 제외됩니다.</p>
          </article>
          <article className="panel parameter-map">
            <h3>LoRA가 움직이는 곳</h3>
            {["Q projection", "K / V projection", "Output projection", "SwiGLU gate / up / down"].map((label, index) => <div key={label}><span>{label}</span><i style={{ width: `${72 + index * 7}%` }} /><b>r=8</b></div>)}
            <footer><strong>2.69%</strong><span>trainable</span><small>나머지 base weight는 frozen</small></footer>
          </article>
        </section>
      )}

      {selected === "reward" && (
        <section className="reward-stage panel">
          <div className="response-score chosen"><span>CHOSEN</span><blockquote>She shared the treasure with her friend and they returned home safely.</blockquote><strong>+{(0.7 + rewardGap / 2).toFixed(2)}</strong></div>
          <div className="reward-gap"><span>reward gap</span><strong>{rewardGap.toFixed(2)}</strong><i /></div>
          <div className="response-score rejected"><span>REJECTED</span><blockquote>Treasure treasure treasure. Nothing made sense and the story stopped.</blockquote><strong>{(0.7 - rewardGap / 2).toFixed(2)}</strong></div>
        </section>
      )}

      {selected === "ppo" && (
        <section className="alignment-grid">
          <article className="panel ppo-ratio">
            <h3>Policy ratio와 clipping</h3>
            <div className="ratio-axis"><i className="clip-left" /><i className="clip-right" /><span style={{ left: `${Math.min(96, Math.max(4, (ppoRatio - 0.7) / 0.6 * 100))}%` }} /></div>
            <div className="ratio-labels"><b>0.8</b><strong>rₜ = {ppoRatio.toFixed(3)}</strong><b>1.2</b></div>
            <p>{ppoRatio > 1.2 ? "clip 경계를 넘어 업데이트 이득이 제한됩니다." : "신뢰 구간 안에서 정책 확률을 업데이트합니다."}</p>
          </article>
          <article className="panel advantage-stack">
            <h3>Reward → Advantage</h3>
            <div><span>Reward</span><strong>{number(metric, "reward", 1.74).toFixed(3)}</strong></div>
            <div><span>− KL penalty</span><strong>{number(metric, "kl", 0.12).toFixed(3)}</strong></div>
            <div><span>− Value baseline</span><strong>{number(metric, "value", 0.78).toFixed(3)}</strong></div>
            <footer><span>Advantage</span><strong>{advantage.toFixed(3)}</strong></footer>
          </article>
        </section>
      )}

      {selected === "dpo" && (
        <section className="dpo-balance panel">
          <div><span>Policy chosen − rejected</span><strong>{number(metric, "policy_margin", 0.86).toFixed(3)}</strong><i style={{ width: "76%" }} /></div>
          <div><span>Reference chosen − rejected</span><strong>{number(metric, "reference_margin", 0.38).toFixed(3)}</strong><i style={{ width: "42%" }} /></div>
          <div className="dpo-result"><span>Implicit reward margin</span><strong>{dpoMargin.toFixed(3)}</strong><p>양수면 현재 policy가 reference보다 선호 응답을 더 강하게 선택합니다.</p></div>
        </section>
      )}

      {selected === "grpo" && (
        <section className="grpo-group">
          {group.map((item, index) => (
            <article className="panel" key={index}>
              <div><span>RANK {index + 1}</span><b>A {Number(item.advantage).toFixed(2)}</b></div>
              <blockquote>{String(item.response)}</blockquote>
              <footer><span>reward</span><strong>{Number(item.reward).toFixed(3)}</strong></footer>
            </article>
          ))}
          <div className="group-baseline">group mean을 중심으로 advantage 평균은 <strong>0</strong></div>
        </section>
      )}

      <section className="reward-hacking panel">
        <div className="warning-icon">!</div>
        <div><div className="eyebrow">REWARD HACKING</div><h3>점수는 올랐는데 답은 나빠질 수 있어요</h3><p>“reward reward helpful helpful”처럼 proxy의 허점을 반복하면 보상은 높아져도 다양성과 유용성은 떨어집니다.</p></div>
        <div className="split-score"><span>Proxy reward <b>0.95</b></span><span>Quality <b>0.12</b></span></div>
      </section>
    </main>
  );
}

