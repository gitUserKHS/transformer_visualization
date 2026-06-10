import { useEffect, useState } from "react";
import { api } from "../api";
import type { PreferencePair } from "../types";

export function PreferenceArena({ initialVotes }: { initialVotes: number }) {
  const [pair, setPair] = useState<PreferencePair | null>(null);
  const [votes, setVotes] = useState(initialVotes);
  const [status, setStatus] = useState("");

  const load = () => {
    setStatus("");
    api.nextPreference().then(setPair).catch((error) => setStatus(error.message));
  };
  useEffect(load, []);

  const vote = async (choice: "a" | "b" | "tie") => {
    if (!pair) return;
    try {
      const result = await api.submitPreference(pair.pair_id, choice);
      setVotes(result.user_vote_count);
      setStatus("선호가 저장됐어요. 다음 RM·DPO·RL 배치에 최대 25% 비율로 반영됩니다.");
      setTimeout(load, 650);
    } catch (caught) {
      setStatus(caught instanceof Error ? caught.message : "선호를 저장하지 못했습니다.");
    }
  };

  return (
    <main className="arena-page">
      <header className="page-heading">
        <div><div className="eyebrow">HUMAN FEEDBACK ARENA</div><h1>모델에게 “더 좋은 답”을 가르치기</h1><p>출처를 숨긴 두 응답을 비교해 사람의 선호 데이터를 직접 만듭니다.</p></div>
        <div className="vote-counter"><strong>{votes}</strong><span>local votes</span></div>
      </header>
      <section className="arena-prompt panel">
        <span>PROMPT</span>
        <p>{pair?.prompt ?? "응답 쌍을 준비하는 중…"}</p>
      </section>
      <div className="arena-responses">
        {(["a", "b"] as const).map((side) => (
          <article className="panel" key={side}>
            <div className="anonymous-label">RESPONSE {side.toUpperCase()}</div>
            <blockquote>{pair?.[`response_${side}`] ?? "…"}</blockquote>
            <button className="primary-button wide" disabled={!pair} onClick={() => vote(side)}>
              이 응답이 더 좋아요
            </button>
          </article>
        ))}
      </div>
      <button className="tie-button" disabled={!pair} onClick={() => vote("tie")}>두 응답이 비슷해요</button>
      {status && <div className="arena-status">{status}</div>}
      <section className="feedback-flow panel">
        {["A/B 선택", "user_preferences.jsonl", "최대 25% 샘플링", "RM · DPO · PPO · GRPO"].map((item, index) => <div key={item}><small>{index + 1}</small><strong>{item}</strong></div>)}
      </section>
    </main>
  );
}

