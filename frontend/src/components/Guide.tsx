import { useMemo, useState } from "react";
import type { Bootstrap } from "../types";
import { VectorStrip } from "./VectorStrip";

const chapters = [
  {
    number: "01",
    title: "데이터와 토큰화",
    subtitle: "문장을 모델이 읽을 수 있는 정수열로",
  },
  {
    number: "02",
    title: "임베딩과 위치",
    subtitle: "토큰의 의미와 순서를 하나의 벡터로",
  },
  {
    number: "03",
    title: "Q · K · V와 Attention",
    subtitle: "어떤 앞 토큰을 얼마나 참고할지 계산",
  },
  {
    number: "04",
    title: "Residual · FFN · Logits",
    subtitle: "정보를 보존하고 다음 토큰 점수를 만들기",
  },
  {
    number: "05",
    title: "Loss와 역전파",
    subtitle: "오답의 크기를 따라 파라미터를 수정",
  },
  {
    number: "06",
    title: "자기회귀 생성",
    subtitle: "한 토큰씩 다시 입력하며 문장 완성",
  },
];

export function Guide({
  bootstrap,
  onOpenLab,
}: {
  bootstrap: Bootstrap;
  onOpenLab: () => void;
}) {
  const [chapter, setChapter] = useState(0);
  const [sampleIndex, setSampleIndex] = useState(0);
  const sample = bootstrap.dataset.samples[sampleIndex];
  const fakeTokenVector = useMemo(
    () => Array.from({ length: 32 }, (_, index) => Math.sin(index * 1.7) * 0.08),
    [],
  );
  const fakePositionVector = useMemo(
    () => Array.from({ length: 32 }, (_, index) => Math.cos(index * 0.8) * 0.05),
    [],
  );

  return (
    <main className="guide-layout">
      <aside className="chapter-nav panel">
        <div className="eyebrow">GUIDED COURSE</div>
        <h2>트랜스포머를<br />한 층씩 열어봐요</h2>
        <p>각 단계의 출력이 다음 단계의 입력이 되는 흐름을 따라갑니다.</p>
        <div className="chapter-list">
          {chapters.map((item, index) => (
            <button
              key={item.number}
              className={chapter === index ? "active" : ""}
              onClick={() => setChapter(index)}
            >
              <span>{item.number}</span>
              <div>
                <strong>{item.title}</strong>
                <small>{item.subtitle}</small>
              </div>
            </button>
          ))}
        </div>
        <div className="course-progress">
          <span>학습 흐름</span>
          <strong>{chapter + 1} / {chapters.length}</strong>
          <div><i style={{ width: `${((chapter + 1) / chapters.length) * 100}%` }} /></div>
        </div>
      </aside>

      <section className="lesson panel">
        <div className="lesson-head">
          <div>
            <div className="eyebrow">CHAPTER {chapters[chapter].number}</div>
            <h1>{chapters[chapter].title}</h1>
            <p>{chapters[chapter].subtitle}</p>
          </div>
          <div className="lesson-actions">
            <button
              className="ghost-button"
              disabled={chapter === 0}
              onClick={() => setChapter((value) => value - 1)}
            >
              이전
            </button>
            {chapter < chapters.length - 1 ? (
              <button className="primary-button" onClick={() => setChapter((value) => value + 1)}>
                다음 단계
              </button>
            ) : (
              <button className="primary-button" onClick={onOpenLab}>실험실 열기</button>
            )}
          </div>
        </div>

        {chapter === 0 && (
          <div className="lesson-body">
            <div className="concept-note">
              <span>왜 토큰이 필요할까요?</span>
              <p>
                신경망은 글자가 아니라 숫자를 계산해요. 토크나이저는 문장을 반복 가능한
                규칙으로 나누고, 각 조각에 어휘표의 정수 ID를 붙입니다.
              </p>
            </div>
            <div className="dataset-card">
              <div className="dataset-meta">
                <span className="status-dot" />
                <strong>{bootstrap.dataset.name}</strong>
                <span>{bootstrap.dataset.train_count.toLocaleString()} train</span>
                <span>{bootstrap.dataset.validation_count.toLocaleString()} validation</span>
              </div>
              <p>{sample.text}</p>
              <div className="token-flow">
                {sample.tokens.map((token, index) => (
                  <span key={`${token}-${index}`}>
                    <small>{index + 4}</small>{token}
                  </span>
                ))}
              </div>
              <button
                className="text-button"
                onClick={() =>
                  setSampleIndex((value) => (value + 1) % bootstrap.dataset.samples.length)
                }
              >
                다른 이야기 보기
              </button>
            </div>
            <div className="formula-card">
              <code>x = [BOS, token₁, token₂, …, tokenₙ]</code>
              <span>입력</span>
              <code>y = [token₁, token₂, …, tokenₙ, EOS]</code>
              <span>정답을 한 칸 왼쪽으로 이동</span>
            </div>
          </div>
        )}

        {chapter === 1 && (
          <div className="lesson-body">
            <div className="concept-note">
              <span>같은 단어라도 위치가 다르면?</span>
              <p>
                학습 가능한 토큰 임베딩과 위치 임베딩을 더합니다. 이 모델에서 각 토큰은
                128개의 실수로 표현되고, 최대 32개 토큰을 한 번에 봅니다.
              </p>
            </div>
            <div className="vector-demo">
              <VectorStrip label="E(token)" values={fakeTokenVector} color="coral" />
              <div className="operator">＋</div>
              <VectorStrip label="P(position)" values={fakePositionVector} color="blue" />
              <div className="operator">＝</div>
              <VectorStrip
                label="x₀"
                values={fakeTokenVector.map((value, index) => value + fakePositionVector[index])}
                color="gold"
              />
            </div>
            <div className="formula-card large">
              <code>x₀ = E<sub>token</sub>[id] + E<sub>position</sub>[position]</code>
              <span>[batch, sequence] → [batch, sequence, 128]</span>
            </div>
          </div>
        )}

        {chapter === 2 && (
          <div className="lesson-body">
            <div className="concept-grid">
              {[
                ["Q · Query", "지금 이 토큰이 찾고 싶은 정보"],
                ["K · Key", "각 토큰이 자신을 설명하는 표지"],
                ["V · Value", "선택되었을 때 실제로 가져올 내용"],
              ].map(([title, text]) => (
                <div className="mini-concept" key={title}>
                  <strong>{title}</strong><p>{text}</p>
                </div>
              ))}
            </div>
            <div className="formula-card attention-formula">
              <code>Attention(Q,K,V) = softmax(QKᵀ / √d<sub>k</sub> + M)V</code>
              <span>
                M은 미래 토큰을 −∞로 가리는 causal mask예요. 그래서 모델은 정답을 미리
                볼 수 없습니다.
              </span>
            </div>
            <div className="mask-demo">
              {Array.from({ length: 8 }, (_, row) =>
                Array.from({ length: 8 }, (_, col) => (
                  <span
                    key={`${row}-${col}`}
                    className={col <= row ? "visible" : "masked"}
                    title={col <= row ? "참조 가능" : "미래 토큰: 차단"}
                  >
                    {col <= row ? "✓" : "×"}
                  </span>
                )),
              )}
            </div>
          </div>
        )}

        {chapter === 3 && (
          <div className="lesson-body">
            <div className="pipeline">
              {[
                ["Attention", "문맥에서 필요한 정보 수집"],
                ["＋ Residual", "원래 입력을 더해 정보 보존"],
                ["LayerNorm", "값의 분포를 안정화"],
                ["FFN", "각 토큰을 독립적으로 변환"],
                ["Linear", "2,048개 토큰의 점수 출력"],
              ].map(([title, text], index) => (
                <div className="pipeline-step" key={title}>
                  <small>{String(index + 1).padStart(2, "0")}</small>
                  <strong>{title}</strong><span>{text}</span>
                </div>
              ))}
            </div>
            <div className="formula-card large">
              <code>h′ = h + Attention(LN(h))</code>
              <code>h″ = h′ + FFN(LN(h′))</code>
              <code>logits = Linear(LN(h″))</code>
            </div>
          </div>
        )}

        {chapter === 4 && (
          <div className="lesson-body">
            <div className="loss-visual">
              <div><span>모델 예측</span><strong>garden</strong><b>0.18</b></div>
              <div><span>정답</span><strong>flower</strong><b>1.00</b></div>
              <i>오차를 뒤에서 앞으로 전달</i>
            </div>
            <div className="formula-card large">
              <code>L = −Σ log p(y<sub>t</sub> | x<sub>≤t</sub>)</code>
              <code>θ ← θ − η · ∇<sub>θ</sub>L</code>
              <span>
                Cross Entropy가 정답 토큰의 낮은 확률을 벌점으로 바꾸고, AdamW가 약 93만
                개 파라미터를 조금씩 수정합니다.
              </span>
            </div>
            <div className="concept-note warning">
              <span>Gradient clipping</span>
              <p>기울기 norm을 1.0 이하로 제한해 갑작스러운 큰 업데이트를 막습니다.</p>
            </div>
          </div>
        )}

        {chapter === 5 && (
          <div className="lesson-body">
            <div className="generation-loop">
              {["Once", "upon", "a", "time", "there", "was"].map((token, index) => (
                <span key={token} style={{ animationDelay: `${index * 120}ms` }}>
                  <small>{index === 0 ? "입력" : "생성"}</small>{token}
                </span>
              ))}
            </div>
            <div className="formula-card large">
              <code>p(x<sub>t+1</sub> | x<sub>1</sub>, …, x<sub>t</sub>)</code>
              <span>
                새 토큰을 뽑아 입력 끝에 붙이고 같은 계산을 반복합니다. temperature는
                확률 분포의 날카로움, top-k는 후보의 수를 조절합니다.
              </span>
            </div>
            <button className="primary-button wide" onClick={onOpenLab}>
              이제 실제 모델로 확인하기
            </button>
          </div>
        )}
      </section>
    </main>
  );
}

