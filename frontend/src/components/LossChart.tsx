interface Point {
  step: number;
  loss: number;
}

export function LossChart({ points }: { points: Point[] }) {
  const width = 700;
  const height = 220;
  const padding = 28;
  const data = points.length ? points : [{ step: 0, loss: 0 }];
  const maxStep = Math.max(...data.map((point) => point.step), 1);
  const maxLoss = Math.max(...data.map((point) => point.loss), 1);
  const minLoss = Math.min(...data.map((point) => point.loss), maxLoss * 0.8);
  const range = Math.max(maxLoss - minLoss, 0.1);
  const path = data
    .map((point, index) => {
      const x = padding + (point.step / maxStep) * (width - padding * 2);
      const y =
        height -
        padding -
        ((point.loss - minLoss) / range) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="chart-wrap" aria-label="학습 손실 곡선">
      {points.length === 0 ? (
        <div className="empty-chart">학습을 시작하면 손실 곡선이 그려져요.</div>
      ) : null}
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          <linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#da7566" stopOpacity="0.25" />
            <stop offset="1" stopColor="#da7566" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((ratio) => (
          <line
            key={ratio}
            x1={padding}
            x2={width - padding}
            y1={padding + ratio * (height - padding * 2)}
            y2={padding + ratio * (height - padding * 2)}
            className="grid-line"
          />
        ))}
        <path d={path} fill="none" className="loss-line" />
        <text x={padding} y={height - 6} className="axis-label">
          0 step
        </text>
        <text x={width - padding} y={height - 6} textAnchor="end" className="axis-label">
          {maxStep} step
        </text>
        <text x={padding} y={18} className="axis-label">
          loss {maxLoss.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}

