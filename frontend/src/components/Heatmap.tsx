import { useMemo, useState } from "react";

export function Heatmap({
  matrix,
  tokens,
}: {
  matrix: number[][];
  tokens: string[];
}) {
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(null);
  const visible = useMemo(() => matrix.slice(0, 32).map((row) => row.slice(0, 32)), [matrix]);
  if (!visible.length) return <div className="empty-state">attention 데이터가 아직 없어요.</div>;
  const cell = Math.max(12, Math.min(25, 560 / visible.length));
  return (
    <div className="heatmap-shell">
      <div
        className="heatmap"
        style={{
          gridTemplateColumns: `repeat(${visible.length}, ${cell}px)`,
          gridTemplateRows: `repeat(${visible.length}, ${cell}px)`,
        }}
        role="grid"
        aria-label="attention heatmap"
      >
        {visible.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => (
            <button
              className="heat-cell"
              key={`${rowIndex}-${columnIndex}`}
              style={{
                backgroundColor: `rgba(196, 73, 83, ${Math.max(0.03, value)})`,
                width: cell,
                height: cell,
              }}
              title={`${tokens[rowIndex]} → ${tokens[columnIndex]}: ${value.toFixed(4)}`}
              onMouseEnter={() => setHovered({ row: rowIndex, col: columnIndex })}
              onMouseLeave={() => setHovered(null)}
              aria-label={`${tokens[rowIndex]}가 ${tokens[columnIndex]}에 주는 가중치 ${value.toFixed(4)}`}
            />
          )),
        )}
      </div>
      <div className="heatmap-caption">
        {hovered
          ? `${tokens[hovered.row]} → ${tokens[hovered.col]} = ${visible[hovered.row][hovered.col].toFixed(4)}`
          : "셀을 가리키면 토큰 사이의 attention 가중치를 볼 수 있어요."}
      </div>
    </div>
  );
}

