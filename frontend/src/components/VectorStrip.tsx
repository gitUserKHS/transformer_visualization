export function VectorStrip({
  label,
  values,
  color = "coral",
}: {
  label: string;
  values: number[];
  color?: "coral" | "blue" | "gold";
}) {
  const max = Math.max(...values.map((value) => Math.abs(value)), 0.001);
  return (
    <div className="vector-row">
      <div className="vector-label">{label}</div>
      <div className="vector-strip">
        {values.slice(0, 32).map((value, index) => {
          const strength = Math.abs(value) / max;
          return (
            <span
              key={index}
              className={`vector-cell ${color} ${value < 0 ? "negative" : ""}`}
              style={{ opacity: 0.18 + strength * 0.82 }}
              title={`[${index}] ${value.toFixed(5)}`}
            />
          );
        })}
      </div>
      <code>{values.slice(0, 3).map((value) => value.toFixed(3)).join(", ")}…</code>
    </div>
  );
}

