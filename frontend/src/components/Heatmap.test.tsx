import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Heatmap } from "./Heatmap";

describe("Heatmap", () => {
  it("shows the selected attention value", () => {
    render(<Heatmap matrix={[[1, 0], [0.25, 0.75]]} tokens={["once", "upon"]} />);
    fireEvent.mouseEnter(screen.getByLabelText("upon가 once에 주는 가중치 0.2500"));
    expect(screen.getByText("upon → once = 0.2500")).toBeInTheDocument();
  });
});

