import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Bootstrap } from "../types";
import { NeuralFlowLab } from "./NeuralFlowLab";

const bootstrap = {
  factory: {
    model: {
      vocab_size: 8192,
      context_length: 128,
      d_model: 384,
      n_layers: 8,
      n_heads: 8,
      n_kv_heads: 2,
      d_ff: 1024,
    },
  },
} as Bootstrap;

describe("NeuralFlowLab", () => {
  it("switches the activation inspector and renders a 5x5 surface", () => {
    const { container } = render(
      <NeuralFlowLab bootstrap={bootstrap} trace={null} />,
    );

    fireEvent.click(screen.getByRole("button", { name: /block 4/ }));

    expect(
      screen.getByRole("heading", { name: "Transformer block 4" }),
    ).toBeInTheDocument();
    expect(container.querySelectorAll(".surface-heatmap span")).toHaveLength(25);
    expect(container.querySelectorAll(".attention-mini i")).toHaveLength(144);
  });
});
