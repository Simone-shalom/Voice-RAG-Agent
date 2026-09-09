import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SourcePlayer from "../SourcePlayer";

describe("SourcePlayer speaker labels", () => {
  it("prefixes the citation text with the speaker label when present", () => {
    render(
      <SourcePlayer
        sources={[
          { episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there", speaker_label: "Speaker 0" },
        ]}
      />
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });

  it("renders unprefixed citation text when speaker_label is null", () => {
    render(
      <SourcePlayer
        sources={[{ episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there", speaker_label: null }]}
      />
    );
    expect(screen.queryByText(/^Speaker \d/)).not.toBeInTheDocument();
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });

  it("renders unprefixed citation text when speaker_label is entirely absent", () => {
    render(
      <SourcePlayer sources={[{ episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there" }]} />
    );
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });
});
