import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import FileUploader, { Episode } from "../FileUploader";

const baseEpisode: Episode = {
  id: "e1",
  filename: "test.mp3",
  chunk_count: 5,
  source_url: "https://example.com/audio.mp3",
  summary: "A great episode about testing.",
  chapters: [
    { start_ts: 0, title: "Introduction" },
    { start_ts: 120, title: "Deep dive" },
  ],
};

interface RenderOverrides {
  selectedEpisodeId?: string | null;
  episodes?: Episode[];
}

function renderUploader(overrides: RenderOverrides = {}) {
  return render(
    <FileUploader
      selectedEpisodeId={overrides.selectedEpisodeId ?? null}
      onSelectEpisode={vi.fn()}
      episodes={overrides.episodes ?? [baseEpisode]}
      episodesLoaded={true}
      onIngested={vi.fn()}
    />
  );
}

describe("FileUploader focused-episode summary/chapters panel", () => {
  it("does not show the summary panel when the episode is not selected", () => {
    renderUploader({ selectedEpisodeId: null });
    expect(screen.queryByText(/A great episode about testing/)).not.toBeInTheDocument();
  });

  it("shows the summary and chapter list when the episode is selected", () => {
    renderUploader({ selectedEpisodeId: "e1" });
    expect(screen.getByText(/A great episode about testing/)).toBeInTheDocument();
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Deep dive")).toBeInTheDocument();
  });

  it("does not render a chapters section when the episode has no chapters", () => {
    const noChapters = { ...baseEpisode, chapters: null };
    renderUploader({ selectedEpisodeId: "e1", episodes: [noChapters] });
    expect(screen.queryByText("Introduction")).not.toBeInTheDocument();
  });

  it("clicking a chapter sets the inline audio player's time via a #t= fragment", async () => {
    const user = userEvent.setup();
    renderUploader({ selectedEpisodeId: "e1" });

    const chapterButton = screen.getByRole("button", { name: /Deep dive/ });
    await user.click(chapterButton);

    const audio = screen.getByTestId("episode-audio-player") as HTMLAudioElement;
    expect(audio.src).toContain("#t=120");
  });

  it("does not render an audio player when the episode has no source_url", () => {
    const noUrl = { ...baseEpisode, source_url: null };
    renderUploader({ selectedEpisodeId: "e1", episodes: [noUrl] });
    expect(screen.queryByTestId("episode-audio-player")).not.toBeInTheDocument();
  });
});
