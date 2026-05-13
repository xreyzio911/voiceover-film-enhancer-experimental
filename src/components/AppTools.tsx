"use client";

import { useState } from "react";
import AudioTrackSplitter from "./AudioTrackSplitter";
import VoLeveler from "./VoLeveler";
import styles from "./AppTools.module.css";

type ToolId = "vo-leveler" | "audio-splitter";

const TOOLS: { id: ToolId; label: string }[] = [
  { id: "vo-leveler", label: "VO Optimizer" },
  { id: "audio-splitter", label: "Audio Track Splitter" },
];

export default function AppTools() {
  const [activeTool, setActiveTool] = useState<ToolId>("vo-leveler");

  return (
    <div className={styles.layout}>
      <div className={styles.tabs} role="tablist" aria-label="Audio tools">
        {TOOLS.map((tool) => (
          <button
            key={tool.id}
            type="button"
            role="tab"
            aria-selected={activeTool === tool.id}
            className={`${styles.tab} ${activeTool === tool.id ? styles.tabActive : ""}`}
            onClick={() => setActiveTool(tool.id)}
          >
            {tool.label}
          </button>
        ))}
      </div>
      {activeTool === "vo-leveler" ? <VoLeveler /> : <AudioTrackSplitter />}
    </div>
  );
}
