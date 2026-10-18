export const executionPhases = ["A load", "B load", "FMA", "C store"] as const;

export type ExecutionPhase = (typeof executionPhases)[number];

export interface NaiveAccess {
  outputRow: number;
  outputColumn: number;
  innerRow: number;
  innerColumn: number;
}

export const getExecutionPhase = (step: number): ExecutionPhase =>
  executionPhases[((step % executionPhases.length) + executionPhases.length) % executionPhases.length];

/**
 * Return the row-major access pattern used by the naive kernel for one
 * representative output. A and C stay row-local, while B is read down a
 * column. In row-major storage each B read therefore advances by a full row
 * stride and usually lands on a cache line that has not been reused.
 */
export const getNaiveAccess = (step: number, rows: number, columns: number): NaiveAccess => {
  const safeRows = Math.max(1, rows);
  const safeColumns = Math.max(1, columns);
  const outputIndex = ((step % (safeRows * safeColumns)) + safeRows * safeColumns) % (safeRows * safeColumns);

  return {
    outputRow: Math.floor(outputIndex / safeColumns),
    outputColumn: outputIndex % safeColumns,
    innerRow: ((step % safeRows) + safeRows) % safeRows,
    innerColumn: ((step % safeColumns) + safeColumns) % safeColumns,
  };
};
