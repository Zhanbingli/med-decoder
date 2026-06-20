/**
 * The signature element: a scrolling ECG rhythm strip that reflects app state.
 * idle = calm baseline, recording = fast arterial-red QRS, busy = scan cadence.
 * One PQRST beat is tiled and scrolled exactly one tile-width, so it loops
 * seamlessly. Honors prefers-reduced-motion via CSS.
 */
const BASE = 30;
const TILE = 110;

function beatPoints(x0: number): string {
  const p: [number, number][] = [
    [0, 0], [16, 0],                 // baseline
    [22, -7], [28, 0],               // P wave
    [42, 0],                         // PR segment
    [46, 5], [50, -26], [54, 12], [58, 0], // QRS: Q dip, tall R spike, S dip
    [74, 0],                         // ST segment
    [86, -11], [98, 0],              // T wave
    [TILE, 0],                       // baseline to tile end
  ];
  return p.map(([x, y]) => `${x0 + x},${BASE + y}`).join(" ");
}

const BEATS = 12;
const POINTS = Array.from({ length: BEATS }, (_, i) => beatPoints(i * TILE)).join(" ");

export function Ecg({ state }: { state: "idle" | "recording" | "busy" }) {
  return (
    <div className={`ecg ${state}`} aria-hidden="true">
      <svg width={BEATS * TILE} height="52" viewBox={`0 0 ${BEATS * TILE} 52`} preserveAspectRatio="none">
        <polyline points={POINTS} />
      </svg>
    </div>
  );
}
