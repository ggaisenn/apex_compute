import type { SVGProps } from "react";

type IconName =
  | "arrow-up-right"
  | "bolt"
  | "book"
  | "box"
  | "check"
  | "chevron-down"
  | "chevron-right"
  | "circle-info"
  | "cpu"
  | "file"
  | "gauge"
  | "layers"
  | "pause"
  | "play"
  | "refresh"
  | "server"
  | "sliders"
  | "spark"
  | "target"
  | "upload"
  | "users";

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

const paths: Record<IconName, string> = {
  "arrow-up-right": "M5 19 19 5M9 5h10v10",
  bolt: "m13 2-9 12h7l-1 8 9-12h-7l1-8Z",
  book: "M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5v-16ZM4 19h16",
  box: "m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm-8 4.5 8 4.5 8-4.5M12 12v9",
  check: "m5 12 4 4L19 6",
  "chevron-down": "m6 9 6 6 6-6",
  "chevron-right": "m9 18 6-6-6-6",
  "circle-info": "M12 17v-5M12 8h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  cpu: "M9 3v2M15 3v2M9 19v2M15 19v2M19 9h2M19 14h2M3 9h2M3 14h2M7 7h10v10H7V7Z",
  file: "M6 3h8l4 4v14H6V3Zm8 0v5h4M9 13h6M9 17h6",
  gauge: "M4 16a8 8 0 1 1 16 0M12 12l3-3M6 19h12",
  layers: "m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 17l9 5 9-5",
  pause: "M8 5v14M16 5v14",
  play: "m8 5 11 7-11 7V5Z",
  refresh: "M20 11a8 8 0 1 0 1 5M20 5v6h-6",
  server: "M4 4h16v6H4V4Zm0 10h16v6H4v-6ZM8 7h.01M8 17h.01M12 7h5M12 17h5",
  sliders: "M4 6h16M4 12h16M4 18h16M8 4v4M16 10v4M10 16v4",
  spark: "m12 2 1.4 6.6L20 10l-6.6 1.4L12 18l-1.4-6.6L4 10l6.6-1.4L12 2ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z",
  target: "M12 3v3M12 18v3M3 12h3M18 12h3M12 7a5 5 0 1 1 0 10 5 5 0 0 1 0-10ZM12 10v2l2 1",
  upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
  users: "M16 19v-1a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v1M10 10a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm6-7a4 4 0 0 1 0 7.7M20 19v-1a4 4 0 0 0-3-3.87",
};

export const Icon = ({ name, size = 18, ...props }: IconProps) => (
  <svg
    aria-hidden="true"
    fill="none"
    height={size}
    viewBox="0 0 24 24"
    width={size}
    {...props}
  >
    <path d={paths[name]} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" />
  </svg>
);

export default Icon;
