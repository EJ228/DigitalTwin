import { useEffect, useState } from "react";

/**
 * Wall-clock time in India Standard Time.
 *
 * The zone is pinned to Asia/Kolkata rather than read from the browser, so the
 * dashboard reads the same on a laptop that happens to be set to another zone --
 * which is the usual case when this is demoed off a travelling machine.
 *
 * Note this is distinct from the replay clock in the snapshot: that one is the
 * position of the recorded run and starts at 06:00 of a synthetic shift day.
 */
const FMT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Kolkata",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export function useIstClock() {
  const [now, setNow] = useState(() => FMT.format(new Date()));

  useEffect(() => {
    // Tick on the minute boundary rather than every second -- the display only
    // has minute resolution, so a 1 Hz interval would be pure re-render churn.
    let timer;
    const schedule = () => {
      const ms = 60000 - (Date.now() % 60000);
      timer = setTimeout(() => {
        setNow(FMT.format(new Date()));
        schedule();
      }, ms + 50);
    };
    schedule();
    return () => clearTimeout(timer);
  }, []);

  return now;
}
