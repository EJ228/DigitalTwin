import { useCallback, useEffect, useRef, useState } from "react";
import { streamUrl } from "./api";

/**
 * Live replay socket.
 *
 * Owns the connection and exposes the latest snapshot plus controls. Speed,
 * pause, scrub and the blind toggle are sent over the OPEN socket rather than
 * reconnecting, so changing them does not drop a frame -- which matters because
 * the blind-station toggle is a live demo moment and a reconnect flash would
 * ruin it.
 */
export function useTwinStream({ run = "run_s7", initialT = 40000, speed = 60 }) {
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [paused, setPausedState] = useState(false);
  const [blind, setBlindState] = useState("");
  const ws = useRef(null);
  const retry = useRef(null);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      const sock = new WebSocket(streamUrl({ run, t: initialT, speed, blind: "" }));
      ws.current = sock;
      sock.onopen = () => { setConnected(true); setError(null); };
      sock.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.error) setError(d.error);
        else setSnapshot(d);
      };
      // Ignore errors from a socket the cleanup has already abandoned. In dev,
      // StrictMode double-mounts this effect and the discarded first socket
      // errors out, which would otherwise flash the banner on every load.
      sock.onerror = () => { if (!closed) setError("connection failed"); };
      sock.onclose = () => {
        if (closed) return;
        setConnected(false);
        retry.current = setTimeout(connect, 1500);
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry.current);
      ws.current?.close();
    };
    // Reconnect only on run change. Everything else is sent over the socket.
  }, [run]); // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback((msg) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(msg));
  }, []);

  return {
    snapshot, connected, error, paused, blind,
    setSpeed: (v) => send({ speed: v }),
    seek: (t) => send({ t }),
    setPaused: (v) => { setPausedState(v); send({ paused: v }); },
    setBlind: (v) => { setBlindState(v); send({ blind: v }); },
  };
}
