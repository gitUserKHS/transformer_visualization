import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, openFactorySocket, openRunSocket } from "./api";
import { activityLabel, isActiveRun, statusLabel } from "./activity";
import type {
  ConnectionState,
  DeviceTelemetry,
  FactoryRunSummary,
  RunSummary,
} from "./types";

type Track = "factory" | "microscope";
type TrackRun = FactoryRunSummary | RunSummary;

const retryDelays = [500, 1000, 2000, 5000];

export function reduceActivityEvent<T extends TrackRun>(
  current: T | null,
  event: Record<string, unknown>,
): T | null {
  if (event.summary) return event.summary as T;
  if (!current) return current;
  if (event.type === "status") {
    return { ...current, status: String(event.status ?? current.status) };
  }
  if ("stage" in current) {
    const factory = current as FactoryRunSummary;
    const eventType = String(event.type ?? "");
    return {
      ...factory,
      stage:
        eventType === "stage" && event.stage
          ? String(event.stage)
          : factory.stage,
      stage_step:
        typeof event.step === "number" ? event.step : factory.stage_step,
      metrics:
        eventType && !["stage", "heartbeat"].includes(eventType)
          ? { ...factory.metrics, [eventType]: event }
          : factory.metrics,
    } as T;
  }
  return current;
}

function useTrackedRun<T extends TrackRun>(
  track: Track,
  notify: (message: string, failed?: boolean, info?: boolean) => void,
) {
  const [run, setRun] = useState<T | null>(null);
  const [events, setEvents] = useState<Array<Record<string, unknown>>>([]);
  const [telemetry, setTelemetry] = useState<DeviceTelemetry | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const socketRef = useRef<WebSocket | null>(null);
  const lastIndexRef = useRef(-1);
  const retryRef = useRef(0);
  const retryTimerRef = useRef<number | null>(null);
  const lastMessageRef = useRef(0);
  const previousStatusRef = useRef<string | null>(null);

  const adopt = useCallback((next: T | null) => {
    setRun(next);
  }, []);

  useEffect(() => {
    const runId = run?.id;
    const active = isActiveRun(run);
    if (!runId || !active) {
      socketRef.current?.close();
      socketRef.current = null;
      setConnection("idle");
      return;
    }

    let disposed = false;
    let runMissing = false;
    if (previousStatusRef.current === null) {
      previousStatusRef.current = run.status;
    }

    const connect = () => {
      if (disposed) return;
      setConnection(retryRef.current ? "reconnecting" : "connecting");
      const open = track === "factory" ? openFactorySocket : openRunSocket;
      const socket = open(
        runId,
        lastIndexRef.current,
        (event) => {
          lastMessageRef.current = Date.now();
          setConnection("connected");
          if (event.type === "error" && event.reason === "not_found") {
            runMissing = true;
            setRun(null);
            setEvents([]);
            setConnection("idle");
            notify(
              "서버가 재시작되어 이전 학습 실행을 종료 처리했어요.",
              false,
              true,
            );
            socketRef.current?.close();
            return;
          }
          if (typeof event.index === "number") {
            if (event.index <= lastIndexRef.current) return;
            lastIndexRef.current = event.index;
          }
          if (event.telemetry) {
            setTelemetry(event.telemetry as DeviceTelemetry);
          }
          if (event.type !== "heartbeat") {
            setEvents((current) => [...current.slice(-199), event]);
          }
          const summary = event.summary as T | undefined;
          const nextStatus =
            summary?.status ??
            (event.type === "status" ? String(event.status) : undefined);
          if (
            nextStatus &&
            ["completed", "failed", "stopped"].includes(nextStatus) &&
            previousStatusRef.current &&
            !["completed", "failed", "stopped"].includes(
              previousStatusRef.current,
            )
          ) {
            notify(
              `${track === "factory" ? "모델 공장" : "학습 현미경"} ${
                nextStatus === "completed" ? "실행이 완료됐어요." : statusLabel(nextStatus)
              }`,
              nextStatus === "failed",
            );
          }
          if (nextStatus) previousStatusRef.current = nextStatus;
          setRun((current) => reduceActivityEvent(current, event));
        },
        () => {
          retryRef.current = 0;
          lastMessageRef.current = Date.now();
          setConnection("connected");
        },
        () => {
          if (disposed || runMissing) return;
          setConnection("reconnecting");
          const delay =
            retryDelays[Math.min(retryRef.current, retryDelays.length - 1)];
          retryRef.current += 1;
          retryTimerRef.current = window.setTimeout(connect, delay);
        },
      );
      socketRef.current = socket;
    };

    lastIndexRef.current = -1;
    retryRef.current = 0;
    lastMessageRef.current = Date.now();
    setEvents([]);
    connect();

    const staleTimer = window.setInterval(() => {
      if (Date.now() - lastMessageRef.current > 3500) {
        setConnection("stale");
      }
    }, 1000);

    return () => {
      disposed = true;
      window.clearInterval(staleTimer);
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [notify, run?.id, isActiveRun(run), track]);

  return { run, adopt, events, telemetry, connection };
}

export function useActivity() {
  const [notice, setNotice] = useState<{
    message: string;
    failed: boolean;
    info: boolean;
  } | null>(null);
  const notify = useCallback((message: string, failed = false, info = false) => {
    setNotice({ message, failed, info });
  }, []);
  const factory = useTrackedRun<FactoryRunSummary>("factory", notify);
  const microscope = useTrackedRun<RunSummary>("microscope", notify);

  useEffect(() => {
    api.activity()
      .then((current) => {
        factory.adopt(current.factory);
        microscope.adopt(current.microscope);
      })
      .catch(() => {
        notify("현재 학습 상태를 불러오지 못했어요.", true);
      });
  }, [factory.adopt, microscope.adopt, notify]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const active = useMemo(
    () => {
      const items: Array<{
        track: Track;
        run: FactoryRunSummary | RunSummary;
      }> = [];
      if (factory.run && isActiveRun(factory.run)) {
        items.push({ track: "factory", run: factory.run });
      }
      if (microscope.run && isActiveRun(microscope.run)) {
        items.push({ track: "microscope", run: microscope.run });
      }
      return items;
    },
    [factory.run, microscope.run],
  );

  useEffect(() => {
    const primary = active[0];
    document.title = notice
      ? `[${notice.info ? "알림" : notice.failed ? "실패" : "완료"}] Transformer Learning Studio`
      : primary
        ? `[${Math.round((primary.run.overall_progress ?? 0) * 100)}% · ${activityLabel(
            primary.run,
          )}] Transformer Learning Studio`
        : "Transformer Learning Studio";
  }, [active, notice]);

  return { factory, microscope, active, notice };
}
