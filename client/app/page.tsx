"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---- Configuration ----
const WS_URL = process.env.NEXT_PUBLIC_WS_URL;
const SCAN_TRIGGER_PHRASES = ["scan this", "scan it", "capture this", "take a picture", "click a picture", "scan", "capture", "click", "see this", "see", "look at this", "look"];
const NEW_SESSION_PHRASES = ["change the session", "start a new session", "new session", "reset session", "start over", "restart session","restart","restart the session"];
const STOP_PHRASES = ["stop", "stop talking", "be quiet", "shut up", "pause"];
const FILLER_MESSAGE = "Processing... please wait a little.";
const TTS_RATE = 1.35; // faster speech
const TTS_PITCH = 1.1;

type ClientState = "idle" | "listening" | "capturing" | "processing" | "speaking" | "error";

export default function VoiceClientPage() {
  const [state, setState] = useState<ClientState>("idle");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const isSpeakingRef = useRef(false); // true ONLY while the real final answer is being spoken
  const micEnabledRef = useRef(true); // mic is always meant to be on
  const sessionIdRef = useRef<string | null>(null);
  const femaleVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Lifecycle guards for the Web Speech API, which throws/errors if you
  // call start() while already running, or start() immediately after stop()
  // before the browser has actually finished tearing it down.
  const isRecognitionActiveRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ---- Request correlation (no request_id from the backend, so we infer it) ----
  // The server processes one full request/response cycle (status -> tokens ->
  // final) per incoming message before reading the next one, so responses
  // arrive in the same order requests were sent. We track that order
  // ourselves and only ever display/speak the response matching the LATEST
  // request the user actually asked for — anything older is discarded.
  const requestSeqRef = useRef(0);
  const pendingRequestQueueRef = useRef<number[]>([]); // FIFO of dispatched request ids awaiting their "final"
  const latestRequestIdRef = useRef<number | null>(null); // id we still care about; null = don't care about any pending answer

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  function isRequestStillRelevant(requestId: number | undefined): boolean {
    return requestId !== undefined && latestRequestIdRef.current !== null && requestId === latestRequestIdRef.current;
  }

  // ---------- 1. Camera: always-on preview ----------
  useEffect(() => {
    let stream: MediaStream | null = null;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setErrorMessage("Camera access denied or unavailable.");
        setState("error");
      }
    }

    startCamera();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ---------- 2. WebSocket: persistent connection ----------
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    };

    ws.onerror = () => {
      setErrorMessage("WebSocket connection error.");
      setState("error");
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };

    return () => {
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleServerMessage(msg: any) {
    switch (msg.type) {
      case "session":
        setSessionId(msg.session_id);
        break;

      case "status": {
        // Only reflect status in the UI if it belongs to the request we still care about.
        const frontId = pendingRequestQueueRef.current[0];
        if (isRequestStillRelevant(frontId)) {
          setState("processing");
        }
        break;
      }

      case "token": {
        const frontId = pendingRequestQueueRef.current[0];
        if (isRequestStillRelevant(frontId)) {
          setAnswer((prev) => prev + msg.text);
        }
        // else: tokens belong to a superseded request — silently discard
        break;
      }

      case "final": {
        const finishedId = pendingRequestQueueRef.current.shift();
        setSessionId(msg.session_id);

        if (isRequestStillRelevant(finishedId)) {
          // Overwrite with the authoritative full text (covers any dropped tokens)
          setAnswer(msg.answer);
          speakFinalAnswer(msg.answer);
        } else {
          // Stale response — the user already asked something else. Discard
          // it entirely: don't display it, don't speak it. If the request we
          // DO care about is still pending, stay in "processing".
          if (pendingRequestQueueRef.current.length > 0) {
            setState("processing");
          }
        }
        break;
      }

      case "error":
        setErrorMessage(msg.message);
        setState("error");
        break;

      default:
        break;
    }
  }

  // ---------- 3. Pick a female voice for TTS once voices are loaded ----------
  useEffect(() => {
    function pickFemaleVoice() {
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) return;

      const femaleByName = voices.find((v) =>
        /female|zira|susan|samantha|victoria|karen|moira|tessa|fiona|google us english/i.test(v.name)
      );
      femaleVoiceRef.current = femaleByName || voices.find((v) => v.lang.startsWith("en")) || voices[0];
    }

    pickFemaleVoice();
    window.speechSynthesis.onvoiceschanged = pickFemaleVoice;
  }, []);

  // ---------- 4. Speech recognition: continuous listening, auto-started ----------
  // Recognition is kept ALIVE at all times — including while the assistant
  // speaks the final answer — so a "stop" command can always interrupt it.
  // While isSpeakingRef is true (i.e. the REAL final answer is playing), we
  // ignore every transcript except an exact "stop" match, so the mic picking
  // up the assistant's own voice can't misfire as a new query. During
  // "processing" (including while the short filler cue plays) isSpeakingRef
  // stays false, so normal queries are accepted immediately — this is what
  // lets the user interrupt/replace an in-flight request with a new one.
  const startRecognition = useCallback(() => {
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }

    if (isRecognitionActiveRef.current) {
      return; // already running, don't double-start (causes InvalidStateError)
    }

    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      setErrorMessage("SpeechRecognition not supported in this browser. Use Chrome.");
      setState("error");
      return;
    }

    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isRecognitionActiveRef.current = true;
    };

    recognition.onresult = (event: any) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += text;
        } else {
          interim += text;
        }
      }

      const cleanedFinal = final.trim().toLowerCase();
      const saidStop = cleanedFinal && STOP_PHRASES.some((phrase) => cleanedFinal.includes(phrase));

      // While the real final answer is playing, only react to "stop" —
      // everything else (including likely echo of the assistant's own
      // voice) is ignored so it can't be misread as a new query.
      if (isSpeakingRef.current) {
        if (saidStop) {
          handleStopCommand();
        }
        return;
      }

      setTranscript(interim || final);

      if (final.trim()) {
        handleFinalTranscript(final.trim(), saidStop);
      }
    };

    recognition.onerror = (event: any) => {
      // "aborted" fires whenever recognition is stopped (e.g. on unmount) —
      // expected, not a real error. "no-speech" just means silence.
      if (event.error === "aborted" || event.error === "no-speech") {
        return;
      }
      console.warn("Speech recognition error:", event.error);
    };

    recognition.onend = () => {
      isRecognitionActiveRef.current = false;

      // Chrome periodically ends long-running recognition sessions on its
      // own; restart automatically as long as the mic is meant to be on.
      if (micEnabledRef.current) {
        restartTimeoutRef.current = setTimeout(() => {
          startRecognition();
        }, 250);
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
      setState((prev) => (prev === "speaking" ? prev : "listening"));
    } catch {
      // Already starting/started — ignore, onstart/onend will settle it.
    }
  }, []);

  // Auto-start listening as soon as the component mounts — no button needed.
  useEffect(() => {
    startRecognition();
    return () => {
      micEnabledRef.current = false;
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      try {
        recognitionRef.current?.stop();
      } catch {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- 5. Handle a finished utterance ----------
  function handleFinalTranscript(text: string, saidStop: boolean) {
    setTranscript("");

    // "stop" must NEVER be sent to the backend as a query, whether or not
    // the assistant is currently speaking.
    if (saidStop) {
      handleStopCommand();
      return;
    }

    const lowerText = text.toLowerCase();

    const matchedNewSession = NEW_SESSION_PHRASES.find((phrase) => lowerText.includes(phrase));
    if (matchedNewSession) {
      startNewSession();
      return;
    }

    const matchedScanTrigger = SCAN_TRIGGER_PHRASES.find((phrase) => lowerText.includes(phrase));
    if (matchedScanTrigger) {
      const remainingText = lowerText.replace(matchedScanTrigger, "").trim();
      captureFrameAndSend(remainingText || null);
    } else {
      sendQuery(text, null);
    }
  }

  // ---------- 6. Frame capture from the always-on video ----------
  function captureFrame(): string | null {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  function captureFrameAndSend(text: string | null) {
    setState("capturing");
    const imageBase64 = captureFrame();
    if (!imageBase64) {
      setErrorMessage("Could not capture a frame from the camera.");
      setState("error");
      return;
    }
    sendQuery(text, imageBase64);
  }

  // ---------- 7. Send query over WebSocket ----------
  function sendQuery(text: string | null, imageBase64: string | null) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setErrorMessage("WebSocket is not connected.");
      setState("error");
      return;
    }

    // A new query always supersedes whatever the assistant was doing/saying —
    // cut off any current speech (filler or a previous final answer) immediately.
    window.speechSynthesis.cancel();
    isSpeakingRef.current = false;

    const requestId = ++requestSeqRef.current;
    pendingRequestQueueRef.current.push(requestId);
    latestRequestIdRef.current = requestId;

    setAnswer(""); // discard whatever was shown for a previous, now-superseded request
    setState("processing");

    ws.send(
      JSON.stringify({
        session_id: sessionIdRef.current,
        text: text,
        image_base64: imageBase64,
      })
    );

    speakFillerCue();
  }

  // ---------- 8a. Filler cue while a request is in flight (does NOT gate the mic) ----------
  function speakFillerCue() {
    const utterance = new SpeechSynthesisUtterance(FILLER_MESSAGE);
    if (femaleVoiceRef.current) {
      utterance.voice = femaleVoiceRef.current;
    }
    utterance.pitch = TTS_PITCH;
    utterance.rate = TTS_RATE;
    // Deliberately do NOT set isSpeakingRef here — the mic must keep
    // listening normally (not stop-only) while this plays and while the
    // request is processing, so the user can ask something else right away.
    window.speechSynthesis.speak(utterance);
  }

  // ---------- 8b. Text-to-speech playback of the REAL final answer (gates the mic to stop-only) ----------
  function speakFinalAnswer(text: string) {
    if (!text) {
      setState("listening");
      return;
    }

    window.speechSynthesis.cancel(); // clear the filler cue or any leftover speech
    isSpeakingRef.current = true;
    setState("speaking");

    const utterance = new SpeechSynthesisUtterance(text);
    if (femaleVoiceRef.current) {
      utterance.voice = femaleVoiceRef.current;
    }
    utterance.pitch = TTS_PITCH;
    utterance.rate = TTS_RATE;

    utterance.onend = () => {
      isSpeakingRef.current = false;
      setState("listening");
    };
    utterance.onerror = () => {
      isSpeakingRef.current = false;
      setState("listening");
    };

    window.speechSynthesis.speak(utterance);
  }

  // ---------- 9. "Stop" — never sent to backend; halts speech and invalidates any pending answer ----------
  function handleStopCommand() {
    window.speechSynthesis.cancel();
    isSpeakingRef.current = false;
    setTranscript("");
    // Whatever request(s) are still in flight, we no longer care about their
    // answers — when their "final" eventually arrives it will be discarded.
    latestRequestIdRef.current = null;
    setState("listening");
  }

  // ---------- 10. Voice-triggered new session ----------
  function startNewSession() {
    setSessionId(null);
    sessionIdRef.current = null;
    setAnswer("");
    setTranscript("");
    latestRequestIdRef.current = null; // discard any answer still in flight from the old session
    window.speechSynthesis.cancel();
    isSpeakingRef.current = false;
    speakFinalAnswer("Starting a new session.");
  }

  return (
    <div className="container" suppressHydrationWarning>
      <h2>Shopping Assistant</h2>

      <div className="video-wrap" suppressHydrationWarning>
        <video ref={videoRef} autoPlay playsInline muted />
      </div>

      <div className="status-bar" suppressHydrationWarning>
        <span className={`dot ${state}`} suppressHydrationWarning />
        <span suppressHydrationWarning>{state.toUpperCase()}</span>
      </div>

      <div className="panel" suppressHydrationWarning>
        <div className="label" suppressHydrationWarning>
          Live transcript
        </div>
        <div className="transcript-text" suppressHydrationWarning>
          {transcript || "…"}
        </div>
      </div>

      <div className="panel" suppressHydrationWarning>
        <div className="label" suppressHydrationWarning>
          Assistant answer
        </div>
        <div className="answer-text" suppressHydrationWarning>
          {answer || "…"}
        </div>
      </div>

      {errorMessage && (
        <div className="panel" style={{ borderColor: "#e74c3c" }} suppressHydrationWarning>
          <div className="label" suppressHydrationWarning>
            Error
          </div>
          <div suppressHydrationWarning>{errorMessage}</div>
        </div>
      )}

      <div className="session-id" suppressHydrationWarning>
        Session ID: {sessionId || "(not yet assigned)"}
      </div>
      <p style={{ fontSize: 12, color: "#6b7280", marginTop: 14 }} suppressHydrationWarning>
        Just speak — the system is always listening. Say <b>"scan this"</b> (optionally with a
        question) to capture the camera frame, <b>"stop"</b> to interrupt the assistant mid-answer,
        or <b>"start a new session"</b> to reset the conversation. You can ask a new question at any
        time, even while a previous one is still being answered. Use Chrome — SpeechRecognition is
        not supported in Firefox/Safari.
      </p>
    </div>
  );
}