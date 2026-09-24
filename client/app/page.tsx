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

// Words in the filler cue. Used to recognise the phone's mic hearing the
// filler cue through its own speaker (echo) so it isn't sent as a query.
const FILLER_WORDS = new Set(
  FILLER_MESSAGE.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(Boolean)
);

type ClientState = "idle" | "listening" | "capturing" | "processing" | "speaking" | "error";

// ---- Helpers (no React state, safe to call from anywhere on the client) ----
function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  return (
    /Android|iPhone|iPad|iPod|Mobile/i.test(ua) ||
    // iPadOS reports itself as a Mac but has a touch screen
    (/Macintosh/i.test(ua) && navigator.maxTouchPoints > 1)
  );
}

function isFillerEcho(text: string): boolean {
  const words = text.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(Boolean);
  return words.length > 0 && words.every((w) => FILLER_WORDS.has(w));
}

// Mobile (and long-utterance) TTS is much more reliable with short chunks:
// Chrome silently stops long utterances, and Android often never fires onend.
function splitIntoChunks(text: string, maxLen = 180): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  const sentences = clean.match(/[^.!?]+[.!?]*\s*/g) || [clean];
  const pieces: string[] = [];
  for (const s of sentences) {
    if (s.length > maxLen) {
      // break very long sentences on commas / semicolons / colons
      pieces.push(...(s.match(/[^,;:]+[,;:]?\s*/g) || [s]));
    } else {
      pieces.push(s);
    }
  }
  const chunks: string[] = [];
  let current = "";
  for (const p of pieces) {
    if (current && (current + p).length > maxLen) {
      chunks.push(current.trim());
      current = p;
    } else {
      current += p;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

export default function VoiceClientPage() {
  const [state, setState] = useState<ClientState>("idle");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Mobile browsers only allow the mic + speech synthesis after a user tap,
  // so nothing voice-related starts until the user presses "Tap to start".
  const [started, setStarted] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const isSpeakingRef = useRef(false); // true ONLY while the real final answer is being spoken
  const micEnabledRef = useRef(false); // mic is meant to be on (set true once the user taps start)
  const startedRef = useRef(false);
  const sessionIdRef = useRef<string | null>(null);
  const femaleVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Lifecycle guards for the Web Speech API, which throws/errors if you
  // call start() while already running, or start() immediately after stop()
  // before the browser has actually finished tearing it down.
  const isRecognitionActiveRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastRecognitionErrorRef = useRef<string | null>(null);
  // On phones the mic is paused while the final answer is spoken, so the
  // phone's speaker isn't picked up by its own microphone.
  const recognitionPausedRef = useRef(false);
  // Guards against Android re-firing the same final result twice.
  const lastFinalRef = useRef<{ text: string; time: number }>({ text: "", time: 0 });

  // TTS bookkeeping. speechTokenRef is bumped on every cancel so callbacks
  // from old utterances can tell they are stale and do nothing.
  const speechTokenRef = useRef(0);
  const speechWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null); // keep a reference so Chrome doesn't garbage-collect it mid-speech
  const fillerEchoUntilRef = useRef(0); // until this time, filler-cue echo is ignored

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
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false,
        });
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
  function pickFemaleVoice() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return;

    const femaleByName = voices.find((v) =>
      /female|zira|susan|samantha|victoria|karen|moira|tessa|fiona|google us english/i.test(v.name)
    );
    femaleVoiceRef.current = femaleByName || voices.find((v) => v.lang.startsWith("en")) || voices[0];
  }

  useEffect(() => {
    pickFemaleVoice();
    window.speechSynthesis.onvoiceschanged = pickFemaleVoice;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- 3b. TTS helpers ----------
  function makeUtterance(text: string): SpeechSynthesisUtterance {
    // Android often loads its voice list late (or never fires onvoiceschanged),
    // so retry picking a voice at speak time.
    if (!femaleVoiceRef.current) pickFemaleVoice();

    const utterance = new SpeechSynthesisUtterance(text);
    if (femaleVoiceRef.current) {
      utterance.voice = femaleVoiceRef.current;
      // A voice whose language differs from utterance.lang can be silent on Android.
      utterance.lang = femaleVoiceRef.current.lang.replace("_", "-");
    } else {
      utterance.lang = "en-US";
    }
    utterance.pitch = TTS_PITCH;
    utterance.rate = TTS_RATE;
    return utterance;
  }

  function clearSpeechWatchdog() {
    if (speechWatchdogRef.current) {
      clearTimeout(speechWatchdogRef.current);
      speechWatchdogRef.current = null;
    }
  }

  // Stops all speech and invalidates every pending TTS callback.
  function cancelSpeech() {
    speechTokenRef.current += 1;
    clearSpeechWatchdog();
    fillerEchoUntilRef.current = 0;
    window.speechSynthesis.cancel();
  }

  // Called from the "Tap to start" button. Speech synthesis stays silent on
  // mobile until it has been triggered from a user gesture at least once.
  function unlockSpeech() {
    try {
      const u = new SpeechSynthesisUtterance(" ");
      u.volume = 0;
      window.speechSynthesis.speak(u);
      window.speechSynthesis.resume();
    } catch {
      /* ignore */
    }
  }

  // ---------- 3c. Pause / resume the mic (used on phones while speaking) ----------
  function pauseRecognition() {
    recognitionPausedRef.current = true;
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    try {
      recognitionRef.current?.abort();
    } catch {
      /* ignore */
    }
  }

  function resumeRecognition() {
    recognitionPausedRef.current = false;
    if (!micEnabledRef.current) return;
    if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
    // Small delay so the tail of the assistant's voice isn't heard by the mic.
    restartTimeoutRef.current = setTimeout(() => {
      startRecognition();
    }, 400);
  }

  // ---------- 4. Speech recognition: continuous listening ----------
  // Recognition is kept ALIVE at all times so it is always ready for the next
  // query. On desktop it also stays on while the assistant speaks the final
  // answer, so a spoken "stop" can interrupt it; while isSpeakingRef is true
  // every transcript except "stop" is ignored so the mic picking up the
  // assistant's own voice can't misfire as a new query.
  //
  // On phones the mic is paused while the final answer is spoken (the speaker
  // is right next to the mic, so it hears itself, and Android also mutes or
  // cuts off TTS while recognition is running). The on-screen Stop button
  // replaces the voice "stop" there.
  //
  // During "processing" (including while the short filler cue plays)
  // isSpeakingRef stays false, so normal queries are accepted immediately —
  // this is what lets the user interrupt/replace an in-flight request.
  const startRecognition = useCallback(() => {
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }

    if (isRecognitionActiveRef.current) {
      return; // already running, don't double-start (causes InvalidStateError)
    }

    if (!micEnabledRef.current || recognitionPausedRef.current) {
      return;
    }

    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      setErrorMessage("SpeechRecognition not supported in this browser. Use Chrome.");
      setState("error");
      micEnabledRef.current = false;
      startedRef.current = false;
      setStarted(false);
      return;
    }

    const recognition = new SpeechRecognitionCtor();
    // Android Chrome's continuous mode repeats/duplicates text and often gives
    // no interim results, so phones use one-utterance sessions that we restart
    // in onend. Desktop keeps continuous mode.
    recognition.continuous = !isMobileDevice();
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isRecognitionActiveRef.current = true;
    };

    recognition.onresult = (event: any) => {
      if (recognitionRef.current !== recognition) return; // stale instance

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

      const heard = (interim || final).trim();

      // The phone's mic can hear the "Processing... please wait" cue coming out
      // of its own speaker. Ignore that echo so it doesn't replace the real query.
      if (heard && Date.now() < fillerEchoUntilRef.current && isFillerEcho(heard)) {
        return;
      }

      setTranscript(interim || final);

      if (final.trim()) {
        // Android can deliver the same final result twice — drop the repeat.
        const now = Date.now();
        if (cleanedFinal === lastFinalRef.current.text && now - lastFinalRef.current.time < 2000) {
          return;
        }
        lastFinalRef.current = { text: cleanedFinal, time: now };

        handleFinalTranscript(final.trim(), saidStop);
      }
    };

    recognition.onerror = (event: any) => {
      lastRecognitionErrorRef.current = event.error;

      // "aborted" fires whenever recognition is stopped (e.g. on unmount or
      // when we pause the mic) — expected, not a real error. "no-speech" just
      // means silence.
      if (event.error === "aborted" || event.error === "no-speech") {
        return;
      }

      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        // Permission denied / blocked: retrying in a loop can never work.
        micEnabledRef.current = false;
        startedRef.current = false;
        setStarted(false);
        setErrorMessage(
          "Microphone access is blocked. Allow the microphone for this site in your browser settings (phones also need the page to be served over HTTPS), then tap Start again."
        );
        setState("error");
        return;
      }

      if (event.error === "audio-capture") {
        setErrorMessage("Microphone not available. Close other apps using the mic and try again.");
        return;
      }

      console.warn("Speech recognition error:", event.error);
    };

    recognition.onend = () => {
      if (recognitionRef.current !== recognition) return; // stale instance
      isRecognitionActiveRef.current = false;

      // Chrome (and every phone utterance in non-continuous mode) ends the
      // session on its own; restart automatically as long as the mic is
      // meant to be on and isn't deliberately paused.
      if (micEnabledRef.current && !recognitionPausedRef.current) {
        const delay = lastRecognitionErrorRef.current === "network" ? 1500 : 300;
        lastRecognitionErrorRef.current = null;
        restartTimeoutRef.current = setTimeout(() => {
          startRecognition();
        }, delay);
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
      // Don't overwrite "processing"/"speaking" every time the mic restarts.
      setState((prev) => (prev === "idle" ? "listening" : prev));
    } catch {
      // start() can throw InvalidStateError right after a stop on mobile —
      // try again shortly instead of leaving the mic dead.
      if (micEnabledRef.current && !recognitionPausedRef.current) {
        restartTimeoutRef.current = setTimeout(() => {
          startRecognition();
        }, 500);
      }
    }
  }, []);

  // ---------- 4b. Tap-to-start (required by mobile browsers) ----------
  function handleStart() {
    unlockSpeech();
    micEnabledRef.current = true;
    startedRef.current = true;
    recognitionPausedRef.current = false;
    setErrorMessage(null);
    setState("listening");
    setStarted(true);
    startRecognition();
  }

  // Clean up the mic on unmount. (Recognition itself is started by the
  // "Tap to start" button, not automatically.)
  useEffect(() => {
    return () => {
      micEnabledRef.current = false;
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      clearSpeechWatchdog();
      try {
        recognitionRef.current?.stop();
      } catch {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Phones kill recognition when the tab is backgrounded or the screen locks;
  // bring the mic back when the user returns.
  useEffect(() => {
    function onVisibilityChange() {
      if (
        document.visibilityState === "visible" &&
        startedRef.current &&
        micEnabledRef.current &&
        !recognitionPausedRef.current
      ) {
        startRecognition();
      }
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [startRecognition]);

  // ---------- 5. Handle a finished utterance ----------
  function handleFinalTranscript(text: string, saidStop: boolean) {
    // "stop" must NEVER be sent to the backend as a query, whether or not
    // the assistant is currently speaking.
    if (saidStop) {
      handleStopCommand();
      return;
    }

    // Keep what the user said on screen (it used to be cleared in the same
    // instant it was set, so on phones — which often deliver only a final
    // result and no interim ones — it was never visible).
    setTranscript(text);

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
    cancelSpeech();
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
    const token = speechTokenRef.current;
    fillerEchoUntilRef.current = Date.now() + 6000;

    // Deliberately do NOT set isSpeakingRef here — the mic must keep
    // listening normally (not stop-only) while this plays and while the
    // request is processing, so the user can ask something else right away.
    //
    // The short delay after cancel() matters: on Chrome/Android, speak()
    // called in the same tick as cancel() is frequently dropped.
    setTimeout(() => {
      if (token !== speechTokenRef.current) return; // superseded or cancelled
      const utterance = makeUtterance(FILLER_MESSAGE);
      utteranceRef.current = utterance;
      utterance.onend = () => {
        fillerEchoUntilRef.current = Date.now() + 1200; // tail of the echo
      };
      window.speechSynthesis.speak(utterance);
    }, 120);
  }

  // ---------- 8b. Text-to-speech playback of the REAL final answer (gates the mic to stop-only) ----------
  function speakFinalAnswer(text: string) {
    if (!text) {
      setState("listening");
      return;
    }

    cancelSpeech(); // clear the filler cue or any leftover speech
    const token = speechTokenRef.current;
    isSpeakingRef.current = true;
    setState("speaking");

    const mobile = isMobileDevice();
    if (mobile) pauseRecognition();

    const chunks = splitIntoChunks(text);
    let index = 0;

    const finish = () => {
      if (token !== speechTokenRef.current) return;
      clearSpeechWatchdog();
      isSpeakingRef.current = false;
      setState("listening");
      if (mobile) resumeRecognition();
    };

    const speakNext = () => {
      if (token !== speechTokenRef.current) return;
      clearSpeechWatchdog();
      if (index >= chunks.length) {
        finish();
        return;
      }

      const chunk = chunks[index++];
      const utterance = makeUtterance(chunk);
      utteranceRef.current = utterance;

      let advanced = false;
      const advance = () => {
        if (advanced) return;
        advanced = true;
        speakNext();
      };

      utterance.onend = advance;
      utterance.onerror = (e) => {
        if (token !== speechTokenRef.current) return;
        // "interrupted"/"canceled" come from our own cancel() calls.
        if (e.error === "interrupted" || e.error === "canceled") return;
        console.warn("Speech synthesis error:", e.error);
        finish();
      };

      // Android sometimes never fires onend; don't let the UI (and the paused
      // mic) hang forever waiting for it.
      speechWatchdogRef.current = setTimeout(() => {
        if (token !== speechTokenRef.current) return;
        window.speechSynthesis.cancel();
        advance();
      }, 4000 + chunk.length * 100);

      window.speechSynthesis.speak(utterance);
    };

    // Short delay after cancel() — see note in speakFillerCue.
    setTimeout(speakNext, 120);
  }

  // ---------- 9. "Stop" — never sent to backend; halts speech and invalidates any pending answer ----------
  // Triggered by voice ("stop") or by the on-screen Stop button.
  function handleStopCommand() {
    cancelSpeech();
    isSpeakingRef.current = false;
    setTranscript("");
    // Whatever request(s) are still in flight, we no longer care about their
    // answers — when their "final" eventually arrives it will be discarded.
    latestRequestIdRef.current = null;
    setState("listening");
    if (recognitionPausedRef.current) {
      resumeRecognition(); // phones: bring the mic back
    }
  }

  // ---------- 10. Voice-triggered new session ----------
  function startNewSession() {
    setSessionId(null);
    sessionIdRef.current = null;
    setAnswer("");
    setTranscript("");
    latestRequestIdRef.current = null; // discard any answer still in flight from the old session
    cancelSpeech();
    isSpeakingRef.current = false;
    speakFinalAnswer("Starting a new session.");
  }

  return (
    <div className="container" suppressHydrationWarning>
      <h2>Shopping Assistant</h2>

      <div className="video-wrap" suppressHydrationWarning>
        <video ref={videoRef} autoPlay playsInline muted />
      </div>

      {!started && (
        <button
          onClick={handleStart}
          style={{
            width: "100%",
            padding: "14px 16px",
            marginTop: 12,
            fontSize: 16,
            fontWeight: 600,
            borderRadius: 10,
            border: "none",
            background: "#2563eb",
            color: "#fff",
            cursor: "pointer",
          }}
          suppressHydrationWarning
        >
          🎤 Tap to start
        </button>
      )}

      <div className="status-bar" suppressHydrationWarning>
        <span className={`dot ${state}`} suppressHydrationWarning />
        <span suppressHydrationWarning>{state.toUpperCase()}</span>
      </div>

      {(state === "speaking" || state === "processing") && (
        <button
          onClick={handleStopCommand}
          style={{
            padding: "10px 16px",
            marginBottom: 8,
            fontSize: 14,
            fontWeight: 600,
            borderRadius: 8,
            border: "1px solid #e74c3c",
            background: "transparent",
            color: "#e74c3c",
            cursor: "pointer",
          }}
          suppressHydrationWarning
        >
          ⏹ Stop
        </button>
      )}

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
        Tap <b>Start</b> once, then just speak — the system keeps listening. Say <b>"scan this"</b>{" "}
        (optionally with a question) to capture the camera frame, <b>"stop"</b> (or tap the Stop
        button) to interrupt the assistant mid-answer, or <b>"start a new session"</b> to reset the
        conversation. On phones the microphone pauses while the assistant is speaking, so use the
        Stop button to interrupt. You can ask a new question at any time while a previous one is
        still being processed. Use Chrome — SpeechRecognition is not supported in Firefox/Safari.
      </p>
    </div>
  );
}