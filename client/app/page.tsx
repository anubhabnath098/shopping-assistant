"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---- Configuration ----
const WS_URL = process.env.NEXT_PUBLIC_WS_URL;

const SCAN_TRIGGER_PHRASES = ["scan this", "scan it", "capture this", "take a picture", "click a picture", "scan", "capture", "click", "see this", "see", "look at this", "look"];
const NEW_SESSION_PHRASES = ["change the session", "start a new session", "new session", "reset session", "start over", "restart session", "restart", "restart the session"];
const STOP_PHRASES = ["stop", "stop talking", "be quiet", "shut up", "pause"];
// Lets the user replay the image result later if they declined the first offer.
const REPLAY_IMAGE_PHRASES = ["read the product details", "read product details", "read the details", "read the image result"];

const FILLER_MESSAGE = "Processing... please wait a little.";
const IMAGE_OFFER_MESSAGE = "Product details have been fetched. Would you like me to read it?";
// The text LLM is fast, so the filler cue is only spoken for image requests.
// Set to true to also speak it for text requests.
const FILLER_FOR_TEXT = false;

const YES_REGEX = /\b(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|read it|read)\b/i;
const NO_REGEX = /\b(no|nope|nah|not now|don't|do not|skip|later|cancel|never mind)\b/i;

const TTS_RATE = 1.15; // faster speech
const TTS_PITCH = 1.1;

// Words in the assistant's own cues. Used to recognise the mic hearing the
// cue through its own speaker (echo) so it isn't treated as a query / answer.
const wordsOf = (s: string) => s.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(Boolean);
const ECHO_WORDS = new Set([...wordsOf(FILLER_MESSAGE), ...wordsOf(IMAGE_OFFER_MESSAGE)]);

type ClientState = "idle" | "listening" | "capturing" | "processing" | "speaking" | "error";
type RequestKind = "text" | "image";
type SpeechKind = "text" | "image" | "offer" | "system";

interface RequestMeta {
  kind: RequestKind;
  epoch: number; // session epoch the request was sent in (bumped on "new session")
  discarded: boolean; // true = don't display / speak the answer when it arrives
}

// A reading that the user paused by tapping the right half of the camera.
interface PausedSpeech {
  text: string;
  kind: SpeechKind;
  index: number; // chunk that was being spoken when paused
}

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

function isEcho(text: string): boolean {
  const words = wordsOf(text);
  return words.length > 0 && words.every((w) => ECHO_WORDS.has(w));
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
  const [textAnswer, setTextAnswer] = useState("");
  const [imageAnswer, setImageAnswer] = useState("");
  const [imageBusy, setImageBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [speechPaused, setSpeechPaused] = useState(false); // UI only: a reading is paused via the right half of the camera
  // Mobile browsers only allow the mic + speech synthesis after a user tap,
  // so nothing voice-related starts until the user presses "Tap to start".
  const [started, setStarted] = useState(false);

  // IMPORTANT: startRecognition is memoized once (useCallback with []), so every
  // function it can reach (handleFinalTranscript, handleStopCommand, ...) is the
  // FIRST render's closure. They must therefore only read REFS, never React
  // state values. State setters are fine.

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const isSpeakingRef = useRef(false); // true ONLY while a real answer / offer is being spoken (mic = stop-only)
  const speakingKindRef = useRef<SpeechKind | null>(null);
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
  // On phones the mic is paused while an answer is spoken, so the
  // phone's speaker isn't picked up by its own microphone.
  const recognitionPausedRef = useRef(false);
  // Guards against Android re-firing the same final result twice.
  const lastFinalRef = useRef<{ text: string; time: number }>({ text: "", time: 0 });

  // TTS bookkeeping. speechTokenRef is bumped on every cancel so callbacks
  // from old utterances can tell they are stale and do nothing.
  const speechTokenRef = useRef(0);
  const speechWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null); // keep a reference so Chrome doesn't garbage-collect it mid-speech
  const echoUntilRef = useRef(0); // until this time, echo of our own cues is ignored
  const noticeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The item currently being spoken, so it can be paused and resumed.
  const activeSpeechRef = useRef<{ text: string; kind: SpeechKind; chunks: string[]; index: number } | null>(null);
  // Set while the user has paused the reading (right half of the camera).
  const pausedSpeechRef = useRef<PausedSpeech | null>(null);

  // ---- Request tracking (server echoes request_id + kind on every message) ----
  const requestSeqRef = useRef(0);
  const requestsRef = useRef<Map<string, RequestMeta>>(new Map());
  // Locks: at most ONE text request and ONE image request in flight at a time.
  const textRequestIdRef = useRef<string | null>(null);
  const imageRequestIdRef = useRef<string | null>(null);
  // Bumped on "new session" so answers from the old session are dropped.
  const sessionEpochRef = useRef(0);

  // ---- Image result hand-off ----
  const imageAnswerRef = useRef(""); // latest image answer (for reading / replay)
  const imageOfferDueRef = useRef(false); // an image result is waiting to be offered
  const offerActiveRef = useRef(false); // "would you like me to read it?" asked, awaiting yes/no

  // ---------- UI helpers ----------
  function showNotice(msg: string) {
    setNotice(msg);
    if (noticeTimeoutRef.current) clearTimeout(noticeTimeoutRef.current);
    noticeTimeoutRef.current = setTimeout(() => setNotice(null), 4000);
  }

  function updateSession(id: string) {
    sessionIdRef.current = id;
    setSessionId(id);
  }

  // Derives the main status from refs. The image request never affects it:
  // while only an image is in flight the user is free to keep talking.
  function refreshIdleState() {
    if (!startedRef.current) return;
    if (isSpeakingRef.current) setState("speaking");
    else if (textRequestIdRef.current) setState("processing");
    else setState("listening");
  }

  // ---------- Camera halves: LEFT = stop, RIGHT = pause / resume ----------
  function setPausedSpeech(p: PausedSpeech | null) {
    pausedSpeechRef.current = p;
    setSpeechPaused(!!p);
  }

  // Left half of the camera: same as saying "stop".
  function handleStopTap() {
    if (!startedRef.current) return;
    handleStopCommand();
  }

  // Right half of the camera: reading -> pause + listen, paused -> resume where it left off.
  function handlePauseTap() {
    if (!startedRef.current) return;
    if (isSpeakingRef.current) {
      pauseSpeech();
    } else if (pausedSpeechRef.current) {
      resumeSpeech();
    }
  }

  function pauseSpeech() {
    const s = activeSpeechRef.current;
    if (!s) return;
    // Remember where we were BEFORE cancelSpeech() clears the active speech.
    setPausedSpeech({ text: s.text, kind: s.kind, index: s.index });
    cancelSpeech();
    isSpeakingRef.current = false;
    refreshIdleState(); // -> "listening" (or "processing" if a text request is in flight)
    // Phones: the mic was paused while speaking, bring it back now.
    if (recognitionPausedRef.current) resumeRecognition();
  }

  function resumeSpeech() {
    const p = pausedSpeechRef.current;
    if (!p) return;
    // Re-speak from the start of the sentence that was playing.
    // (speakText clears the paused state.)
    speakText(p.text, p.kind, p.index);
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
      if (wsRef.current !== ws) return;
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    };

    ws.onerror = () => {
      if (wsRef.current !== ws) return;
      setErrorMessage("WebSocket connection error.");
      setState("error");
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      console.log("WebSocket closed");
      // Nothing in flight can ever return now — release the locks so the UI isn't stuck.
      requestsRef.current.clear();
      textRequestIdRef.current = null;
      imageRequestIdRef.current = null;
      setImageBusy(false);
      setErrorMessage("Connection lost. Reload the page to reconnect.");
    };

    return () => {
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleServerMessage(msg: any) {
    const requestId: string | undefined = msg.request_id;
    const meta = requestId ? requestsRef.current.get(requestId) : undefined;
    // Only trust session ids coming from requests of the CURRENT session,
    // otherwise a late reply from an old session would revert a "new session".
    const sameEpoch = !!meta && meta.epoch === sessionEpochRef.current;

    switch (msg.type) {
      case "session":
        updateSession(msg.session_id);
        break;

      case "accepted":
        if (sameEpoch && msg.session_id) updateSession(msg.session_id);
        break;

      case "status":
        // Nothing to do: the busy state is driven by our own request locks.
        break;

      case "token":
        // Only text requests stream tokens to the UI (server suppresses image tokens anyway).
        if (
          meta &&
          meta.kind === "text" &&
          !meta.discarded &&
          requestId === textRequestIdRef.current
        ) {
          setTextAnswer((prev) => prev + msg.text);
        }
        break;

      case "final": {
        if (!meta || !requestId) break;
        requestsRef.current.delete(requestId);
        if (sameEpoch && msg.session_id) updateSession(msg.session_id);

        if (meta.kind === "text") {
          onTextFinal(msg.answer, meta.discarded);
        } else {
          onImageFinal(msg.answer, meta.discarded);
        }
        break;
      }

      case "error": {
        setErrorMessage(msg.message);
        if (meta && requestId) {
          requestsRef.current.delete(requestId);
          if (meta.kind === "text") {
            textRequestIdRef.current = null;
          } else {
            imageRequestIdRef.current = null;
            setImageBusy(false);
          }
          refreshIdleState();
          maybeOfferImageResult();
        }
        break;
      }

      default:
        break;
    }
  }

  function onTextFinal(answer: string | undefined, discarded: boolean) {
    textRequestIdRef.current = null; // text lock released — next text question is allowed

    if (discarded) {
      // User said "stop" (or started a new session) while it was processing.
      refreshIdleState();
      maybeOfferImageResult();
      return;
    }

    // Overwrite with the authoritative full text (covers any dropped tokens)
    setTextAnswer(answer || "");
    speakText(answer || "", "text"); // when finished -> maybeOfferImageResult()
  }

  function onImageFinal(answer: string | undefined, discarded: boolean) {
    imageRequestIdRef.current = null; // image lock released — next scan is allowed
    setImageBusy(false);

    if (discarded) return; // belonged to an old session

    setImageAnswer(answer || "");
    imageAnswerRef.current = answer || "";

    if (!answer) {
      showNotice("The image request returned no details.");
      return;
    }

    // Always ask before reading it. If a text conversation is still going on
    // (request in flight or answer being spoken) the offer is deferred until
    // that turn is over — it is triggered again from onSpeechFinished /
    // onTextFinal / handleStopCommand.
    imageOfferDueRef.current = true;
    maybeOfferImageResult();
  }

  // ---------- Offering the image result ----------
  function maybeOfferImageResult() {
    if (!imageOfferDueRef.current || !imageAnswerRef.current) return;
    if (!startedRef.current || !micEnabledRef.current) return;
    if (textRequestIdRef.current) return; // text turn still in flight
    if (isSpeakingRef.current) return; // something is being spoken
    if (pausedSpeechRef.current) return; // a paused reading is still unfinished
    if (offerActiveRef.current) return; // already asked

    imageOfferDueRef.current = false;
    offerActiveRef.current = true; // the next thing the user says is the answer
    speakText(IMAGE_OFFER_MESSAGE, "offer");
  }

  function readImageAnswer() {
    const answer = imageAnswerRef.current;
    if (!answer) {
      showNotice("There is no image result to read yet.");
      return;
    }
    if (textRequestIdRef.current) {
      showNotice("Wait for the current answer to finish first.");
      return;
    }
    imageOfferDueRef.current = false;
    offerActiveRef.current = false;
    speakText(answer, "image");
  }

  // Called whenever a spoken item finishes on its own (not when cancelled).
  function onSpeechFinished(kind: SpeechKind) {
    if (kind === "offer") return; // now waiting for the user's yes / no
    // A text answer (or anything else) is done -> the text conversation is idle,
    // so a waiting image result can be offered now.
    maybeOfferImageResult();
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
    echoUntilRef.current = 0;
    speakingKindRef.current = null;
    activeSpeechRef.current = null;
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
  // query. On desktop it also stays on while the assistant speaks, so a spoken
  // "stop" can interrupt it; while isSpeakingRef is true every transcript
  // except "stop" is ignored so the mic picking up the assistant's own voice
  // can't misfire as a new query.
  //
  // On phones the mic is paused while an answer is spoken (the speaker is right
  // next to the mic, and Android also mutes or cuts off TTS while recognition
  // is running). The left / right halves of the camera replace the voice "stop"
  // there.
  //
  // During "processing" and while an IMAGE is being analysed, isSpeakingRef
  // stays false, so the mic accepts normal input (subject to the one-text /
  // one-image locks).
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

      // While an answer / the image offer is playing, only react to "stop" —
      // everything else (including likely echo of the assistant's own
      // voice) is ignored so it can't be misread as a new query.
      if (isSpeakingRef.current) {
        if (saidStop) {
          handleStopCommand();
        }
        return;
      }

      const heard = (interim || final).trim();

      // The mic can hear the filler cue / the "would you like me to read it?"
      // prompt coming out of the speaker. Ignore that echo so it isn't taken
      // as a query or as a "yes".
      if (heard && Date.now() < echoUntilRef.current && isEcho(heard)) {
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

        handleFinalTranscript(final.trim(), !!saidStop);
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
      if (noticeTimeoutRef.current) clearTimeout(noticeTimeoutRef.current);
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

    // Keep what the user said on screen.
    setTranscript(text);

    const lowerText = text.toLowerCase();

    // (a) Answer to "Product details have been fetched. Would you like me to read it?"
    //     "yes" -> read the image result. Anything else -> offer declined, and
    //     the utterance carries on as a normal one (unless it was a plain "no").
    if (offerActiveRef.current) {
      offerActiveRef.current = false;
      if (NO_REGEX.test(lowerText)) {
        refreshIdleState();
        return;
      }
      if (YES_REGEX.test(lowerText)) {
        readImageAnswer();
        return;
      }
      // not a yes/no -> fall through and treat it as a normal query
    }

    // (b) Replay of an earlier image result the user declined.
    if (REPLAY_IMAGE_PHRASES.some((p) => lowerText.includes(p))) {
      readImageAnswer();
      return;
    }

    // (c) New session
    const matchedNewSession = NEW_SESSION_PHRASES.find((phrase) => lowerText.includes(phrase));
    if (matchedNewSession) {
      startNewSession();
      return;
    }

    // (d) Image query or (e) plain text query
    const matchedScanTrigger = SCAN_TRIGGER_PHRASES.find((phrase) => lowerText.includes(phrase));
    if (matchedScanTrigger) {
      const remainingText = lowerText.replace(matchedScanTrigger, "").trim();
      captureFrameAndSend(remainingText || null);
    } else {
      sendTextQuery(text);
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
    // Only one image at a time — wait until the previous image response returns.
    if (imageRequestIdRef.current) {
      showNotice("Still processing the previous image. You can keep asking questions meanwhile.");
      return;
    }

    setState("capturing");
    const imageBase64 = captureFrame();
    if (!imageBase64) {
      setErrorMessage("Could not capture a frame from the camera.");
      refreshIdleState();
      return;
    }
    dispatchRequest("image", text, imageBase64);
  }

  function sendTextQuery(text: string) {
    // Only one text question at a time — wait until its answer has come back.
    if (textRequestIdRef.current) {
      showNotice("Still answering your previous question. Please wait for it to finish.");
      return;
    }
    dispatchRequest("text", text, null);
  }

  // ---------- 7. Send a request over WebSocket ----------
  function dispatchRequest(kind: RequestKind, text: string | null, imageBase64: string | null) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setErrorMessage("WebSocket is not connected.");
      refreshIdleState();
      return;
    }

    // A new query supersedes any filler cue that may still be playing.
    cancelSpeech();
    isSpeakingRef.current = false;
    setErrorMessage(null);

    const requestId = `${kind}-${++requestSeqRef.current}-${Date.now().toString(36)}`;
    requestsRef.current.set(requestId, {
      kind,
      epoch: sessionEpochRef.current,
      discarded: false,
    });

    if (kind === "text") {
      textRequestIdRef.current = requestId;
      setPausedSpeech(null); // a new question replaces any paused reading
      setTextAnswer(""); // discard whatever was shown for the previous text answer
    } else {
      imageRequestIdRef.current = requestId;
      setImageBusy(true);
      setImageAnswer("");
      imageAnswerRef.current = "";
      imageOfferDueRef.current = false; // a new scan replaces any unread old result
    }

    ws.send(
      JSON.stringify({
        request_id: requestId,
        session_id: sessionIdRef.current,
        text: text,
        image_base64: imageBase64,
      })
    );

    refreshIdleState(); // -> "processing" only when a TEXT request is in flight

    if (kind === "image" || FILLER_FOR_TEXT) {
      speakFillerCue();
    }
  }

  // ---------- 8a. Filler cue while a request is in flight (does NOT gate the mic) ----------
  function speakFillerCue() {
    const token = speechTokenRef.current;
    echoUntilRef.current = Date.now() + 6000;

    // Deliberately do NOT set isSpeakingRef here — the mic must keep listening
    // normally (not stop-only) while this plays, so the user can ask something
    // else right away.
    //
    // The short delay after cancel() matters: on Chrome/Android, speak()
    // called in the same tick as cancel() is frequently dropped.
    setTimeout(() => {
      if (token !== speechTokenRef.current) return; // superseded or cancelled
      const utterance = makeUtterance(FILLER_MESSAGE);
      utteranceRef.current = utterance;
      utterance.onend = () => {
        echoUntilRef.current = Date.now() + 1200; // tail of the echo
      };
      window.speechSynthesis.speak(utterance);
    }, 120);
  }

  // ---------- 8b. Speak a real item (text answer / image answer / image offer / system msg) ----------
  // Gates the mic to stop-only while playing. When it finishes on its own,
  // onSpeechFinished(kind) decides what happens next.
  // startIndex lets a paused reading resume from the sentence it was on.
  function speakText(text: string, kind: SpeechKind, startIndex = 0) {
    setPausedSpeech(null); // any new speech replaces a paused one

    if (!text || !text.trim()) {
      cancelSpeech();
      isSpeakingRef.current = false;
      refreshIdleState();
      onSpeechFinished(kind);
      return;
    }

    cancelSpeech(); // clear the filler cue or any leftover speech
    const token = speechTokenRef.current;
    isSpeakingRef.current = true;
    speakingKindRef.current = kind;
    setState("speaking");

    const mobile = isMobileDevice();
    if (mobile) pauseRecognition();

    const chunks = splitIntoChunks(text);
    let index = Math.min(Math.max(startIndex, 0), Math.max(chunks.length - 1, 0));
    // Tracks the chunk currently being spoken so pauseSpeech() knows where to resume.
    const session = { text, kind, chunks, index };
    activeSpeechRef.current = session;

    const finish = () => {
      if (token !== speechTokenRef.current) return;
      clearSpeechWatchdog();
      isSpeakingRef.current = false;
      speakingKindRef.current = null;
      activeSpeechRef.current = null;
      // The mic may still "hear" the tail of the prompt — don't let it count as a yes.
      if (kind === "offer") echoUntilRef.current = Date.now() + 2000;
      refreshIdleState();
      if (mobile) resumeRecognition();
      onSpeechFinished(kind);
    };

    const speakNext = () => {
      if (token !== speechTokenRef.current) return;
      clearSpeechWatchdog();
      if (index >= chunks.length) {
        finish();
        return;
      }

      session.index = index; // the chunk about to be spoken
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

  // ---------- 9. "Stop" — never sent to backend ----------
  // Triggered by voice ("stop") or by tapping the LEFT half of the camera.
  // Stops speech and drops the pending TEXT answer. It does NOT touch the
  // image request: that keeps processing and its result is still offered.
  // Unlike the pause (right half), Stop also discards a paused reading (no resume).
  function handleStopCommand() {
    cancelSpeech();
    isSpeakingRef.current = false;
    setTranscript("");
    setPausedSpeech(null);

    // Stopping while the "read it?" prompt is playing / awaiting = "no thanks".
    offerActiveRef.current = false;

    // If a text request is still in flight, we no longer want its answer.
    // (The text lock stays until it returns, so two text requests never overlap.)
    const textId = textRequestIdRef.current;
    if (textId) {
      const meta = requestsRef.current.get(textId);
      if (meta) meta.discarded = true;
    }

    refreshIdleState();
    if (recognitionPausedRef.current) {
      resumeRecognition(); // phones: bring the mic back
    }

    // Even after a stopped text answer, a waiting image result must still be offered.
    maybeOfferImageResult();
  }

  // ---------- 10. Voice-triggered new session ----------
  function startNewSession() {
    sessionEpochRef.current += 1;
    updateSession("");
    sessionIdRef.current = null;
    setSessionId(null);
    setTextAnswer("");
    setImageAnswer("");
    imageAnswerRef.current = "";
    imageOfferDueRef.current = false;
    offerActiveRef.current = false;
    setPausedSpeech(null);
    setTranscript("");

    // Answers still in flight belong to the old session -> drop them when they
    // arrive. (Locks are released only when those replies actually return.)
    requestsRef.current.forEach((meta) => {
      meta.discarded = true;
    });

    cancelSpeech();
    isSpeakingRef.current = false;
    speakText("Starting a new session.", "system");
  }

  // Shared style for the two tap zones on top of the camera.
  const halfStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: "50%",
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "center",
    paddingBottom: 10,
    cursor: started ? "pointer" : "default",
    userSelect: "none",
    WebkitUserSelect: "none",
    WebkitTapHighlightColor: "transparent",
    touchAction: "manipulation",
  };

  const halfLabelStyle: React.CSSProperties = {
    padding: "6px 14px",
    borderRadius: 999,
    fontSize: 13,
    fontWeight: 600,
    color: "#fff",
    background: "rgba(0,0,0,0.55)",
    pointerEvents: "none",
  };

  return (
    <div className="container" suppressHydrationWarning>
      <h2>Shopping Assistant</h2>

      {/* Camera: nearly full width (small side margins), split into two tap halves.
          LEFT half = stop reading, RIGHT half = pause / resume reading. */}
      <div
        className="video-wrap"
        style={{
          position: "relative",
          width: "96vw",
          maxWidth: "none",
          marginLeft: "calc(50% - 48vw)",
          marginRight: 0,
          borderRadius: 12,
          overflow: "hidden",
          background: "#000",
        }}
        suppressHydrationWarning
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ display: "block", width: "100%", height: "auto" }}
        />

        {/* LEFT: stop */}
        <div
          role="button"
          aria-label="Stop reading"
          onClick={handleStopTap}
          style={{ ...halfStyle, left: 0 }}
          suppressHydrationWarning
        >
          {started && <span style={halfLabelStyle}>⏹ Stop</span>}
        </div>

        {/* RIGHT: pause / resume */}
        <div
          role="button"
          aria-label={speechPaused ? "Resume reading" : "Pause reading"}
          onClick={handlePauseTap}
          style={{ ...halfStyle, right: 0 }}
          suppressHydrationWarning
        >
          {started && (
            <span style={halfLabelStyle}>{speechPaused ? "▶ Resume" : "⏸ Pause"}</span>
          )}
        </div>

        {/* Thin divider between the halves */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: "50%",
            width: 1,
            background: "rgba(255,255,255,0.35)",
            pointerEvents: "none",
          }}
          suppressHydrationWarning
        />
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
        {speechPaused && (
          <span style={{ marginLeft: 12, fontSize: 13, color: "#6b7280" }} suppressHydrationWarning>
            ⏸ Reading paused
          </span>
        )}
        {imageBusy && (
          <span style={{ marginLeft: 12, fontSize: 13, color: "#6b7280" }} suppressHydrationWarning>
            🖼 Analysing image… (you can keep asking questions)
          </span>
        )}
      </div>

      {notice && (
        <div
          className="panel"
          style={{ borderColor: "#f59e0b", fontSize: 13 }}
          suppressHydrationWarning
        >
          {notice}
        </div>
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
          {textAnswer || "…"}
        </div>
      </div>

      <div className="panel" suppressHydrationWarning>
        <div className="label" suppressHydrationWarning>
          Product details (from image)
        </div>
        <div className="answer-text" suppressHydrationWarning>
          {imageAnswer || (imageBusy ? "Analysing…" : "…")}
        </div>
        {imageAnswer && state !== "speaking" && (
          <button
            onClick={readImageAnswer}
            style={{
              marginTop: 8,
              padding: "8px 14px",
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 8,
              border: "1px solid #2563eb",
              background: "transparent",
              color: "#2563eb",
              cursor: "pointer",
            }}
            suppressHydrationWarning
          >
            🔊 Read product details
          </button>
        )}
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
        (optionally with a question) to capture the camera frame; the image is analysed in the
        background while you can keep asking normal questions (one question at a time). When the
        product details are ready you'll be asked if you want them read out — say <b>"yes"</b> or{" "}
        <b>"no"</b>. Tap the <b>left half</b> of the camera (or say <b>"stop"</b>) to cancel the
        reading completely; this never cancels image processing. Tap the <b>right half</b> to
        pause the reading and start listening, and tap it again to continue from where it
        stopped. Say <b>"start a new session"</b> to reset the conversation. On phones the
        microphone pauses while the assistant is speaking, so use the camera halves to interrupt.
        Use Chrome — SpeechRecognition is not supported in Firefox/Safari.
      </p>
    </div>
  );
}