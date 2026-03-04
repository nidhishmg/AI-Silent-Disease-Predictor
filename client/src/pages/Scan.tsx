import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Camera, Upload, Video, Mic, CheckCircle2, AlertCircle, 
  Play, Square, Loader2, BrainCircuit
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useScanResults, type ScanResults } from "@/context/ScanContext";

const processingSteps = [
  "Analyzing facial landmarks...",
  "Detecting voice biomarkers...",
  "Extracting breathing signals...",
  "Running AI health model...",
  "Generating health insights..."
];

export default function Scan() {
  const [, setLocation] = useLocation();
  const { setResults } = useScanResults();

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);
  const [processingIndex, setProcessingIndex] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);

  // ── Camera state ───────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [faceIndicators, setFaceIndicators] = useState({
    eye: 0, symmetry: 0, fatigue: 0, brightness: 0,
  });

  // ── Audio state ────────────────────────────────────────────────────
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [capturedAudio, setCapturedAudio] = useState<string | null>(null);
  const [audioSampleRate, setAudioSampleRate] = useState(22050);

  // ── Error state ────────────────────────────────────────────────────
  const [error, setError] = useState<string | null>(null);

  // ═══════════════════════════════════════════════════════════════════
  // CAMERA
  // ═══════════════════════════════════════════════════════════════════

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
        setError(null);
      }
    } catch (err: any) {
      console.error("Camera error:", err);
      setError("Camera access denied. You can upload a photo instead.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  }, []);

  // Start camera when step 1 mounts
  useEffect(() => {
    if (step === 1 && !capturedImage) {
      startCamera();
    }
    return () => { if (step !== 1) stopCamera(); };
  }, [step, capturedImage, startCamera, stopCamera]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    setCapturedImage(dataUrl);
    stopCamera();

    // Immediately run face analysis
    runFaceScan(dataUrl);
  }, [stopCamera]);

  const handlePhotoUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const dataUrl = ev.target?.result as string;
        setCapturedImage(dataUrl);
        stopCamera();
        runFaceScan(dataUrl);
      };
      reader.readAsDataURL(file);
    };
    input.click();
  }, [stopCamera]);

  const handleVideoRecord = useCallback(async () => {
    if (!videoRef.current) return;
    const stream = videoRef.current.srcObject as MediaStream;
    if (!stream) return;

    // Record 5 seconds of video, capture last frame
    setTimeout(() => {
      capturePhoto();
    }, 5000);
  }, [capturePhoto]);

  const handleVideoUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      // Extract a frame from the video
      const video = document.createElement("video");
      video.preload = "metadata";
      video.onloadeddata = () => {
        video.currentTime = 1; // seek to 1s
      };
      video.onseeked = () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d")!.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
        setCapturedImage(dataUrl);
        stopCamera();
        runFaceScan(dataUrl);
        URL.revokeObjectURL(video.src);
      };
      video.src = URL.createObjectURL(file);
    };
    input.click();
  }, [stopCamera]);

  async function runFaceScan(imageDataUrl: string) {
    try {
      const res = await fetch("/api/face-scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      // Update scanning indicators with real data
      setFaceIndicators({
        eye: Math.round(100 - (data.blink_instability ?? 0)),
        symmetry: Math.round(data.symmetry_score ?? 0),
        fatigue: Math.round(100 - (data.face_fatigue ?? 0)),
        brightness: Math.round(100 - (data.brightness_variance ?? 0)),
      });
    } catch (err) {
      console.error("Face scan error:", err);
      // Indicators stay at 0; not critical
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // AUDIO RECORDING
  // ═══════════════════════════════════════════════════════════════════

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      setAudioSampleRate(audioCtx.sampleRate);
      audioCtx.close();

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        // Convert blob to base64
        const reader = new FileReader();
        reader.onload = (ev) => {
          setCapturedAudio(ev.target?.result as string);
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);

      // Auto-stop after 5 seconds
      setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
          setIsRecording(false);
        }
      }, 5000);
    } catch (err: any) {
      console.error("Mic error:", err);
      setError("Microphone access denied. You can upload audio instead.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const handleAudioUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        setCapturedAudio(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    };
    input.click();
  }, []);

  // Recording timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => setRecordTime((t) => t + 1), 1000);
    } else {
      setRecordTime(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  // ═══════════════════════════════════════════════════════════════════
  // PROCESSING (Step 3) — calls /api/full-scan
  // ═══════════════════════════════════════════════════════════════════

  // Track which stages are completed
  const [completedStages, setCompletedStages] = useState<boolean[]>(
    new Array(processingSteps.length).fill(false)
  );

  // Use refs for values needed inside the effect to avoid re-triggering it
  const capturedImageRef = useRef(capturedImage);
  capturedImageRef.current = capturedImage;
  const capturedAudioRef = useRef(capturedAudio);
  capturedAudioRef.current = capturedAudio;
  const audioSampleRateRef = useRef(audioSampleRate);
  audioSampleRateRef.current = audioSampleRate;
  const setResultsRef = useRef(setResults);
  setResultsRef.current = setResults;
  const setLocationRef = useRef(setLocation);
  setLocationRef.current = setLocation;
  const apiCalledRef = useRef(false);

  useEffect(() => {
    if (step !== 3) return;
    // Prevent duplicate calls (React Strict Mode or unstable deps)
    if (apiCalledRef.current) return;
    apiCalledRef.current = true;

    let cancelled = false;

    // Reset state
    setProcessingProgress(0);
    setProcessingIndex(0);
    setCompletedStages(new Array(processingSteps.length).fill(false));

    const TOTAL_STAGES = processingSteps.length; // 5
    const STAGE_DURATION = 1400; // ms per stage (minimum ~7s total)
    const TICK_INTERVAL = 50; // ms between progress ticks
    const TICKS_PER_STAGE = STAGE_DURATION / TICK_INTERVAL;
    const PROGRESS_PER_TICK = (100 / TOTAL_STAGES) / TICKS_PER_STAGE;

    let currentStage = 0;
    let currentProgress = 0;
    let apiDone = false;
    let apiData: any = null;
    let apiError: string | null = null;

    // Start the API call in the background (exactly once)
    (async () => {
      try {
        const res = await fetch("/api/full-scan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: capturedImageRef.current || undefined,
            audio: capturedAudioRef.current || undefined,
            sampleRate: audioSampleRateRef.current,
          }),
        });
        const data = await res.json();
        if (data.error) {
          apiError = data.error;
        } else {
          apiData = data;
        }
      } catch (err) {
        console.error("Full scan error:", err);
        apiError = "Analysis failed. Please try again.";
      } finally {
        apiDone = true;
      }
    })();

    // Animate progress smoothly through all stages
    const ticker = setInterval(() => {
      if (cancelled) return;

      const stageEnd = ((currentStage + 1) / TOTAL_STAGES) * 100;

      if (currentProgress < stageEnd) {
        currentProgress = Math.min(currentProgress + PROGRESS_PER_TICK, stageEnd);
        setProcessingProgress(Math.round(currentProgress));
      } else {
        // Stage finished — mark it complete
        setCompletedStages((prev) => {
          const next = [...prev];
          next[currentStage] = true;
          return next;
        });

        if (currentStage < TOTAL_STAGES - 1) {
          currentStage++;
          setProcessingIndex(currentStage);
        } else {
          // All stages done — wait for API if needed, then navigate
          clearInterval(ticker);
          setProcessingProgress(100);

          const waitAndNavigate = () => {
            if (cancelled) return;
            if (!apiDone) {
              setTimeout(waitAndNavigate, 200);
              return;
            }
            if (apiError) {
              setError(apiError);
              return;
            }
            if (apiData) {
              const scanResults: ScanResults = {
                face: apiData.face,
                voice: apiData.voice,
                prediction: apiData.prediction,
                timestamp: new Date().toISOString(),
              };
              setResultsRef.current(scanResults);
              setTimeout(() => {
                if (!cancelled) setLocationRef.current("/dashboard");
              }, 800);
            }
          };
          waitAndNavigate();
        }
      }
    }, TICK_INTERVAL);

    return () => {
      cancelled = true;
      clearInterval(ticker);
      apiCalledRef.current = false;
    };
  // Only re-run when step changes to 3 — all other values read via refs
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  // ═══════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Error banner */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-destructive/10 text-destructive flex items-center gap-2 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setError(null)}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Stepper */}
      <div className="flex items-center justify-center mb-12">
        {[1, 2, 3].map((s, i) => (
          <div key={s} className="flex items-center">
            <div className={`flex items-center justify-center w-10 h-10 rounded-full font-bold text-sm transition-colors ${
              step >= s ? "bg-primary text-white premium-shadow" : "bg-muted text-muted-foreground"
            }`}>
              {s}
            </div>
            {i < 2 && (
              <div className={`w-24 h-1 mx-2 rounded-full ${
                step > s ? "bg-primary" : "bg-muted"
              }`} />
            )}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {/* STEP 1: FACE SCAN */}
        {step === 1 && (
          <motion.div
            key="step1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="grid lg:grid-cols-3 gap-8"
          >
            {/* Instructions */}
            <div className="lg:col-span-1 space-y-6">
              <div className="glass-card p-6 rounded-2xl">
                <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-primary" />
                  Face Scan Instructions
                </h3>
                <ul className="space-y-4 text-sm text-muted-foreground">
                  <li className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                    Keep your face centered in the frame
                  </li>
                  <li className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                    Look directly at the camera
                  </li>
                  <li className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                    Maintain a neutral expression
                  </li>
                  <li className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                    Ensure good front lighting
                  </li>
                  <li className="flex gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                    Remove glasses if possible
                  </li>
                </ul>
              </div>

              <div className="glass-card p-6 rounded-2xl space-y-4">
                <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                  Scanning Indicators
                </h4>
                {[
                  { label: "Eye movement detection", progress: faceIndicators.eye },
                  { label: "Facial symmetry analysis", progress: faceIndicators.symmetry },
                  { label: "Fatigue detection", progress: faceIndicators.fatigue },
                  { label: "Skin brightness analysis", progress: faceIndicators.brightness },
                ].map((ind, i) => (
                  <div key={i} className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span>{ind.label}</span>
                      <span className="text-primary">{ind.progress}%</span>
                    </div>
                    <Progress value={ind.progress} className="h-1.5" />
                  </div>
                ))}
              </div>
            </div>

            {/* Camera View */}
            <div className="lg:col-span-2 space-y-6">
              <div className="relative aspect-video bg-black rounded-3xl overflow-hidden shadow-2xl ring-1 ring-black/5">
                {/* Live camera feed or captured image */}
                {capturedImage ? (
                  <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                      style={{ display: cameraActive ? "block" : "none" }}
                    />
                    {!cameraActive && (
                      <div className="absolute inset-0 bg-zinc-900 flex items-center justify-center">
                        <Camera className="w-16 h-16 text-zinc-700" />
                      </div>
                    )}
                  </>
                )}
                
                {/* Positioning Grid Overlay */}
                {!capturedImage && (
                  <>
                    <div className="absolute inset-0 scanning-grid pointer-events-none opacity-50" />
                    
                    {/* Face Outline Overlay */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-64 h-80 border-2 border-primary/50 border-dashed rounded-[40%] relative">
                        {/* Corner markers */}
                        <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-primary rounded-tl-3xl -translate-x-2 -translate-y-2" />
                        <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-primary rounded-tr-3xl translate-x-2 -translate-y-2" />
                        <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-primary rounded-bl-3xl -translate-x-2 translate-y-2" />
                        <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-primary rounded-br-3xl translate-x-2 translate-y-2" />
                        
                        {/* Alignment Lines */}
                        <div className="absolute top-1/2 left-0 w-full h-px bg-primary/30" />
                        <div className="absolute top-0 left-1/2 w-px h-full bg-primary/30" />
                        
                        {/* Scanning Line Animation */}
                        <motion.div 
                          animate={{ y: ["0%", "100%", "0%"] }}
                          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                          className="absolute top-0 left-0 w-full h-1 bg-primary/80 shadow-[0_0_10px_rgba(26,115,232,0.8)] z-10"
                        />
                      </div>
                    </div>
                  </>
                )}

                {/* Captured badge */}
                {capturedImage && (
                  <div className="absolute top-4 right-4 bg-emerald-500 text-white px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Captured
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Button
                  variant="outline"
                  className="h-12 bg-white"
                  onClick={capturedImage ? () => { setCapturedImage(null); setFaceIndicators({ eye: 0, symmetry: 0, fatigue: 0, brightness: 0 }); startCamera(); } : capturePhoto}
                  disabled={!cameraActive && !capturedImage}
                >
                  <Camera className="w-4 h-4 mr-2" />
                  {capturedImage ? "Retake" : "Capture"}
                </Button>
                <Button variant="outline" className="h-12 bg-white" onClick={handleVideoRecord} disabled={!cameraActive}>
                  <Video className="w-4 h-4 mr-2" /> Record 5s
                </Button>
                <Button variant="outline" className="h-12 bg-white" onClick={handlePhotoUpload}>
                  <Upload className="w-4 h-4 mr-2" /> Photo
                </Button>
                <Button variant="outline" className="h-12 bg-white" onClick={handleVideoUpload}>
                  <Upload className="w-4 h-4 mr-2" /> Video
                </Button>
              </div>

              <div className="flex justify-end pt-4">
                <Button size="lg" className="px-8 rounded-full" onClick={() => { stopCamera(); setStep(2); }}>
                  Continue to Voice Scan
                </Button>
              </div>
            </div>
          </motion.div>
        )}

        {/* STEP 2: VOICE SCAN */}
        {step === 2 && (
          <motion.div
            key="step2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="max-w-3xl mx-auto space-y-8"
          >
            <div className="text-center space-y-2">
              <h2 className="text-3xl font-bold">Voice Analysis</h2>
              <p className="text-muted-foreground">Please read the text below clearly and naturally.</p>
            </div>

            <div className="glass-card p-8 rounded-3xl border-2 border-primary/20 text-center relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-linear-to-r from-primary via-secondary to-primary" />
              <p className="text-2xl font-medium leading-relaxed text-foreground/90">
                "My health is important and I feel calm and steady today.<br/>
                I am speaking clearly for my health analysis."
              </p>
            </div>

            <div className="glass-card p-8 rounded-3xl flex flex-col items-center gap-6">
              {/* Audio visualizer simulation */}
              <div className="h-24 w-full max-w-md flex items-center justify-center gap-1">
                {Array.from({ length: 30 }).map((_, i) => (
                  <motion.div
                    key={i}
                    animate={{ height: isRecording ? ["20%", `${Math.random() * 80 + 20}%`, "20%"] : "10%" }}
                    transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.05 }}
                    className={`w-2 rounded-full ${isRecording ? 'bg-primary' : 'bg-muted-foreground/30'}`}
                    style={{ height: '10%' }}
                  />
                ))}
              </div>

              <div className="flex items-center gap-4">
                <Button 
                  size="lg" 
                  variant={isRecording ? "destructive" : "default"}
                  className="rounded-full w-16 h-16 p-0 shadow-xl"
                  onClick={() => {
                    if (isRecording) {
                      stopRecording();
                    } else {
                      startRecording();
                    }
                  }}
                >
                  {isRecording ? <Square className="w-6 h-6 fill-current" /> : <Mic className="w-6 h-6" />}
                </Button>
                
                {isRecording && (
                  <div className="text-xl font-mono font-medium text-destructive flex items-center gap-2 animate-pulse">
                    <div className="w-3 h-3 rounded-full bg-destructive" />
                    00:0{recordTime} / 00:05
                  </div>
                )}
              </div>

              {/* Status badges */}
              {capturedAudio && !isRecording && (
                <div className="flex items-center gap-2 text-sm text-emerald-600 font-medium">
                  <CheckCircle2 className="w-4 h-4" /> Audio recorded successfully
                </div>
              )}

              {!isRecording && !capturedAudio && (
                <div className="flex gap-4 text-sm text-muted-foreground">
                  <Button variant="link" className="h-auto p-0" onClick={handleAudioUpload}>
                    <Upload className="w-4 h-4 mr-2" /> Upload Audio
                  </Button>
                  <span>•</span>
                  <Button variant="link" className="h-auto p-0" onClick={handleAudioUpload}>
                    <Upload className="w-4 h-4 mr-2" /> Upload Video
                  </Button>
                </div>
              )}
            </div>

            <div className="flex justify-between pt-8 border-t">
              <Button variant="ghost" onClick={() => setStep(1)}>Back</Button>
              <Button size="lg" className="px-8 rounded-full premium-shadow" onClick={() => {
                setProcessingProgress(0);
                setProcessingIndex(0);
                setStep(3);
              }}>
                <BrainCircuit className="w-5 h-5 mr-2" />
                Run AI Health Analysis
              </Button>
            </div>
          </motion.div>
        )}

        {/* STEP 3: PROCESSING */}
        {step === 3 && (
          <motion.div
            key="step3"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-2xl mx-auto space-y-10 py-12"
          >
            {/* Spinner + title */}
            <div className="text-center space-y-4">
              <div className="relative w-32 h-32 mx-auto">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 rounded-full border-4 border-primary/20 border-t-primary"
                />
                <motion.div
                  animate={{ rotate: -360 }}
                  transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-4 rounded-full border-4 border-secondary/20 border-t-secondary"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <BrainCircuit className="w-10 h-10 text-primary animate-pulse" />
                </div>
              </div>
              <h2 className="text-2xl font-bold">Neural Engine Active</h2>
              <p className="text-muted-foreground text-sm">Analyzing your health data with AI — please wait</p>
            </div>

            {/* Stage checklist */}
            <div className="glass-card p-6 rounded-2xl space-y-3">
              {processingSteps.map((label, i) => {
                const isDone = completedStages[i];
                const isCurrent = processingIndex === i && !isDone;
                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${
                      isCurrent
                        ? "bg-primary/10 border border-primary/30"
                        : isDone
                        ? "bg-emerald-50 dark:bg-emerald-500/10"
                        : "bg-muted/30"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                    ) : isCurrent ? (
                      <Loader2 className="w-5 h-5 text-primary animate-spin shrink-0" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-muted-foreground/30 shrink-0" />
                    )}
                    <span
                      className={`text-sm font-medium ${
                        isCurrent
                          ? "text-primary"
                          : isDone
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-muted-foreground"
                      }`}
                    >
                      {label}
                    </span>
                    {isDone && (
                      <span className="ml-auto text-xs text-emerald-500 font-medium">Done</span>
                    )}
                  </motion.div>
                );
              })}
            </div>

            {/* Progress bar */}
            <div className="space-y-2">
              <Progress value={processingProgress} className="h-3" />
              <div className="flex justify-between text-sm font-medium text-muted-foreground">
                <span>Overall progress</span>
                <span className="text-primary font-bold">{processingProgress}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
