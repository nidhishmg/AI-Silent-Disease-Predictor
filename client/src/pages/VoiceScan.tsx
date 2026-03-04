import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import {
  Mic, Upload, Square, CheckCircle2, AlertCircle,
  ArrowRight, ArrowLeft, Volume2, VolumeX, Timer
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useScanResults } from "@/context/ScanContext";

const voiceInstructions = [
  "Speak clearly into your microphone",
  "Maintain a normal speaking pace",
  "Avoid background noise",
  "Hold microphone close to your mouth",
];

export default function VoiceScan() {
  const [, setLocation] = useLocation();
  const { setVoiceData } = useScanResults();

  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);
  const [capturedAudio, setCapturedAudio] = useState<string | null>(null);
  const [audioSampleRate, setAudioSampleRate] = useState(22050);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ─── Waveform visualization ────────────────────────────────
  const drawWaveform = useCallback(() => {
    const analyser = analyserRef.current;
    const canvas = canvasRef.current;
    if (!analyser || !canvas) return;

    const ctx = canvas.getContext("2d")!;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "rgba(246, 249, 252, 0.3)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#1a73e8";
      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  }, []);

  // ─── Recording ─────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      setAudioSampleRate(audioCtx.sampleRate);

      // Set up analyser for waveform
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      drawWaveform();

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        cancelAnimationFrame(animFrameRef.current);
        audioCtx.close();
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onload = (ev) => setCapturedAudio(ev.target?.result as string);
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);

      // Auto-stop after 10 seconds
      setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
          setIsRecording(false);
        }
      }, 10000);
    } catch {
      setError("Microphone access denied. You can upload audio instead.");
    }
  }, [drawWaveform]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const handleAudioUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => setCapturedAudio(ev.target?.result as string);
      reader.readAsDataURL(file);
    };
    input.click();
  }, []);

  const handleVideoUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => setCapturedAudio(ev.target?.result as string);
      reader.readAsDataURL(file);
    };
    input.click();
  }, []);

  // Recording timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => setRecordTime((t: number) => t + 1), 1000);
    } else {
      setRecordTime(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const handleProceed = () => {
    setVoiceData({ audio: capturedAudio, sampleRate: audioSampleRate });
    setLocation("/processing");
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <div className="inline-flex items-center gap-2 text-xs font-semibold text-primary bg-primary/10 px-3 py-1 rounded-full mb-4">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          STEP 2 OF 3
        </div>
        <h1 className="text-4xl font-bold mb-2">Voice Biomarker Analysis</h1>
        <p className="text-muted-foreground max-w-lg mx-auto">
          Vocal patterns reveal micro-tremors, breathing irregularities, and stress markers invisible to the human ear.
        </p>
      </motion.div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-destructive/10 text-destructive flex items-center gap-2 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setError(null)}>Dismiss</Button>
        </div>
      )}

      <div className="space-y-8">
        {/* Instructions Card */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card p-6 rounded-2xl"
        >
          <h3 className="font-bold text-base mb-3 flex items-center gap-2">
            <Volume2 className="w-5 h-5 text-primary" />
            Recording Instructions
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {voiceInstructions.map((inst, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
                {inst}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Sentence to read */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="relative glass-card p-8 rounded-3xl border-2 border-primary/20 text-center overflow-hidden"
        >
          <div className="absolute top-0 left-0 w-full h-1.5 bg-linear-to-r from-primary via-secondary to-primary" />
          <div className="absolute top-4 right-4 bg-primary/10 text-primary text-xs px-2 py-0.5 rounded-full font-semibold">
            Read Aloud
          </div>
          <VolumeX className="w-8 h-8 text-primary/30 mx-auto mb-4" />
          <p className="text-xl md:text-2xl font-medium leading-relaxed text-foreground/90 max-w-2xl mx-auto">
            &ldquo;My health is important and I feel calm and steady today.
            <br />
            I am speaking clearly for my health analysis.&rdquo;
          </p>
        </motion.div>

        {/* Recording Area */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-8 rounded-3xl flex flex-col items-center gap-6"
        >
          {/* Waveform Canvas */}
          <div className="w-full h-28 bg-muted/30 rounded-xl overflow-hidden relative">
            <canvas
              ref={canvasRef}
              width={800}
              height={112}
              className="w-full h-full"
            />
            {!isRecording && !capturedAudio && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex items-center gap-1">
                  {Array.from({ length: 40 }).map((_, i) => (
                    <div
                      key={i}
                      className="w-1 bg-muted-foreground/20 rounded-full"
                      style={{ height: `${10 + Math.sin(i * 0.5) * 15}px` }}
                    />
                  ))}
                </div>
              </div>
            )}
            {/* Animated bars for recording */}
            {isRecording && (
              <div className="absolute inset-0 flex items-center justify-center gap-0.5 pointer-events-none">
                {Array.from({ length: 50 }).map((_, i) => (
                  <motion.div
                    key={i}
                    animate={{ height: ["15%", `${Math.random() * 70 + 20}%`, "15%"] }}
                    transition={{ duration: 0.4 + Math.random() * 0.4, repeat: Infinity, delay: i * 0.03 }}
                    className="w-1 rounded-full bg-primary/40"
                  />
                ))}
              </div>
            )}
          </div>

          {/* Record Button + Timer */}
          <div className="flex items-center gap-6">
            <Button
              size="lg"
              variant={isRecording ? "destructive" : "default"}
              className="rounded-full w-20 h-20 p-0 shadow-xl text-lg"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!!capturedAudio}
            >
              {isRecording ? <Square className="w-8 h-8 fill-current" /> : <Mic className="w-8 h-8" />}
            </Button>

            {isRecording && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2"
              >
                <Timer className="w-5 h-5 text-destructive" />
                <span className="text-2xl font-mono font-bold text-destructive tabular-nums">
                  00:{String(recordTime).padStart(2, "0")}
                </span>
                <span className="text-sm text-muted-foreground">/ 00:10</span>
                <div className="w-3 h-3 rounded-full bg-destructive animate-pulse ml-2" />
              </motion.div>
            )}
          </div>

          {/* Status */}
          {capturedAudio && !isRecording && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 text-sm text-emerald-600 font-semibold bg-emerald-50 px-4 py-2 rounded-full"
            >
              <CheckCircle2 className="w-4 h-4" /> Voice recorded successfully
            </motion.div>
          )}

          {/* Upload options */}
          {!isRecording && !capturedAudio && (
            <div className="flex gap-6 text-sm text-muted-foreground">
              <Button variant="link" className="h-auto p-0 gap-2" onClick={handleAudioUpload}>
                <Upload className="w-4 h-4" /> Upload Audio File
              </Button>
              <span className="text-muted-foreground/30">|</span>
              <Button variant="link" className="h-auto p-0 gap-2" onClick={handleVideoUpload}>
                <Upload className="w-4 h-4" /> Upload Video File
              </Button>
            </div>
          )}

          {/* Retake */}
          {capturedAudio && !isRecording && (
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              onClick={() => setCapturedAudio(null)}
            >
              Re-record
            </Button>
          )}
        </motion.div>

        {/* Navigation */}
        <div className="flex justify-between pt-4 border-t">
          <Button variant="ghost" className="gap-2" onClick={() => setLocation("/face-scan")}>
            <ArrowLeft className="w-4 h-4" /> Back to Face Scan
          </Button>
          <Button
            size="lg"
            className="px-8 rounded-full premium-shadow group"
            onClick={handleProceed}
          >
            Run AI Health Analysis
            <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Button>
        </div>
      </div>
    </div>
  );
}
