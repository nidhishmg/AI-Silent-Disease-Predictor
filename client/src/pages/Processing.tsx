import { useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import {
  CheckCircle2, Loader2, BrainCircuit, ScanFace, Mic,
  Waves, HeartPulse, Sparkles
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useScanResults, type ScanResults } from "@/context/ScanContext";

const stages = [
  { label: "Analyzing facial landmarks", icon: ScanFace },
  { label: "Detecting facial biomarkers", icon: Sparkles },
  { label: "Extracting vocal features", icon: Mic },
  { label: "Detecting breathing patterns", icon: Waves },
  { label: "Running AI health prediction model", icon: BrainCircuit },
  { label: "Generating health insights", icon: HeartPulse },
];

export default function Processing() {
  const [, setLocation] = useLocation();
  const { faceData, voiceData, setResults } = useScanResults();

  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState(0);
  const [completedStages, setCompletedStages] = useState<boolean[]>(
    new Array(stages.length).fill(false)
  );
  const [error, setError] = useState<string | null>(null);

  const faceRef = useRef(faceData);
  faceRef.current = faceData;
  const voiceRef = useRef(voiceData);
  voiceRef.current = voiceData;
  const setResultsRef = useRef(setResults);
  setResultsRef.current = setResults;
  const setLocationRef = useRef(setLocation);
  setLocationRef.current = setLocation;
  const apiCalledRef = useRef(false);

  useEffect(() => {
    if (apiCalledRef.current) return;
    apiCalledRef.current = true;

    let cancelled = false;
    const TOTAL = stages.length;
    const STAGE_MS = 1300;
    const TICK = 40;
    const TICKS_PER_STAGE = STAGE_MS / TICK;
    const PROGRESS_PER_TICK = (100 / TOTAL) / TICKS_PER_STAGE;

    let stage = 0;
    let prog = 0;
    let apiDone = false;
    let apiData: any = null;
    let apiErr: string | null = null;

    // API call
    (async () => {
      try {
        const res = await fetch("/api/full-scan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: faceRef.current?.image || undefined,
            audio: voiceRef.current?.audio || undefined,
            sampleRate: voiceRef.current?.sampleRate || 22050,
          }),
        });
        const data = await res.json();
        if (data.error) apiErr = data.error;
        else apiData = data;
      } catch {
        apiErr = "Analysis failed. Please try again.";
      } finally {
        apiDone = true;
      }
    })();

    // Animation
    const ticker = setInterval(() => {
      if (cancelled) return;
      const stageEnd = ((stage + 1) / TOTAL) * 100;

      if (prog < stageEnd) {
        prog = Math.min(prog + PROGRESS_PER_TICK, stageEnd);
        setProgress(Math.round(prog));
      } else {
        setCompletedStages(prev => {
          const next = [...prev];
          next[stage] = true;
          return next;
        });
        if (stage < TOTAL - 1) {
          stage++;
          setCurrentStage(stage);
        } else {
          clearInterval(ticker);
          setProgress(100);
          const nav = () => {
            if (cancelled) return;
            if (!apiDone) { setTimeout(nav, 200); return; }
            if (apiErr) { setError(apiErr); return; }
            if (apiData) {
              const r: ScanResults = {
                face: apiData.face,
                voice: apiData.voice,
                prediction: apiData.prediction,
                timestamp: new Date().toISOString(),
              };
              setResultsRef.current(r);
              setTimeout(() => { if (!cancelled) setLocationRef.current("/dashboard"); }, 600);
            }
          };
          nav();
        }
      }
    }, TICK);

    return () => { cancelled = true; clearInterval(ticker); apiCalledRef.current = false; };
  }, []);

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-2xl w-full space-y-10 py-12"
      >
        {/* Spinner */}
        <div className="text-center space-y-4">
          <div className="relative w-36 h-36 mx-auto">
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
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
              className="absolute inset-8 rounded-full border-2 border-emerald-200 border-t-emerald-500"
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <BrainCircuit className="w-10 h-10 text-primary animate-pulse" />
            </div>
          </div>
          <h2 className="text-3xl font-bold">Neural Engine Active</h2>
          <p className="text-muted-foreground text-sm max-w-md mx-auto">
            Processing multi-modal biometric data through the AI health prediction model
          </p>
        </div>

        {/* Stage Checklist */}
        <div className="glass-card p-6 rounded-2xl space-y-2">
          {stages.map((s, i) => {
            const isDone = completedStages[i];
            const isCurrent = currentStage === i && !isDone;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  isCurrent
                    ? "bg-primary/10 border border-primary/30 shadow-sm"
                    : isDone
                    ? "bg-emerald-50"
                    : "bg-muted/20"
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                ) : isCurrent ? (
                  <Loader2 className="w-5 h-5 text-primary animate-spin shrink-0" />
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-muted-foreground/20 shrink-0" />
                )}
                <s.icon className={`w-4 h-4 shrink-0 ${
                  isDone ? "text-emerald-500" : isCurrent ? "text-primary" : "text-muted-foreground/40"
                }`} />
                <span className={`text-sm font-medium ${
                  isCurrent ? "text-primary" : isDone ? "text-emerald-600" : "text-muted-foreground/60"
                }`}>
                  {s.label}
                </span>
                {isDone && (
                  <span className="ml-auto text-xs text-emerald-500 font-medium">Complete</span>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Progress */}
        <div className="space-y-2">
          <Progress value={progress} className="h-3" />
          <div className="flex justify-between text-sm font-medium text-muted-foreground">
            <span>Overall progress</span>
            <span className="text-primary font-bold">{progress}%</span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 rounded-xl bg-destructive/10 text-destructive text-sm text-center">
            {error}
          </div>
        )}
      </motion.div>
    </div>
  );
}
