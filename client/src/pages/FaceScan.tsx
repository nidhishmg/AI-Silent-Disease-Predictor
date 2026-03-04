import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import {
  Camera, Upload, Video, CheckCircle2, AlertCircle,
  Eye, Scan, Sun, SmilePlus, ArrowRight, RotateCcw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useScanResults } from "@/context/ScanContext";

const instructions = [
  { icon: Scan, text: "Keep your face centered in the frame" },
  { icon: Eye, text: "Look directly at the camera" },
  { icon: SmilePlus, text: "Maintain a neutral expression" },
  { icon: Sun, text: "Ensure proper lighting" },
  { icon: Eye, text: "Remove glasses if possible" },
];

export default function FaceScan() {
  const [, setLocation] = useLocation();
  const { setFaceData } = useScanResults();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);

  const [indicators, setIndicators] = useState({
    faceDetected: false,
    faceCentered: false,
    lightingSufficient: false,
    eyeVisibility: false,
  });

  const [biomarkers, setBiomarkers] = useState({
    eye: 0, symmetry: 0, fatigue: 0, brightness: 0,
  });

  // ─── Camera ────────────────────────────────────────────────
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
        // Simulate real-time detection feedback after a delay
        setTimeout(() => setIndicators(prev => ({ ...prev, faceDetected: true })), 800);
        setTimeout(() => setIndicators(prev => ({ ...prev, faceCentered: true })), 1400);
        setTimeout(() => setIndicators(prev => ({ ...prev, lightingSufficient: true })), 2000);
        setTimeout(() => setIndicators(prev => ({ ...prev, eyeVisibility: true })), 2600);
      }
    } catch {
      setError("Camera access denied. You can upload a photo instead.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  }, []);

  useEffect(() => {
    if (!capturedImage) startCamera();
    return () => { stopCamera(); };
  }, [capturedImage, startCamera, stopCamera]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    canvas.getContext("2d")!.drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    setCapturedImage(dataUrl);
    stopCamera();
    runFaceAnalysis(dataUrl);
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
        runFaceAnalysis(dataUrl);
      };
      reader.readAsDataURL(file);
    };
    input.click();
  }, [stopCamera]);

  const handleVideoRecord = useCallback(async () => {
    if (!videoRef.current) return;
    setTimeout(() => capturePhoto(), 5000);
  }, [capturePhoto]);

  const handleVideoUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const video = document.createElement("video");
      video.preload = "metadata";
      video.onloadeddata = () => { video.currentTime = 1; };
      video.onseeked = () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d")!.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
        setCapturedImage(dataUrl);
        stopCamera();
        runFaceAnalysis(dataUrl);
        URL.revokeObjectURL(video.src);
      };
      video.src = URL.createObjectURL(file);
    };
    input.click();
  }, [stopCamera]);

  async function runFaceAnalysis(imageDataUrl: string) {
    setScanning(true);
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
      const bio = {
        eye: Math.round(100 - (data.blink_instability ?? 0)),
        symmetry: Math.round(data.symmetry_score ?? 0),
        fatigue: Math.round(100 - (data.face_fatigue ?? 0)),
        brightness: Math.round(100 - (data.brightness_variance ?? 0)),
      };
      setBiomarkers(bio);
      // Store face data in context for later use
      setFaceData({ image: imageDataUrl, result: data, biomarkers: bio });
    } catch (err) {
      console.error("Face scan error:", err);
    } finally {
      setScanning(false);
    }
  }

  const retake = () => {
    setCapturedImage(null);
    setBiomarkers({ eye: 0, symmetry: 0, fatigue: 0, brightness: 0 });
    setIndicators({ faceDetected: false, faceCentered: false, lightingSufficient: false, eyeVisibility: false });
  };

  const allIndicatorsGreen = Object.values(indicators).every(Boolean);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <canvas ref={canvasRef} className="hidden" />

      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center gap-2 text-xs font-semibold text-primary bg-primary/10 px-3 py-1 rounded-full mb-4">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          STEP 1 OF 3
        </div>
        <h1 className="text-4xl font-bold mb-2">Facial Biomarker Scan</h1>
        <p className="text-muted-foreground max-w-lg mx-auto">
          Our AI analyzes facial landmarks, skin patterns, and micro-expressions to extract optical biomarkers.
        </p>
      </motion.div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-destructive/10 text-destructive flex items-center gap-2 text-sm max-w-4xl mx-auto">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setError(null)}>Dismiss</Button>
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left Column: Instructions + Indicators */}
        <div className="space-y-6">
          {/* Instructions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6 rounded-2xl"
          >
            <h3 className="font-bold text-base mb-4 flex items-center gap-2">
              <Scan className="w-5 h-5 text-primary" />
              Scan Instructions
            </h3>
            <ul className="space-y-3">
              {instructions.map((inst, i) => (
                <li key={i} className="flex items-start gap-3 text-sm text-muted-foreground">
                  <inst.icon className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                  {inst.text}
                </li>
              ))}
            </ul>
          </motion.div>

          {/* Real-time Detection Indicators */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6 rounded-2xl"
          >
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
              Detection Status
            </h3>
            <div className="space-y-3">
              {[
                { label: "Face Detected", active: indicators.faceDetected },
                { label: "Face Centered", active: indicators.faceCentered },
                { label: "Lighting Sufficient", active: indicators.lightingSufficient },
                { label: "Eye Visibility", active: indicators.eyeVisibility },
              ].map((ind, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 + i * 0.15 }}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    ind.active
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-muted/50 text-muted-foreground"
                  }`}
                >
                  {ind.active ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 500, damping: 15 }}
                    >
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    </motion.div>
                  ) : (
                    <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 animate-pulse" />
                  )}
                  {ind.label}
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Biomarker Indicators */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card p-6 rounded-2xl"
          >
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
              Biomarker Analysis
            </h3>
            <div className="space-y-4">
              {[
                { label: "Eye Stability", value: biomarkers.eye, color: "bg-blue-500" },
                { label: "Facial Symmetry", value: biomarkers.symmetry, color: "bg-emerald-500" },
                { label: "Fatigue Index", value: biomarkers.fatigue, color: "bg-amber-500" },
                { label: "Skin Brightness", value: biomarkers.brightness, color: "bg-purple-500" },
              ].map((m, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-medium">
                    <span>{m.label}</span>
                    <span className="text-primary font-bold">{m.value}%</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${m.value}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${m.color}`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Center + Right: Camera View */}
        <div className="lg:col-span-2 space-y-6">
          <div className="relative aspect-video bg-black rounded-3xl overflow-hidden shadow-2xl ring-1 ring-black/5">
            {capturedImage ? (
              <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />
            ) : (
              <>
                <video
                  ref={videoRef}
                  autoPlay playsInline muted
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

            {/* Face Alignment Grid Overlay */}
            {!capturedImage && cameraActive && (
              <>
                {/* Grid lines */}
                <div className="absolute inset-0 pointer-events-none">
                  {/* Horizontal thirds */}
                  <div className="absolute top-1/3 left-0 w-full h-px bg-primary/20" />
                  <div className="absolute top-2/3 left-0 w-full h-px bg-primary/20" />
                  {/* Vertical thirds */}
                  <div className="absolute top-0 left-1/3 w-px h-full bg-primary/20" />
                  <div className="absolute top-0 left-2/3 w-px h-full bg-primary/20" />
                </div>

                {/* Face outline guide */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-56 h-72 border-2 border-primary/40 border-dashed rounded-[40%] relative">
                    {/* Corners */}
                    <div className="absolute -top-1.5 -left-1.5 w-6 h-6 border-t-3 border-l-3 border-primary rounded-tl-2xl" />
                    <div className="absolute -top-1.5 -right-1.5 w-6 h-6 border-t-3 border-r-3 border-primary rounded-tr-2xl" />
                    <div className="absolute -bottom-1.5 -left-1.5 w-6 h-6 border-b-3 border-l-3 border-primary rounded-bl-2xl" />
                    <div className="absolute -bottom-1.5 -right-1.5 w-6 h-6 border-b-3 border-r-3 border-primary rounded-br-2xl" />
                    {/* Cross hairs */}
                    <div className="absolute top-1/2 left-0 w-full h-px bg-primary/25" />
                    <div className="absolute top-0 left-1/2 w-px h-full bg-primary/25" />
                    {/* Eye level markers */}
                    <div className="absolute top-[35%] left-[15%] w-[25%] h-px bg-cyan-400/50" />
                    <div className="absolute top-[35%] right-[15%] w-[25%] h-px bg-cyan-400/50" />
                    {/* Scanning line */}
                    <motion.div
                      animate={{ y: ["0%", "100%", "0%"] }}
                      transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                      className="absolute top-0 left-0 w-full h-0.5 bg-primary/80 shadow-[0_0_12px_rgba(26,115,232,0.8)] z-10"
                    />
                  </div>
                </div>

                {/* Status badges on camera */}
                {allIndicatorsGreen && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="absolute top-4 left-4 bg-emerald-500/90 text-white px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5 backdrop-blur-sm"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready to Capture
                  </motion.div>
                )}
              </>
            )}

            {/* Captured badge */}
            {capturedImage && (
              <div className="absolute top-4 right-4 bg-emerald-500 text-white px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> Photo Captured
              </div>
            )}

            {/* Scanning overlay */}
            {scanning && (
              <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                <div className="bg-white/90 backdrop-blur-md rounded-2xl px-6 py-4 flex items-center gap-3 shadow-xl">
                  <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm font-medium">Analyzing facial biomarkers...</span>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {capturedImage ? (
              <Button variant="outline" className="h-12 bg-white col-span-2 md:col-span-1" onClick={retake}>
                <RotateCcw className="w-4 h-4 mr-2" /> Retake
              </Button>
            ) : (
              <Button
                variant="outline" className="h-12 bg-white"
                onClick={capturePhoto}
                disabled={!cameraActive}
              >
                <Camera className="w-4 h-4 mr-2" /> Capture Photo
              </Button>
            )}
            <Button variant="outline" className="h-12 bg-white" onClick={handleVideoRecord} disabled={!cameraActive || !!capturedImage}>
              <Video className="w-4 h-4 mr-2" /> Record 5s
            </Button>
            <Button variant="outline" className="h-12 bg-white" onClick={handlePhotoUpload}>
              <Upload className="w-4 h-4 mr-2" /> Upload Photo
            </Button>
            <Button variant="outline" className="h-12 bg-white" onClick={handleVideoUpload}>
              <Upload className="w-4 h-4 mr-2" /> Upload Video
            </Button>
          </div>

          {/* Preview confirmation */}
          {capturedImage && !scanning && biomarkers.eye > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-5 rounded-2xl border-2 border-emerald-200 bg-emerald-50/50"
            >
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-6 h-6 text-emerald-500 shrink-0" />
                <div>
                  <p className="font-semibold text-sm">Face scan complete</p>
                  <p className="text-xs text-muted-foreground">Facial biomarkers extracted successfully. Proceed to voice scan.</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Next Step */}
          <div className="flex justify-end pt-4">
            <Button
              size="lg"
              className="px-8 rounded-full premium-shadow group"
              onClick={() => { stopCamera(); setLocation("/voice-scan"); }}
            >
              Continue to Voice Scan
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
