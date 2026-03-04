import { useMemo } from "react";
import { motion } from "framer-motion";
import { useLocation } from "wouter";
import {
  Download, Activity, ShieldAlert, HeartPulse, BrainCircuit,
  Eye, SmilePlus, Zap, Sun, Mic, Wind, Music2,
  BarChart3, Gauge, FileText, ShieldCheck, Sparkles,
  ChevronRight, Database, Clock, Hash, Lock,
  Heart, AlertTriangle, CheckCircle2, Info
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid
} from "recharts";
import { useScanResults } from "@/context/ScanContext";

// ── Dataset references ────────────────────────────────────────
const datasets = [
  {
    name: "UTKFace Dataset",
    purpose: "Study facial attributes such as age distribution, skin tone variation, and facial structure differences.",
    usage: "Statistical distributions of facial attributes guided synthetic generation of symmetry_score and brightness_variance.",
    features: ["symmetry_score", "brightness_variance"],
  },
  {
    name: "CelebA Dataset",
    purpose: "Large-scale dataset containing facial attributes and landmark annotations.",
    usage: "Provided statistical reference for facial landmark distributions and eye region features.",
    features: ["eye stability", "facial symmetry patterns"],
  },
  {
    name: "Driver Drowsiness Dataset",
    purpose: "Eye closure and fatigue patterns dataset.",
    usage: "Used to model fatigue and blink instability distributions.",
    features: ["blink_instability", "face_fatigue"],
  },
  {
    name: "RAVDESS Emotional Speech",
    purpose: "Audio dataset containing emotional speech recordings.",
    usage: "Guided generation of pitch variation and speech stability patterns.",
    features: ["voice_stress", "pitch_instability"],
  },
  {
    name: "Coswara Respiratory Dataset",
    purpose: "Breathing, cough, and speech audio dataset.",
    usage: "Used to simulate breathing rhythm variability patterns.",
    features: ["breathing_score"],
  },
  {
    name: "UCI Heart Disease Dataset",
    purpose: "Clinical dataset containing cardiovascular risk factors.",
    usage: "Guided the probability distribution of overall health risk labels.",
    features: ["risk labels"],
  },
  {
    name: "PIMA Diabetes Dataset",
    purpose: "Metabolic health indicators dataset.",
    usage: "Simulated correlations between fatigue, stress, and risk prediction.",
    features: ["fatigue-stress correlations"],
  },
];

export default function Dashboard() {
  const { results } = useScanResults();
  const [, setLocation] = useLocation();

  if (!results) {
    return (
      <div className="max-w-xl mx-auto px-4 py-24 text-center space-y-6">
        <BrainCircuit className="w-16 h-16 text-primary mx-auto opacity-40" />
        <h2 className="text-2xl font-bold">No Scan Results Yet</h2>
        <p className="text-muted-foreground">
          Complete a face and voice scan first to see your health analysis.
        </p>
        <Button className="rounded-full px-8" onClick={() => setLocation("/face-scan")}>
          Start Health Scan
        </Button>
      </div>
    );
  }

  const { face, voice, prediction } = results;

  // ── Computed values ─────────────────────────────────────────
  const riskScore = Math.round(prediction.overall_risk);
  const confidence = prediction.confidence_score;
  const hci = Math.round((100 - riskScore) * confidence / 100);

  const scanId = `HX-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;

  const scanQuality = useMemo(() => {
    const faceAlign = Math.min(100, Math.round(face.symmetry_score + 10));
    const lighting = Math.round(100 - face.brightness_variance);
    const audioClarity = Math.round(100 - voice.pitch_instability);
    return {
      faceAlign: Math.max(0, Math.min(100, faceAlign)),
      lighting: Math.max(0, Math.min(100, lighting)),
      audioClarity: Math.max(0, Math.min(100, audioClarity)),
      overall: Math.max(0, Math.min(100, Math.round((faceAlign + lighting + audioClarity) / 3))),
    };
  }, [face, voice]);

  const featureData = useMemo(() => {
    if (!prediction.feature_contribution) return [];
    const labelMap: Record<string, string> = {
      face_fatigue: "Facial Fatigue",
      symmetry_score: "Facial Symmetry",
      blink_instability: "Blink Instability",
      brightness_variance: "Skin Tone Variation",
      voice_stress: "Voice Stress",
      breathing_score: "Breathing Pattern",
      pitch_instability: "Pitch Instability",
      face_risk_score: "Face Risk Score",
      voice_risk_score: "Voice Risk Score",
    };
    return Object.entries(prediction.feature_contribution)
      .map(([key, value]) => ({
        name: labelMap[key] || key,
        value: Math.round(value * 1000),
        key,
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 7);
  }, [prediction.feature_contribution]);

  const primaryIndicator = featureData.length > 0 ? featureData[0] : null;

  const gaugeData = [
    {
      name: "Risk",
      value: riskScore,
      fill: riskScore < 40 ? "#10b981" : riskScore < 70 ? "#f59e0b" : "#ef4444",
    },
    { name: "Safe", value: 100 - riskScore, fill: "#f1f5f9" },
  ];

  const recommendations = useMemo(() => {
    const recs: { title: string; text: string; color: string; icon: typeof Heart }[] = [];
    if (riskScore < 40) {
      recs.push({
        title: "Maintain Healthy Lifestyle",
        text: "Your biomarkers indicate low risk. Continue maintaining a balanced diet, regular exercise, and adequate sleep.",
        color: "bg-emerald-500", icon: Heart,
      });
    } else if (riskScore < 70) {
      recs.push({
        title: "Improve Sleep & Stress Management",
        text: "Moderate risk detected. Focus on improving sleep quality (7-8 hours), practicing relaxation techniques, and reducing daily stressors.",
        color: "bg-amber-500", icon: AlertTriangle,
      });
    } else {
      recs.push({
        title: "Consider Medical Consultation",
        text: "Elevated risk indicators detected. We recommend scheduling a consultation with a healthcare professional for a comprehensive evaluation.",
        color: "bg-destructive", icon: ShieldAlert,
      });
    }
    if (face.face_fatigue > 50) {
      recs.push({
        title: "Address Fatigue Indicators",
        text: `Facial fatigue index at ${Math.round(face.face_fatigue)}%. Consider improving sleep schedule and taking regular breaks during screen time.`,
        color: "bg-amber-500", icon: Zap,
      });
    }
    if (voice.voice_stress > 50) {
      recs.push({
        title: "Manage Vocal Stress",
        text: `Voice stress detected at ${Math.round(voice.voice_stress)}%. Practice breathing exercises and ensure proper hydration.`,
        color: "bg-amber-500", icon: Mic,
      });
    }
    return recs;
  }, [riskScore, face, voice]);

  // ── Download report ────────────────────────────────────────
  const downloadReport = () => {
    const report = {
      title: "AI Silent Disease Predictor — Health Report",
      scan_id: scanId,
      model_version: prediction.model_version,
      generated: results.timestamp,
      face_analysis: face,
      voice_analysis: voice,
      prediction,
      health_confidence_index: hci,
      scan_quality: scanQuality,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `health-report-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const fadeIn = (delay = 0) => ({
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { delay, duration: 0.5 },
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-semibold text-primary bg-primary/10 px-3 py-1 rounded-full mb-3">
            <Sparkles className="w-3 h-3" /> AI PREVENTIVE HEALTH SCREENING
          </div>
          <h1 className="text-3xl font-bold mb-1">Health Analysis Dashboard</h1>
          <p className="text-muted-foreground text-sm">
            Report generated on {new Date(results.timestamp).toLocaleString()}
          </p>
        </div>
        <Button className="rounded-full shadow-lg gap-2" onClick={downloadReport}>
          <Download className="w-4 h-4" />
          Download Report
        </Button>
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 1 — BIOMARKER ANALYSIS                                 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.1)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">Biomarker Analysis</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Facial Biomarkers */}
          <div className="glass-card p-6 rounded-2xl">
            <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-5 flex items-center gap-2">
              <SmilePlus className="w-4 h-4 text-primary" /> Facial Biomarker Analysis
            </h3>
            <div className="space-y-4">
              {[
                { label: "Eye Stability", value: Math.round(100 - face.blink_instability), icon: Eye, color: "from-blue-500 to-blue-400" },
                { label: "Facial Symmetry", value: Math.round(face.symmetry_score), icon: SmilePlus, color: "from-emerald-500 to-emerald-400" },
                { label: "Fatigue Index", value: Math.round(100 - face.face_fatigue), icon: Zap, color: "from-amber-500 to-amber-400" },
                { label: "Skin Brightness Score", value: Math.round(100 - face.brightness_variance), icon: Sun, color: "from-purple-500 to-purple-400" },
              ].map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center shrink-0">
                    <m.icon className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span>{m.label}</span>
                      <span className="font-bold text-foreground">{m.value}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${m.value}%` }}
                        transition={{ duration: 1, delay: 0.2 + i * 0.1, ease: "easeOut" }}
                        className={`h-full rounded-full bg-linear-to-r ${m.color}`}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Voice Biomarkers */}
          <div className="glass-card p-6 rounded-2xl">
            <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-5 flex items-center gap-2">
              <Mic className="w-4 h-4 text-primary" /> Voice Biomarker Analysis
            </h3>
            <div className="space-y-4">
              {[
                { label: "Voice Stress", value: Math.round(100 - voice.voice_stress), icon: Mic, color: "from-rose-500 to-rose-400" },
                { label: "Breathing Pattern", value: Math.round(100 - voice.breathing_score), icon: Wind, color: "from-cyan-500 to-cyan-400" },
                { label: "Pitch Stability", value: Math.round(100 - voice.pitch_instability), icon: Music2, color: "from-indigo-500 to-indigo-400" },
              ].map((m, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center shrink-0">
                    <m.icon className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span>{m.label}</span>
                      <span className="font-bold text-foreground">{m.value}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${m.value}%` }}
                        transition={{ duration: 1, delay: 0.2 + i * 0.1, ease: "easeOut" }}
                        className={`h-full rounded-full bg-linear-to-r ${m.color}`}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Voice risk badge */}
            <div className="mt-6 p-3 rounded-xl bg-muted/50 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                <Gauge className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Voice Risk Score</p>
                <p className="text-lg font-bold">{Math.round(voice.voice_risk_score)}/100</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 2 — AI RISK ANALYSIS                                   */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.2)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <ShieldAlert className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">AI Risk Analysis</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              title: "Risk Score",
              value: `${riskScore}`,
              subtitle: "/100",
              icon: ShieldAlert,
              color: riskScore < 40 ? "text-emerald-500" : riskScore < 70 ? "text-amber-500" : "text-destructive",
              bg: riskScore < 40 ? "bg-emerald-50" : riskScore < 70 ? "bg-amber-50" : "bg-destructive/10",
              desc: riskScore < 40 ? "Low Risk" : riskScore < 70 ? "Moderate Risk" : "High Risk",
            },
            {
              title: "Confidence Score",
              value: `${confidence.toFixed(1)}`,
              subtitle: "%",
              icon: Activity,
              color: "text-primary",
              bg: "bg-primary/10",
              desc: "Model prediction confidence",
            },
            {
              title: "Health Confidence Index",
              value: `${hci}`,
              subtitle: "/100",
              icon: HeartPulse,
              color: "text-emerald-500",
              bg: "bg-emerald-50",
              desc: "HCI = (100 − Risk) × Confidence / 100",
            },
          ].map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 + i * 0.1 }}
              className="glass-card p-6 rounded-2xl border border-black/5 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-24 h-24 rounded-full bg-linear-to-br from-primary/5 to-transparent -translate-y-8 translate-x-8" />
              <div className={`p-3 rounded-xl ${m.bg} w-fit mb-4`}>
                <m.icon className={`w-6 h-6 ${m.color}`} />
              </div>
              <p className="text-muted-foreground font-medium text-sm mb-1">{m.title}</p>
              <h3 className="text-4xl font-bold tracking-tight">
                {m.value}
                <span className="text-lg text-muted-foreground font-normal">{m.subtitle}</span>
              </h3>
              <p className="text-xs text-muted-foreground mt-2">{m.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Drift warning */}
        {prediction.drift_warning && (
          <div className="mt-4 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 shrink-0" />
            <strong>Feature Drift Detected:</strong> Some biomarker values are outside the model&apos;s training distribution. Results may be less reliable.
          </div>
        )}
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 3 — RISK GAUGE + FEATURE CONTRIBUTION                  */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.3)}>
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Risk gauge */}
          <div className="glass-card p-6 rounded-2xl flex flex-col items-center justify-center text-center relative overflow-hidden">
            <h3 className="text-lg font-bold self-start w-full mb-4 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-primary" /> Risk Gauge
            </h3>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={gaugeData}
                    cx="50%"
                    cy="75%"
                    startAngle={180}
                    endAngle={0}
                    innerRadius="58%"
                    outerRadius="82%"
                    paddingAngle={0}
                    dataKey="value"
                    stroke="none"
                  >
                    {gaugeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="absolute bottom-14 flex flex-col items-center">
              <span className="text-5xl font-bold">{riskScore}</span>
              <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mt-1">
                {riskScore < 40 ? "Low Risk" : riskScore < 70 ? "Moderate Risk" : "High Risk"}
              </span>
            </div>
            <div className="flex w-full justify-between px-6 text-xs font-medium text-muted-foreground -mt-4">
              <span>0 (Low)</span>
              <span>100 (High)</span>
            </div>
          </div>

          {/* Feature contribution */}
          <div className="glass-card p-6 rounded-2xl lg:col-span-2">
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" /> Feature Contribution
              </h3>
            </div>
            {primaryIndicator && (
              <div className="mb-4 inline-flex items-center gap-2 text-xs font-semibold text-primary bg-primary/10 px-3 py-1 rounded-full">
                <Sparkles className="w-3 h-3" />
                Primary Indicator: {primaryIndicator.name}
              </div>
            )}
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={featureData}
                  layout="vertical"
                  margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                  <XAxis type="number" hide />
                  <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} width={130} />
                  <Tooltip
                    cursor={{ fill: "transparent" }}
                    contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1)" }}
                  />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={22}>
                    {featureData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? "#1a73e8" : "#93c5fd"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 4 — SCAN QUALITY SCORE                                 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.35)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <CheckCircle2 className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">Scan Quality Score</h2>
        </div>
        <div className="glass-card p-6 rounded-2xl">
          <div className="grid md:grid-cols-4 gap-6">
            {/* Overall */}
            <div className="flex flex-col items-center justify-center text-center p-4 rounded-xl bg-primary/5">
              <div className="text-5xl font-bold text-primary mb-1">{scanQuality.overall}%</div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Overall Quality</p>
            </div>
            {/* Breakdown */}
            {[
              { label: "Face Alignment", value: scanQuality.faceAlign, color: "bg-blue-500" },
              { label: "Lighting Quality", value: scanQuality.lighting, color: "bg-amber-500" },
              { label: "Audio Clarity", value: scanQuality.audioClarity, color: "bg-emerald-500" },
            ].map((q, i) => (
              <div key={i} className="space-y-2">
                <div className="flex justify-between text-sm font-medium">
                  <span>{q.label}</span>
                  <span className="font-bold">{q.value}%</span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${q.value}%` }}
                    transition={{ duration: 0.8, delay: 0.4 + i * 0.1 }}
                    className={`h-full rounded-full ${q.color}`}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {q.value >= 80 ? "Excellent" : q.value >= 60 ? "Good" : q.value >= 40 ? "Fair" : "Poor"}
                </p>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 5 — AI RECOMMENDATIONS                                 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.4)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Heart className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">AI Recommendations</h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {recommendations.map((rec, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45 + i * 0.1 }}
              className="glass-card p-5 rounded-2xl border border-black/5 relative overflow-hidden hover:shadow-lg transition-shadow"
            >
              <div className={`absolute left-0 top-0 bottom-0 w-1 ${rec.color}`} />
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg ${rec.color === "bg-emerald-500" ? "bg-emerald-50" : rec.color === "bg-amber-500" ? "bg-amber-50" : "bg-destructive/10"}`}>
                  <rec.icon className={`w-4 h-4 ${rec.color === "bg-emerald-500" ? "text-emerald-500" : rec.color === "bg-amber-500" ? "text-amber-500" : "text-destructive"}`} />
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">{rec.title}</h4>
                  <p className="text-xs text-muted-foreground leading-relaxed">{rec.text}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 6 — SCAN METADATA                                      */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.45)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <FileText className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">Scan Metadata</h2>
        </div>
        <div className="glass-card p-6 rounded-2xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center">
                <Hash className="w-5 h-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground font-medium">Scan ID</p>
                <p className="font-mono font-bold text-sm">{scanId}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center">
                <BrainCircuit className="w-5 h-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground font-medium">Model Version</p>
                <p className="font-mono font-bold text-sm">{prediction.model_version}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center">
                <Clock className="w-5 h-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground font-medium">Scan Timestamp</p>
                <p className="font-mono font-bold text-sm">
                  {new Date(results.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* SECTION 7 — PRIVACY NOTICE                                     */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.5)}>
        <div className="glass-card p-6 rounded-2xl border-2 border-emerald-200/50 bg-emerald-50/30">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-emerald-100 shrink-0">
              <Lock className="w-6 h-6 text-emerald-600" />
            </div>
            <div>
              <h3 className="font-bold text-base mb-1 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                Privacy Assurance
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                All biometric analysis is performed locally on your device.
                <br />
                No facial images or voice recordings are stored or transmitted to external servers.
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* DATASET TRANSPARENCY SECTION                                   */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.55)}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Database className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-xl font-bold">Dataset Transparency</h2>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-6">
          <div className="p-4 rounded-xl bg-primary/5 border border-primary/10">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-primary mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-medium mb-1">Synthetic Dataset Methodology</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  The training dataset was synthetically generated using statistical distributions inspired by real biomedical datasets.
                  This approach ensures privacy while maintaining scientifically grounded feature correlations.
                  For example: higher fatigue + higher voice stress → higher probability of elevated risk;
                  higher facial symmetry + stable pitch → lower predicted risk.
                  Gaussian noise was applied to add realistic variability.
                </p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {datasets.map((ds, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="p-4 rounded-xl border border-black/5 bg-white/60 hover:bg-white transition-colors group"
              >
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 text-xs font-bold text-primary">
                    {i + 1}
                  </div>
                  <div className="min-w-0">
                    <h4 className="font-semibold text-sm mb-1 group-hover:text-primary transition-colors flex items-center gap-1">
                      {ds.name}
                      <ChevronRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    <p className="text-xs text-muted-foreground mb-2">{ds.purpose}</p>
                    <p className="text-xs text-foreground/70 mb-2">{ds.usage}</p>
                    <div className="flex flex-wrap gap-1">
                      {ds.features.map((f, j) => (
                        <span key={j} className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-medium">
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* IMPORTANT LIMITATION DISCLAIMER                                */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <motion.div {...fadeIn(0.6)}>
        <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200/50 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-amber-800 mb-1">Important Disclaimer</p>
            <p className="text-xs text-amber-700 leading-relaxed">
              This system provides <strong>AI Preventive Health Screening</strong> only.
              It does not provide medical diagnosis, treatment advice, or replace professional medical consultation.
              Always consult a qualified healthcare professional for any health concerns.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Footer */}
      <footer className="py-8 border-t border-black/5 flex flex-col md:flex-row items-center justify-between text-xs text-muted-foreground gap-4">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-4 h-4" />
          <span>Core Model v{prediction.model_version} · Risk Level: {prediction.risk_level}</span>
        </div>
        <div className="flex items-center gap-1.5 bg-muted px-3 py-1 rounded-full font-medium">
          <ShieldCheck className="w-3 h-3" />
          All biometric data processed locally. No personal data stored.
        </div>
      </footer>
    </div>
  );
}
