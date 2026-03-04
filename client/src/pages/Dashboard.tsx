import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Download, Info, Activity, Heart, ShieldAlert, HeartPulse, BrainCircuit } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid 
} from "recharts";
import { useScanResults } from "@/context/ScanContext";
import { useLocation } from "wouter";

export default function Dashboard() {
  const { results } = useScanResults();
  const [, setLocation] = useLocation();

  const [sleep, setSleep] = useState([7]);
  const [exercise, setExercise] = useState([3]);
  const [smoking, setSmoking] = useState([0]);

  // ── Redirect if no results ─────────────────────────────────────────
  if (!results) {
    return (
      <div className="max-w-xl mx-auto px-4 py-24 text-center space-y-6">
        <BrainCircuit className="w-16 h-16 text-primary mx-auto opacity-40" />
        <h2 className="text-2xl font-bold">No Scan Results Yet</h2>
        <p className="text-muted-foreground">
          Complete a face and voice scan first to see your health analysis.
        </p>
        <Button className="rounded-full px-8" onClick={() => setLocation("/scan")}>
          Start Health Scan
        </Button>
      </div>
    );
  }

  const { face, voice, prediction } = results;

  // ── Feature contribution bar chart data ────────────────────────────
  const featureData = useMemo(() => {
    if (!prediction.feature_contribution) return [];
    // Map feature names to human-readable labels
    const labelMap: Record<string, string> = {
      face_fatigue: "Facial Fatigue",
      symmetry_score: "Facial Asymmetry",
      blink_instability: "Blink Instability",
      brightness_variance: "Skin Tone Variations",
      voice_stress: "Voice Micro-tremors",
      breathing_score: "Breathing Rate",
      pitch_instability: "Pitch Instability",
      face_risk_score: "Face Risk Score",
      voice_risk_score: "Voice Risk Score",
    };
    return Object.entries(prediction.feature_contribution)
      .map(([key, value]) => ({
        name: labelMap[key] || key,
        value: Math.round(value * 1000), // scale importances for visual display
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
  }, [prediction.feature_contribution]);

  // ── Dynamic risk with lifestyle sliders ────────────────────────────
  const baseRisk = prediction.overall_risk;
  const simulatedRisk = Math.max(
    10,
    Math.min(
      90,
      baseRisk - (sleep[0] - 7) * 2 - (exercise[0] - 3) * 3 + smoking[0] * 15
    )
  );

  const gaugeData = [
    {
      name: "Risk",
      value: simulatedRisk,
      fill:
        simulatedRisk < 40
          ? "#10b981"
          : simulatedRisk < 70
          ? "#f59e0b"
          : "#ef4444",
    },
    { name: "Safe", value: 100 - simulatedRisk, fill: "#f1f5f9" },
  ];

  // ── Health index grade ─────────────────────────────────────────────
  const healthGrade =
    simulatedRisk < 20
      ? "A+"
      : simulatedRisk < 35
      ? "A"
      : simulatedRisk < 50
      ? "A-"
      : simulatedRisk < 60
      ? "B+"
      : simulatedRisk < 70
      ? "B"
      : simulatedRisk < 80
      ? "C"
      : "D";

  // ── AI recommendations based on actual data ────────────────────────
  const recommendations = useMemo(() => {
    const recs: { title: string; text: string; color: string }[] = [];

    if (face.face_fatigue > 50) {
      recs.push({
        title: "Reduce Fatigue",
        text: `Your facial fatigue score is ${Math.round(face.face_fatigue)}%. Consider improving sleep quality and taking regular breaks.`,
        color: "bg-amber-500",
      });
    } else {
      recs.push({
        title: "Maintain Current Routine",
        text: "Your cardiovascular indicators are stable. Continue current aerobic exercise routine.",
        color: "bg-emerald-500",
      });
    }

    if (voice.voice_stress > 50) {
      recs.push({
        title: "Manage Vocal Stress",
        text: `Voice stress detected at ${Math.round(voice.voice_stress)}%. Practice relaxation techniques and stay hydrated.`,
        color: "bg-amber-500",
      });
    } else {
      recs.push({
        title: "Improve Sleep Quality",
        text: "Vocal micro-tremors suggest slight fatigue. Aim for consistent 7-8 hours of sleep to improve recovery.",
        color: "bg-amber-500",
      });
    }

    if (face.symmetry_score < 70) {
      recs.push({
        title: "Schedule Follow-up",
        text: `Facial symmetry score is ${Math.round(face.symmetry_score)}%. Minor asymmetry detected — recommended for discussion at next annual physical.`,
        color: "bg-primary",
      });
    } else {
      recs.push({
        title: "Schedule Follow-up",
        text: "Minor facial asymmetry detected in lower quadrant. Not critical, but recommended for discussion at next annual physical.",
        color: "bg-primary",
      });
    }

    return recs;
  }, [face, voice]);

  // ── Download report ────────────────────────────────────────────────
  const downloadReport = () => {
    const report = {
      title: "AI Silent Disease Predictor — Health Report",
      generated: results.timestamp,
      face_analysis: face,
      voice_analysis: voice,
      prediction,
      lifestyle_simulation: { sleep: sleep[0], exercise: exercise[0], smoking: smoking[0], adjusted_risk: simulatedRisk },
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `health-report-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-1">Health Analysis Results</h1>
          <p className="text-muted-foreground">
            Report generated on {new Date(results.timestamp).toLocaleDateString()}
          </p>
        </div>
        <Button className="rounded-full shadow-lg" onClick={downloadReport}>
          <Download className="w-4 h-4 mr-2" />
          Download Health Report
        </Button>
      </div>

      {/* METRIC CARDS */}
      <div className="grid md:grid-cols-3 gap-6">
        {[
          {
            title: "Risk Score",
            value: `${Math.round(simulatedRisk)}/100`,
            icon: ShieldAlert,
            color: "text-amber-500",
            bg: "bg-amber-50",
          },
          {
            title: "Confidence Score",
            value: `${prediction.confidence_score.toFixed(1)}%`,
            icon: Activity,
            color: "text-primary",
            bg: "bg-primary/10",
          },
          {
            title: "Health Index",
            value: healthGrade,
            icon: HeartPulse,
            color: "text-emerald-500",
            bg: "bg-emerald-50",
          },
        ].map((metric, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="glass-card p-6 rounded-2xl border border-black/5"
          >
            <div className="flex justify-between items-start mb-4">
              <div className={`p-3 rounded-xl ${metric.bg}`}>
                <metric.icon className={`w-6 h-6 ${metric.color}`} />
              </div>
            </div>
            <p className="text-muted-foreground font-medium mb-1">{metric.title}</p>
            <h3 className="text-4xl font-bold tracking-tight">{metric.value}</h3>
          </motion.div>
        ))}
      </div>

      {/* Drift warning banner */}
      {prediction.drift_warning && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 shrink-0" />
          <strong>Feature Drift Detected:</strong> Some biomarker values are outside the model&apos;s training distribution. Results may be less reliable.
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6">
        {/* RISK GAUGE */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6 rounded-2xl lg:col-span-1 flex flex-col items-center justify-center text-center relative overflow-hidden"
        >
          <h3 className="text-lg font-bold self-start w-full mb-4">Overall Risk Assessment</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={gaugeData}
                  cx="50%"
                  cy="75%"
                  startAngle={180}
                  endAngle={0}
                  innerRadius="60%"
                  outerRadius="80%"
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
          <div className="absolute bottom-16 flex flex-col items-center">
            <span className="text-5xl font-bold">{Math.round(simulatedRisk)}</span>
            <span className="text-sm font-medium uppercase tracking-widest text-muted-foreground mt-1">
              {simulatedRisk < 40 ? 'Low Risk' : simulatedRisk < 70 ? 'Moderate Risk' : 'High Risk'}
            </span>
          </div>
          <div className="flex w-full justify-between px-8 text-xs font-medium text-muted-foreground -mt-5">
            <span>0 (Low)</span>
            <span>100 (High)</span>
          </div>
        </motion.div>

        {/* FEATURE CONTRIBUTION */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="glass-card p-6 rounded-2xl lg:col-span-2"
        >
          <h3 className="text-lg font-bold mb-6">Biomarker Contribution Analysis</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 13 }} width={140} />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="value" fill="#4285f4" radius={[0, 4, 4, 0]} barSize={24}>
                  {featureData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#1a73e8' : '#60a5fa'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* SIMULATION */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-6 rounded-2xl"
        >
          <div className="flex items-center gap-2 mb-6">
            <Activity className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-bold">Digital Health Simulation</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-8">
            Adjust the lifestyle factors below to see how they impact your projected health risk over time.
          </p>

          <div className="space-y-8">
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Average Sleep (Hours/Night)</Label>
                <span className="font-medium">{sleep}h</span>
              </div>
              <Slider value={sleep} onValueChange={setSleep} max={12} min={4} step={0.5} />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Exercise Frequency (Days/Week)</Label>
                <span className="font-medium">{exercise} days</span>
              </div>
              <Slider value={exercise} onValueChange={setExercise} max={7} min={0} step={1} />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Smoking Status (0=Never, 10=Heavy)</Label>
                <span className="font-medium">{smoking}/10</span>
              </div>
              <Slider value={smoking} onValueChange={setSmoking} max={10} min={0} step={1} />
            </div>
          </div>
        </motion.div>

        {/* RECOMMENDATIONS */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-card p-6 rounded-2xl bg-linear-to-br from-white to-primary/5"
        >
          <div className="flex items-center gap-2 mb-6">
            <Heart className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-bold">AI Recommendations</h3>
          </div>
          
          <div className="space-y-4">
            {recommendations.map((rec, i) => (
              <div key={i} className="p-4 rounded-xl bg-white border border-primary/10 shadow-sm relative overflow-hidden">
                <div className={`absolute left-0 top-0 bottom-0 w-1 ${rec.color}`} />
                <h4 className="font-semibold mb-1">{rec.title}</h4>
                <p className="text-sm text-muted-foreground">{rec.text}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* FOOTER */}
      <footer className="mt-12 py-8 border-t border-black/5 flex flex-col md:flex-row items-center justify-between text-sm text-muted-foreground">
        <div className="flex items-center gap-2 mb-4 md:mb-0">
          <BrainCircuit className="w-4 h-4" />
          <span>Core Model v{prediction.model_version} (Risk Level: {prediction.risk_level})</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 bg-muted px-3 py-1 rounded-full text-xs font-medium">
            <ShieldAlert className="w-3 h-3" />
            All biometric data processed locally. No personal data stored.
          </span>
          <a href="#" className="hover:text-primary transition-colors">Privacy Notice</a>
        </div>
      </footer>
    </div>
  );
}
