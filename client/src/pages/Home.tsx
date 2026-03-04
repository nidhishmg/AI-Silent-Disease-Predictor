import { Link } from "wouter";
import { motion } from "framer-motion";
import { ArrowRight, ScanFace, Mic, BrainCircuit, ActivitySquare } from "lucide-react";
import heroGraphic from "@/assets/hero-graphic.png";
import { Button } from "@/components/ui/button";

const steps = [
  {
    icon: ScanFace,
    title: "Face Scan",
    description: "Extracting optical biomarkers and micro-expressions."
  },
  {
    icon: Mic,
    title: "Voice Scan",
    description: "Analyzing vocal cord variations and breathing patterns."
  },
  {
    icon: BrainCircuit,
    title: "AI Analysis",
    description: "Processing multi-modal data through our neural engine."
  },
  {
    icon: ActivitySquare,
    title: "Health Insights",
    description: "Delivering actionable preventive healthcare metrics."
  }
];

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center">
      {/* Hero Section */}
      <section className="w-full max-w-7xl mx-auto px-4 py-20 lg:py-32 grid lg:grid-cols-2 gap-12 items-center">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="flex flex-col items-start gap-8"
        >
          <div className="inline-flex items-center rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-sm font-medium text-primary mb-4">
            <span className="flex h-2 w-2 rounded-full bg-primary mr-2 animate-pulse"></span>
            v2.4 Core Model Online
          </div>
          <h1 className="text-5xl lg:text-7xl font-bold font-display tracking-tight text-foreground leading-[1.1]">
            AI Silent Disease <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">
              Predictor
            </span>
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed max-w-xl">
            AI-powered preventive healthcare using face and voice biomarker analysis. Detect subtle health changes before symptoms appear.
          </p>
          <div className="flex gap-4 pt-4">
            <Link href="/scan">
              <Button size="lg" className="h-14 px-8 text-lg rounded-full premium-shadow group">
                Start Health Scan
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="relative flex justify-center items-center"
        >
          <div className="absolute inset-0 bg-gradient-to-tr from-primary/20 to-transparent blur-3xl rounded-full opacity-60"></div>
          <motion.img 
            animate={{ y: [-10, 10, -10] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
            src={heroGraphic} 
            alt="AI Scanning Brain" 
            className="w-full max-w-md relative z-10 drop-shadow-2xl"
          />
        </motion.div>
      </section>

      {/* How it works */}
      <section className="w-full bg-white/50 border-t border-black/5 py-24 mt-auto">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">The Scanning Process</h2>
            <p className="text-muted-foreground">Four simple steps to your comprehensive health analysis</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card p-6 rounded-2xl group hover:-translate-y-1 transition-transform duration-300"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center mb-6 group-hover:bg-primary group-hover:text-white transition-colors duration-300">
                  <step.icon className="w-6 h-6" />
                </div>
                <div className="text-sm font-semibold text-primary/80 mb-2 tracking-wider uppercase">Step {i + 1}</div>
                <h3 className="text-xl font-bold mb-2">{step.title}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}