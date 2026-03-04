import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";

const PYTHON_API = process.env.PYTHON_API_URL || "http://127.0.0.1:5001";

/**
 * Proxy helper — forwards JSON body to the Python Flask API and
 * pipes the response back to the client.
 */
async function proxyToFlask(
  endpoint: string,
  body: unknown,
  res: Response
): Promise<void> {
  try {
    const upstream = await fetch(`${PYTHON_API}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err: any) {
    console.error(`[proxy] ${endpoint} error:`, err.message);
    res
      .status(502)
      .json({ error: "Python API unavailable", detail: err.message });
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // ── Health check (Python API) ──
  app.get("/api/health", async (_req: Request, res: Response) => {
    try {
      const upstream = await fetch(`${PYTHON_API}/api/health`);
      const data = await upstream.json();
      res.json(data);
    } catch {
      res.status(502).json({ error: "Python API offline" });
    }
  });

  // ── Face scan ──
  app.post("/api/face-scan", async (req: Request, res: Response) => {
    await proxyToFlask("/api/face-scan", req.body, res);
  });

  // ── Voice scan ──
  app.post("/api/voice-scan", async (req: Request, res: Response) => {
    await proxyToFlask("/api/voice-scan", req.body, res);
  });

  // ── ML prediction ──
  app.post("/api/predict", async (req: Request, res: Response) => {
    await proxyToFlask("/api/predict", req.body, res);
  });

  // ── Full scan (face + voice + predict in one call) ──
  app.post("/api/full-scan", async (req: Request, res: Response) => {
    await proxyToFlask("/api/full-scan", req.body, res);
  });

  return httpServer;
}
