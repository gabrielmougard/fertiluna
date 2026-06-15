import { describe, it, expect } from "vitest";
import { chooseBackend } from "./backendRouter";
import type { RouterConfig } from "./types";

const base: RouterConfig = {
  cloudEnabled: true,
  cloudShare: 0.9,
  consentGiven: true,
};

describe("chooseBackend (consent + kill-switch gating)", () => {
  it("stays on-device when cloud is disabled (kill switch)", () => {
    expect(chooseBackend({ ...base, cloudEnabled: false }, () => 0)).toBe(
      "on-device",
    );
  });
  it("stays on-device without consent, even if cloud enabled", () => {
    expect(chooseBackend({ ...base, consentGiven: false }, () => 0)).toBe(
      "on-device",
    );
  });
  it("routes to cloud below the share threshold when consented", () => {
    expect(chooseBackend(base, () => 0.5)).toBe("cloud"); // 0.5 < 0.9
  });
  it("routes on-device above the share threshold", () => {
    expect(chooseBackend(base, () => 0.95)).toBe("on-device"); // 0.95 ≥ 0.9
  });
  it("respects a 0 share (never cloud) even with consent", () => {
    expect(chooseBackend({ ...base, cloudShare: 0 }, () => 0)).toBe(
      "on-device",
    );
  });
});
