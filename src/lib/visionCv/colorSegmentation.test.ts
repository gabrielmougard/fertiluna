import { describe, it, expect } from "vitest";
import { __test, segment } from "./colorSegmentation";
import { HSV_BLUE, HSV_ORANGE, HSV_PURPLE } from "./constants";
import type { WorkImage } from "./preprocess";

const { rgbToHsv, inBand } = __test;

describe("rgbToHsv (OpenCV 8-bit convention)", () => {
  it("maps pure red to H≈0", () => {
    const [h, s, v] = rgbToHsv(255, 0, 0);
    expect(h).toBeLessThanOrEqual(2);
    expect(s).toBe(255);
    expect(v).toBe(255);
  });
  it("maps pure green to H≈60 (120°/2)", () => {
    const [h] = rgbToHsv(0, 255, 0);
    expect(Math.abs(h - 60)).toBeLessThanOrEqual(1);
  });
  it("maps pure blue to H≈120 (240°/2)", () => {
    const [h] = rgbToHsv(0, 0, 255);
    expect(Math.abs(h - 120)).toBeLessThanOrEqual(1);
  });
  it("white has zero saturation", () => {
    const [, s] = rgbToHsv(255, 255, 255);
    expect(s).toBe(0);
  });
});

describe("classification matches the Premom palette", () => {
  it("BBT blue #95aeff falls in the blue band only", () => {
    const [h, s, v] = rgbToHsv(0x95, 0xae, 0xff);
    expect(inBand(h, s, v, HSV_BLUE)).toBe(true);
    expect(inBand(h, s, v, HSV_ORANGE)).toBe(false);
    expect(inBand(h, s, v, HSV_PURPLE)).toBe(false);
  });
  it("LH orange #ff9e8d falls in the orange band only", () => {
    const [h, s, v] = rgbToHsv(0xff, 0x9e, 0x8d);
    expect(inBand(h, s, v, HSV_ORANGE)).toBe(true);
    expect(inBand(h, s, v, HSV_BLUE)).toBe(false);
    expect(inBand(h, s, v, HSV_PURPLE)).toBe(false);
  });
  it("Level purple #9e6fe3 falls in the purple band only", () => {
    const [h, s, v] = rgbToHsv(0x9e, 0x6f, 0xe3);
    expect(inBand(h, s, v, HSV_PURPLE)).toBe(true);
    expect(inBand(h, s, v, HSV_BLUE)).toBe(false);
    expect(inBand(h, s, v, HSV_ORANGE)).toBe(false);
  });
  it("white markerfill / pale bands classify as none", () => {
    for (const [r, g, b] of [[255, 255, 255], [239, 239, 244]] as const) {
      const [h, s, v] = rgbToHsv(r, g, b);
      expect(inBand(h, s, v, HSV_BLUE)).toBe(false);
      expect(inBand(h, s, v, HSV_ORANGE)).toBe(false);
      expect(inBand(h, s, v, HSV_PURPLE)).toBe(false);
    }
  });
});

describe("segment over a tiny RGBA buffer", () => {
  it("routes each pixel to the right mask", () => {
    // 3×1 image: blue, orange, purple
    const px = [
      [0x95, 0xae, 0xff],
      [0xff, 0x9e, 0x8d],
      [0x9e, 0x6f, 0xe3],
    ];
    const rgba = new Uint8ClampedArray(3 * 4);
    px.forEach(([r, g, b], i) => {
      rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255;
    });
    const img: WorkImage = { rgba, width: 3, height: 1, scale: 1 };
    const m = segment(img);
    expect(Array.from(m.blue.data)).toEqual([255, 0, 0]);
    expect(Array.from(m.orange.data)).toEqual([0, 255, 0]);
    expect(Array.from(m.purple.data)).toEqual([0, 0, 255]);
  });
});
