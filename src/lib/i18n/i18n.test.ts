import { describe, it, expect } from "vitest";
import { detectLocale, toLocale, t, languageName } from "./index";

describe("toLocale", () => {
  it("normalizes regional tags to base locale", () => {
    expect(toLocale("fr-CA")).toBe("fr");
    expect(toLocale("en-US")).toBe("en");
    expect(toLocale("de-DE")).toBe("fr"); // unsupported → default (fr)
    expect(toLocale(null)).toBe("fr");
  });
});

describe("detectLocale", () => {
  it("override wins", () => {
    expect(detectLocale({ override: "en", country: "FR" })).toBe("en");
  });
  it("Accept-Language beats country", () => {
    expect(
      detectLocale({ acceptLanguage: "en-GB,en;q=0.9", country: "FR" }),
    ).toBe("en");
  });
  it("falls back to country hint when no language header", () => {
    expect(detectLocale({ country: "FR" })).toBe("fr");
    expect(detectLocale({ country: "US" })).toBe("fr"); // default is fr
  });
});

describe("t + languageName", () => {
  it("translates and falls back", () => {
    expect(t("consent.accept", "en")).toMatch(/Allow/);
    expect(t("consent.accept", "fr")).toMatch(/Autoriser/);
  });
  it("language name for the LLM instruction", () => {
    expect(languageName("fr")).toBe("French");
    expect(languageName("en")).toBe("English");
  });
});
