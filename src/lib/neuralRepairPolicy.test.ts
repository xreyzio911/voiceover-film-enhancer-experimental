import assert from "node:assert/strict";
import test from "node:test";
import {
  CLEARVOICE_SE_REQUEST,
  CLEARVOICE_SE_VARIANT,
  formatNeuralRepairCandidate,
  isNeuralRepairCandidateVariant,
  neuralRepairRequestForVariant,
  resolveNeuralRepairRequestInput,
} from "./neuralRepairPolicy.ts";

test("neural speech enhancement is the only neural repair variant", () => {
  assert.equal(isNeuralRepairCandidateVariant(CLEARVOICE_SE_VARIANT), true);
  assert.equal(isNeuralRepairCandidateVariant("clearvoice-enhance"), false);
});

test("neural speech enhancement request uses speech enhancement only", () => {
  assert.deepEqual(neuralRepairRequestForVariant(CLEARVOICE_SE_VARIANT), {
    variant: "clearvoice-se",
    engine: "clearvoice",
    mode: "speech_enhancement",
    model: "MossFormer2_SE_48K",
    description: "Neural speech enhancement",
  });
  assert.equal(formatNeuralRepairCandidate(CLEARVOICE_SE_REQUEST.variant), "clearvoice-se");
});

test("client-supplied neural model text cannot change the fixed speech enhancement profile", () => {
  assert.deepEqual(
    resolveNeuralRepairRequestInput({
      engine: "clearvoice",
      mode: "speech_enhancement",
      model: "UnexpectedModel",
    }),
    CLEARVOICE_SE_REQUEST,
  );
  assert.equal(resolveNeuralRepairRequestInput({ variant: "unsupported-model" }), null);
});
