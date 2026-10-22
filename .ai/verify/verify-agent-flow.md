# Verify Agent Flow

Validate the primary path:

`prompt -> tokenizer -> scheduler -> model runner -> attention/KV cache -> sampler -> generated text`

Capture evidence that:

- package import and configuration succeed;
- a compatible model loads;
- the scheduler and cache path execute;
- generated output returns through `LLM.generate`;
- GPU, dependency, model, and command details are recorded.

Do not mark runtime behavior done when only syntax checks ran.

