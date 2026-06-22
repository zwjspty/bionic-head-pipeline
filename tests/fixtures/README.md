# Real-provider fixtures

Default tests do not require files in this directory. Real deployment validation uses environment variables instead:

- `BIONIC_CONFIG`: local JSON config, usually `config/local.json`.
- `BIONIC_TEST_WAV`: at least one Chinese speech WAV; mono PCM16, 16 kHz.
- `BIONIC_TEST_TTS_WAV`: a valid Chinese response WAV for isolated Morpheus validation.

For Morpheus/UE5 acceptance, keep reference artifacts outside Git unless they are tiny:

- one normal Morpheus output shaped `[N, 52]`;
- one existing UE5 JSON sample;
- notes for the eventual Morpheus-52-to-UE5 curve-name mapping.
