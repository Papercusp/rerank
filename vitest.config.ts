import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    provide: { papercuspTestLayer: 'unit' },
  },
});

