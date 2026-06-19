import type { Config } from 'tailwindcss';
import colors from 'tailwindcss/colors';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: colors.indigo,
        accent: colors.amber,
        glass: 'rgba(255,255,255,0.15)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
} satisfies Config;