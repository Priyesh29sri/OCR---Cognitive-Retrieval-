# ICDI-X Frontend

Modern Next.js frontend for the ICDI-X intelligent document processing system.

## Features

- 🎨 Beautiful animated UI with particle flow field background
- 💬 Real-time AI chat interface with command palette
- 📄 Document upload with progress tracking
- 🔐 Authentication (login/register)
- ⚡ Real-time query responses from ICDI-X backend
- 🎭 Framer Motion animations throughout
- 🎯 TypeScript for type safety
- 🎨 Tailwind CSS + shadcn/ui components

## Prerequisites

- Node.js 18+ 
- ICDI-X backend running on http://127.0.0.1:8000

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Install additional required dependency:
```bash
npm install @radix-ui/react-slot tailwindcss-animate
```

## Development

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Build

Build for production:

```bash
npm run build
```

Start production server:

```bash
npm start
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with Inter font
│   ├── page.tsx            # Main application page
│   └── globals.css         # Global styles with Tailwind
├── components/
│   └── ui/
│       ├── button.tsx              # Base button component
│       ├── file-trigger.tsx        # File upload button
│       ├── animated-ai-chat.tsx    # Chat interface
│       └── flow-field-background.tsx  # Particle background
├── lib/
│   ├── utils.ts            # Utility functions (cn helper)
│   └── api.ts              # Backend API integration
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

## Components

### AnimatedAIChat
Full-featured chat interface with:
- Message history with animations
- Typing indicators
- Command palette (press `/`)
- File attachments
- Real-time responses

### FileTriggerButton
React Aria Components file upload with:
- Progress tracking
- Multiple file types support
- Beautiful button styling

### FlowFieldBackground
Canvas-based particle animation with:
- Flow field physics
- Mouse interaction
- Perlin noise-like movement
- Connection lines between particles

## API Integration

The frontend connects to the ICDI-X backend at `http://127.0.0.1:8000`:

- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `POST /upload` - Document upload
- `POST /query` - Query documents
- `WS /chat` - WebSocket chat (future feature)

## Usage

1. **Sign In/Register**: Click "Sign In" to create an account or log in
2. **Upload Document**: Click "Upload Document" to add PDF/DOC/TXT files
3. **Ask Questions**: Type questions in the chat interface
4. **Use Commands**: Press `/` for quick commands like `/analyze`, `/search`, `/summarize`

## Environment Variables

Create a `.env.local` file if you need to customize:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Troubleshooting

### Port 3000 already in use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
npm run dev -- -p 3001
```

### Backend connection errors
- Ensure ICDI-X backend is running: `http://127.0.0.1:8000/`
- Check CORS settings in backend if requests are blocked

### Build errors
```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: shadcn/ui + React Aria Components
- **Animation**: Framer Motion
- **Icons**: Lucide React
- **HTTP Client**: Fetch API with XMLHttpRequest for uploads

## License

Part of the ICDI-X project.
