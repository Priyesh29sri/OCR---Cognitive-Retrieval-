'use client';

import React, { useState } from 'react';
import { Upload, FileText, Sparkles, Brain, BookOpen, Zap, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { FlowFieldBackground } from '@/components/ui/flow-field-background';
import { AnimatedAIChat } from '@/components/ui/animated-ai-chat';
import { FileTriggerButton } from '@/components/ui/file-trigger';
import { cn } from '@/lib/utils';
import { uploadDocument, query, getInsights, getStudyGuide, InsightsResponse, StudyGuideResponse } from '@/lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: Array<{
    id: string;
    name: string;
    type: string;
    size: number;
  }>;
}

interface UploadedFile {
  id: string;
  name: string;
}

// ── Insights Panel ──────────────────────────────────────────────────────────
function InsightsPanel({
  insights,
  onQuestionClick,
}: {
  insights: InsightsResponse;
  onQuestionClick: (q: string) => void;
}) {
  const [showStudyGuide, setShowStudyGuide] = useState(false);
  const [studyGuide, setStudyGuide] = useState<StudyGuideResponse | null>(null);
  const [loadingGuide, setLoadingGuide] = useState(false);

  const handleLoadGuide = async () => {
    if (studyGuide) { setShowStudyGuide((v) => !v); return; }
    setLoadingGuide(true);
    try {
      const guide = await getStudyGuide(insights.doc_id);
      setStudyGuide(guide);
      setShowStudyGuide(true);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingGuide(false);
    }
  };

  const complexityColor = {
    basic: 'text-green-400',
    intermediate: 'text-yellow-400',
    advanced: 'text-red-400',
    unknown: 'text-gray-400',
  }[insights.complexity] ?? 'text-gray-400';

  return (
    <div className="mt-4 rounded-xl border border-blue-500/30 bg-blue-500/5 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-blue-500/20 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-blue-400" />
          <span className="text-sm font-semibold text-blue-300">Proactive Insights</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300">
            {insights.doc_type?.replace(/_/g, ' ')}
          </span>
          <span className={cn('text-xs font-medium', complexityColor)}>
            {insights.complexity}
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          IB selected {insights.chunks_selected_by_ib}/{insights.chunks_analyzed} chunks
        </span>
      </div>

      <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Key Insights */}
        {insights.insights.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              🔍 Key Insights
            </p>
            <ul className="space-y-1.5">
              {insights.insights.map((ins, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <span className="text-blue-400 mt-0.5 flex-shrink-0">▸</span>
                  <span className="text-foreground/80">{ins}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Suggested Questions */}
        {insights.suggested_questions.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              💡 Ask me…
            </p>
            <div className="flex flex-wrap gap-2">
              {insights.suggested_questions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => onQuestionClick(q)}
                  className="text-xs px-3 py-1.5 rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 transition-colors text-left"
                >
                  {q}
                </button>
              ))}
            </div>

            {/* Key entities */}
            {insights.key_entities.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">
                  🏷️ Entities
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {insights.key_entities.slice(0, 8).map((e, i) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-0.5 rounded-md bg-muted/40 text-muted-foreground border border-border/50"
                    >
                      {e}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Study Guide toggle */}
      <div className="px-4 pb-3">
        <button
          onClick={handleLoadGuide}
          disabled={loadingGuide}
          className="flex items-center gap-2 text-xs text-yellow-400 hover:text-yellow-300 transition-colors font-medium"
        >
          <BookOpen className="w-3.5 h-3.5" />
          {loadingGuide
            ? 'Generating study guide…'
            : showStudyGuide
            ? 'Hide Study Guide'
            : '📚 Generate Bloom\'s Study Guide'}
          {!loadingGuide && (showStudyGuide ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
        </button>

        {showStudyGuide && studyGuide && (
          <div className="mt-3 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3">
            <p className="text-sm font-semibold text-yellow-300 mb-2">{studyGuide.title}</p>
            <p className="text-xs text-muted-foreground mb-3">{studyGuide.summary}</p>

            {/* Bloom's questions — show first 2 levels */}
            {(['remember', 'understand', 'apply'] as const).map((level) => {
              const qs = studyGuide.blooms_questions?.[level] ?? [];
              if (!qs.length) return null;
              return (
                <div key={level} className="mb-2">
                  <p className="text-xs font-semibold text-yellow-400/80 capitalize mb-1">
                    L{['remember','understand','apply','analyze','evaluate','create'].indexOf(level)+1} — {level}
                  </p>
                  {qs.slice(0, 2).map((q, i) => (
                    <button
                      key={i}
                      onClick={() => onQuestionClick(q)}
                      className="block text-xs text-left text-foreground/70 hover:text-foreground py-0.5 transition-colors"
                    >
                      · {q}
                    </button>
                  ))}
                </div>
              );
            })}

            <p className="text-xs text-muted-foreground mt-2">
              Est. study time: {studyGuide.estimated_study_time_minutes} min ·{' '}
              {studyGuide.key_concepts.length} key concepts
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);
  const [loadingInsights, setLoadingInsights] = useState(false);

  // Called by both the header Upload button and the paperclip inside chat
  const handleFileSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    setIsUploading(true);
    setUploadProgress(0);
    setInsights(null); // reset insights panel

    // Show uploading status in chat immediately
    const uploadingMsgId = 'uploading-' + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: uploadingMsgId,
        role: 'assistant' as const,
        content: '⏳ Uploading **' + file.name + '**...',
        timestamp: new Date(),
      },
    ]);

    try {
      const response = await uploadDocument(file, (progress) => {
        setUploadProgress(progress);
      });

      const newDoc: UploadedFile = { id: response.document_id, name: response.filename };
      setUploadedFiles((prev) => [...prev, newDoc]);
      setSelectedDocId(response.document_id);

      const fileName = response.filename;
      const summary = response.summary;
      const pages = response.pages ?? 1;
      const elements = response.elements_detected ?? 0;
      const isImage = file.type.startsWith('image/');
      let summaryContent: string;
      if (summary) {
        summaryContent = (isImage ? '🖼️' : '📄') + ' **' + fileName + '** — uploaded successfully!\n\n' + summary + '\n\n*Ask me anything about this ' + (isImage ? 'image' : 'document') + '.*';
      } else {
        summaryContent = '✅ **' + fileName + '** uploaded! (' + (isImage ? elements + ' elements detected' : pages + ' pages') + ')\n\nAsk me anything about it.';
      }

      // Replace the uploading spinner message with the real summary
      setMessages((prev) => prev.map((m) =>
        m.id === uploadingMsgId
          ? { ...m, content: summaryContent }
          : m
      ));

      // Fetch proactive insights in background (after 5 s to let Qdrant index)
      setTimeout(async () => {
        setLoadingInsights(true);
        try {
          const ins = await getInsights(response.document_id);
          if (!ins.error) setInsights(ins);
        } catch (e) {
          console.error('Insights failed:', e);
        } finally {
          setLoadingInsights(false);
        }
      }, 5000);

    } catch (error) {
      setMessages((prev) => prev.map((m) =>
        m.id === uploadingMsgId
          ? { ...m, content: 'Failed to upload: ' + (error instanceof Error ? error.message : 'Unknown error') }
          : m
      ));
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Paperclip in AnimatedAIChat sends a FileList directly
  const handlePaperclipUpload = (files: FileList) => {
    handleFileSelect(files);
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;

    if (!selectedDocId) {
      alert('Please upload a PDF or image file first, then ask your question.');
      return;
    }

    // Add user message immediately
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), role: 'user' as const, content: message, timestamp: new Date() },
    ]);

    try {
      const response = await query(message, selectedDocId);

      // Build citation footer if available
      let citationNote = '';
      if (response.citations && response.citations.length > 0) {
        citationNote = '\n\n---\n*Sources: ' +
          response.citations
            .slice(0, 3)
            .map((c, i) => `[${i + 1}] "${c.text_preview.slice(0, 60)}…" (score: ${c.relevance_score})`)
            .join(' · ') +
          '*';
      }

      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant' as const,
          content: response.answer + citationNote,
          timestamp: new Date(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant' as const,
          content: 'Error: ' + (error instanceof Error ? error.message : 'Query failed'),
          timestamp: new Date(),
        },
      ]);
    }
  };

  // Click a suggested question → send it immediately
  const handleSuggestedQuestion = (q: string) => {
    handleSendMessage(q);
  };

  return (
    <main className="relative min-h-screen">
      <FlowFieldBackground />


      <div className="relative z-10 flex flex-col min-h-screen">
        <header className="border-b border-border/50 backdrop-blur-md bg-background/80">
          <div className="container mx-auto px-4 py-4">

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold">ICDI-X</h1>
                  <p className="text-xs text-muted-foreground">Intelligent Document Processing</p>
                </div>
              </div>

              <FileTriggerButton
                onSelect={handleFileSelect}
                acceptedFileTypes={['application/pdf', '.pdf', 'image/png', 'image/jpeg', '.png', '.jpg', '.jpeg']}
                variant="outline"
                className="gap-2"
                disabled={isUploading}
              >
                <Upload className="w-4 h-4" />
                {isUploading ? `Uploading ${Math.round(uploadProgress)}%` : 'Upload File'}
              </FileTriggerButton>
            </div>

            {uploadedFiles.length > 0 && (
              <div className="mt-3 px-4 py-3 bg-blue-500/10 border border-blue-500/30 rounded-lg backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground">
                    {uploadedFiles.length} file{uploadedFiles.length > 1 ? 's' : ''} loaded
                  </span>
                  {uploadedFiles.length > 1 && (
                    <span className="text-xs text-muted-foreground">Click to switch</span>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {uploadedFiles.map((file) => {
                    const isActive = selectedDocId === file.id;
                    return (
                      <button
                        key={file.id}
                        onClick={() => setSelectedDocId(file.id)}
                        className={cn(
                          'flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm transition-all',
                          isActive
                            ? 'bg-blue-500/20 border-blue-500 text-foreground font-medium'
                            : 'bg-muted/30 border-transparent text-muted-foreground hover:border-muted-foreground/40'
                        )}
                      >
                        <span
                          className={cn(
                            'w-2 h-2 rounded-full flex-shrink-0',
                            isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
                          )}
                        />
                        <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="max-w-xs truncate">{file.name}</span>
                        {isActive && (
                          <span className="text-green-600 text-xs ml-1 font-bold">Active</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

          </div>
        </header>

        <div className="flex-1 container mx-auto px-4 py-6">
          <div className="max-w-4xl mx-auto flex flex-col gap-4 h-[calc(100vh-160px)]">

            {/* Insights Panel — shown after upload */}
            {loadingInsights && (
              <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-blue-500/20 bg-blue-500/5 text-xs text-blue-300 animate-pulse">
                <Brain className="w-3.5 h-3.5" />
                <span>Generating proactive insights via IB compression…</span>
              </div>
            )}
            {insights && !loadingInsights && (
              <InsightsPanel insights={insights} onQuestionClick={handleSuggestedQuestion} />
            )}

            {/* Main Chat */}
            <div className="flex-1 min-h-0">
              <AnimatedAIChat
                initialMessages={messages}
                onSendMessage={handleSendMessage}
                onFileUpload={handlePaperclipUpload}
                placeholder={
                  selectedDocId
                    ? 'Ask me anything about your document…'
                    : 'Upload a file above to get started…'
                }
                className="h-full shadow-2xl backdrop-blur-md bg-background/90"
              />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
