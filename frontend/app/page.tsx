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
    <div className="flex flex-col h-full rounded-xl border border-blue-500/30 bg-blue-500/5 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-blue-500/20 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <Brain className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
          <span className="text-xs font-semibold text-blue-300 whitespace-nowrap">Proactive Insights</span>
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-300 truncate">
            {insights.doc_type?.replace(/_/g, ' ')}
          </span>
        </div>
        <span className={cn('text-xs font-medium flex-shrink-0 ml-1', complexityColor)}>
          {insights.complexity}
        </span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4 text-sm">

        {/* Key Insights */}
        {insights.insights.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">🔍 Key Insights</p>
            <ul className="space-y-2">
              {insights.insights.map((ins, i) => (
                <li key={i} className="flex gap-2 leading-snug">
                  <span className="text-blue-400 mt-0.5 flex-shrink-0 text-xs">▸</span>
                  <span className="text-foreground/85 text-xs">{ins}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Suggested Questions */}
        {insights.suggested_questions.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">💡 Ask me…</p>
            <div className="flex flex-col gap-1.5">
              {insights.suggested_questions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => onQuestionClick(q)}
                  className="text-xs px-2.5 py-1.5 rounded-lg border border-purple-500/40 bg-purple-500/10 text-purple-200 hover:bg-purple-500/25 transition-colors text-left leading-snug"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Key entities */}
        {insights.key_entities.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">🏷️ Entities</p>
            <div className="flex flex-wrap gap-1">
              {insights.key_entities.slice(0, 8).map((e, i) => (
                <span key={i} className="text-xs px-2 py-0.5 rounded-md bg-muted/40 text-muted-foreground border border-border/50">
                  {e}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Key themes */}
        {insights.key_themes?.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">🗂️ Themes</p>
            <div className="flex flex-wrap gap-1">
              {insights.key_themes.map((t, i) => (
                <span key={i} className="text-xs px-2 py-0.5 rounded-md bg-green-500/10 text-green-400 border border-green-500/20">
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* IB stats */}
        <p className="text-xs text-muted-foreground/60">
          IB selected {insights.chunks_selected_by_ib}/{insights.chunks_analyzed} chunks
        </p>
      </div>

      {/* Study Guide toggle — pinned at bottom */}
      <div className="px-3 py-2.5 border-t border-blue-500/20 flex-shrink-0">
        <button
          onClick={handleLoadGuide}
          disabled={loadingGuide}
          className="flex items-center gap-1.5 text-xs text-yellow-400 hover:text-yellow-300 transition-colors font-medium w-full"
        >
          <BookOpen className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="flex-1 text-left">
            {loadingGuide ? 'Generating study guide…' : showStudyGuide ? 'Hide Study Guide' : "📚 Bloom's Study Guide"}
          </span>
          {!loadingGuide && (showStudyGuide ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
        </button>

        {showStudyGuide && studyGuide && (
          <div className="mt-2 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-2.5 max-h-60 overflow-y-auto">
            <p className="text-xs font-semibold text-yellow-300 mb-1.5">{studyGuide.title}</p>
            <p className="text-xs text-muted-foreground mb-2 leading-snug">{studyGuide.summary}</p>
            {(['remember', 'understand', 'apply'] as const).map((level) => {
              const qs = studyGuide.blooms_questions?.[level] ?? [];
              if (!qs.length) return null;
              return (
                <div key={level} className="mb-1.5">
                  <p className="text-xs font-semibold text-yellow-400/80 capitalize mb-1">
                    L{['remember','understand','apply','analyze','evaluate','create'].indexOf(level)+1} {level}
                  </p>
                  {qs.slice(0, 2).map((q, i) => (
                    <button key={i} onClick={() => onQuestionClick(q)}
                      className="block text-xs text-left text-foreground/70 hover:text-foreground py-0.5 leading-snug transition-colors">
                      · {q}
                    </button>
                  ))}
                </div>
              );
            })}
            <p className="text-xs text-muted-foreground mt-1.5">
              {studyGuide.estimated_study_time_minutes} min · {studyGuide.key_concepts.length} concepts
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

        <div className="flex-1 min-h-0 container mx-auto px-4 py-4">
          <div className="flex gap-4 h-[calc(100vh-130px)] max-w-[1400px] mx-auto">

            {/* ── Left sidebar: Insights Panel ────────────────── */}
            <div className="w-[340px] flex-shrink-0 flex flex-col min-h-0">
              {loadingInsights && (
                <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-blue-500/20 bg-blue-500/5 text-xs text-blue-300 animate-pulse">
                  <Brain className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>Analysing document via IB compression…</span>
                </div>
              )}
              {insights && !loadingInsights && (
                <InsightsPanel insights={insights} onQuestionClick={handleSuggestedQuestion} />
              )}
              {!insights && !loadingInsights && (
                <div className="flex flex-col items-center justify-center h-full rounded-xl border border-border/30 bg-muted/10 text-center p-6 gap-3">
                  <Brain className="w-8 h-8 text-muted-foreground/30" />
                  <p className="text-xs text-muted-foreground/50 leading-relaxed">
                    Upload a document to see proactive insights, suggested questions, and a Bloom's study guide here.
                  </p>
                </div>
              )}
            </div>

            {/* ── Right panel: Chat ────────────────────────────── */}
            <div className="flex-1 min-w-0 min-h-0">
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
