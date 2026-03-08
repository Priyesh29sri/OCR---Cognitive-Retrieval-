'use client';

import React, { useState } from 'react';
import { Upload, FileText, Sparkles } from 'lucide-react';
import { FlowFieldBackground } from '@/components/ui/flow-field-background';
import { AnimatedAIChat } from '@/components/ui/animated-ai-chat';
import { FileTriggerButton } from '@/components/ui/file-trigger';
import { cn } from '@/lib/utils';
import { uploadDocument, query } from '@/lib/api';

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

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Called by both the header Upload button and the paperclip inside chat
  const handleFileSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    setIsUploading(true);
    setUploadProgress(0);

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
      let summaryContent: string;
      if (summary) {
        summaryContent = '📄 **' + fileName + '**\n\n' + summary + '\n\n*Ask me anything about this document.*';
      } else {
        summaryContent = '\u2705 **' + fileName + '** uploaded! (' + pages + ' pages)\n\nAsk me anything about it.';
      }

      // Replace the uploading spinner message with the real summary
      setMessages((prev) => prev.map((m) =>
        m.id === uploadingMsgId
          ? { ...m, content: summaryContent }
          : m
      ));
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

    try {
      const response = await query(message, selectedDocId);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant' as const,
          content: response.answer,
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

        <div className="flex-1 container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto h-[calc(100vh-200px)]">
            <AnimatedAIChat
              initialMessages={messages}
              onSendMessage={handleSendMessage}
              onFileUpload={handlePaperclipUpload}
              placeholder={
                selectedDocId
                  ? 'Ask me anything about your document...'
                  : 'Upload a file above to get started...'
              }
              className="h-full shadow-2xl backdrop-blur-md bg-background/90"
            />
          </div>
        </div>
      </div>
    </main>
  );
}
