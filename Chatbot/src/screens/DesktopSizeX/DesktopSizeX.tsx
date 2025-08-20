import React, { useState, useRef, useEffect } from "react";
import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import { useChatbot } from "../../hooks/useChatbot";
import { ChatMessage } from "../../components/ChatMessage";

export const DesktopSizeX = (): JSX.Element => {
  // Use the chatbot hook
  const { messages, isLoading, error, sendMessage, clearMessages } = useChatbot();
  
  // State for input
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Send message handler
  const handleSendMessage = () => {
    if (input.trim() === "") return;
    sendMessage(input.trim());
    setInput("");
    // Optionally focus input again
    inputRef.current?.focus();
  };

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const chatArea = document.querySelector('main');
    if (chatArea) {
      chatArea.scrollTop = chatArea.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="h-screen w-full bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 fixed inset-0">
      {/* Static background gradient effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute w-[280px] h-[280px] top-[20%] right-[20%] bg-blue-300/30 rounded-full blur-[120px]" />
        <div className="absolute w-[350px] h-[350px] top-[40%] left-[15%] bg-purple-300/30 rounded-full blur-[150px]" />
        <div className="absolute w-[200px] h-[200px] bottom-[30%] right-[30%] bg-pink-300/30 rounded-full blur-[100px]" />
      </div>

      {/* Main grid layout */}
      <div className="h-full grid grid-rows-[auto_1fr_auto] w-full relative z-10">
        {/* Fixed Header */}
        <header className="bg-white/20 backdrop-blur-md border-b border-white/30 shadow-sm w-full">
          <div className="relative py-4 px-4 w-full">
            {/* Centered Logo and Title */}
            <div className="flex flex-col items-center">
              <img
                className="w-8 h-[32px]"
                alt="Logo"
                src="/logo.png"
              />
              <div className="text-center mt-1">
                <h1 className="font-normal text-gray-800 text-l">
                  Talk. Feel. Connect.
                </h1>
              </div>
            </div>
            
            {/* Right side - Clear Chat button (vertically centered) */}
            {messages.length > 0 && (
              <div className="absolute top-1/2 right-4 transform -translate-y-1/2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-gray-600 hover:text-gray-800 hover:bg-white/20 px-3 py-1"
                  onClick={clearMessages}
                >
                  Clear Chat
                </Button>
              </div>
            )}
          </div>
        </header>

        {/* Scrollable Chat Area */}
        <main className="overflow-y-auto scroll-smooth px-4 py-6 w-full">
          <div className="w-full space-y-4">
            {/* Welcome message */}
            {messages.length === 0 && (
              <div className="flex justify-start">
                <div className="max-w-3xl w-full">
                  <div className="text-xs font-medium text-gray-500 mb-2">
                    BOT
                  </div>
                  <Card className="bg-white/60 backdrop-blur-sm border-white/40 shadow-sm">
                    <CardContent className="p-3">
                      <p className="text-gray-800 text-sm leading-relaxed">
                        Hi! How are you feeling today? I'm here to listen.
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {/* Render chat messages dynamically */}
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
                emotions={message.emotions}
                primaryEmotion={message.primaryEmotion}
                primaryConfidence={message.primaryConfidence}
              />
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-3xl w-full">
                  <div className="text-xs font-medium text-gray-500 mb-2">
                    BOT
                  </div>
                  <Card className="bg-white/60 backdrop-blur-sm border-white/40 shadow-sm">
                    <CardContent className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500"></div>
                        <span className="text-gray-600 text-sm">Thinking...</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="flex justify-start">
                <div className="max-w-3xl w-full">
                  <div className="text-xs font-medium text-gray-500 mb-2">
                    SYSTEM
                  </div>
                  <Card className="bg-red-50 border-red-200 shadow-sm">
                    <CardContent className="p-3">
                      <p className="text-red-600 text-sm">
                        Error: {error}
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Fixed Input Area */}
        <footer className="bg-white/20 backdrop-blur-md border-t border-white/20 shadow-lg w-full">
          <div className="w-full p-4">
            <Card className="bg-white/80 backdrop-blur-sm border-white/50 shadow-sm w-full">
              <CardContent className="p-1">
                <div className="flex items-center gap-3">
                  <Input
                    ref={inputRef}
                    className="flex-1 border-none shadow-none focus-visible:ring-0 bg-transparent placeholder:text-gray-500 text-gray-800"
                    placeholder="Ask me anything"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                  <Button
                    variant="ghost"
                    className="p-2 h-9 w-9 hover:bg-white/50"
                    onClick={handleSendMessage}
                  >
                    <img
                      className="w-5 h-5"
                      alt="Send button"
                      src="/send-button.svg"
                    />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </footer>
      </div>
    </div>
  );
};