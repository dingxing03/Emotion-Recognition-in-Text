import React, { useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Emotion } from '../hooks/useChatbot';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  emotions?: Emotion[];
  primaryEmotion?: string;
  primaryConfidence?: number;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  emotions,
  primaryEmotion,
  primaryConfidence,
}) => {
  const [showEmotionAnalysis, setShowEmotionAnalysis] = useState(false);

  const isUser = role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-3xl ${isUser ? '' : 'w-full'}`}>
        <div className={`text-xs font-medium text-gray-500 mb-2 ${isUser ? 'text-right' : ''}`}>
          {isUser ? 'ME' : 'BOT'}
        </div>
        
        {/* Message content */}
        <Card className="bg-white/60 backdrop-blur-sm border-white/40 shadow-sm mb-3">
          <CardContent className="p-3">
            <p className="text-gray-800 text-sm leading-relaxed">
              {content}
            </p>
          </CardContent>
        </Card>

        {/* Emotion Analysis for bot messages */}
        {!isUser && emotions && emotions.length > 0 && (
          <Card className="bg-white/30 backdrop-blur-sm border-white/30 shadow-sm">
            <CardContent className="p-3">
              {/* Toggle header */}
              <div
                className="flex items-center justify-between mb-2 cursor-pointer select-none"
                onClick={() => setShowEmotionAnalysis((v) => !v)}
              >
                <span
                  className={`text-sm ${showEmotionAnalysis ? 'font-bold' : 'font-medium'} text-gray-700`}
                >
                  Show Emotion Analysis
                </span>
                <img
                  className={`w-3 h-2 ${showEmotionAnalysis ? 'opacity-100 brightness-0' : 'opacity-60'}`}
                  alt="Show button"
                  src="/show-button.png"
                />
              </div>
              
              {showEmotionAnalysis && (
                <div className="pt-1">
                  {/* Current emotion display */}
                  <div className="text-center mb-6">
                    <h3 className="text-lg font-normal text-gray-800">
                      {primaryEmotion && primaryConfidence ? (
                        <>
                          {getEmotionEmoji(primaryEmotion)} {primaryEmotion}
                          <span className="block text-sm text-gray-600 mt-1">
                            Confidence: {Math.round(primaryConfidence * 100)}%
                          </span>
                        </>
                      ) : (
                        'Emotion Detected'
                      )}
                    </h3>
                  </div>
                  
                  {/* Emotion visualization */}
                  <div className="relative max-w-sm mx-auto">
                    {/* Chart container */}
                    <div className="flex items-end justify-center gap-3 h-30 mb-2">
                      {emotions.map((emotion, index) => (
                        <div key={emotion.name} className="flex flex-col items-center">
                          {/* Bar container */}
                          <div className="relative w-10 h-32 bg-gray-200/50 rounded-full overflow-hidden">
                            {/* Background track */}
                            <div className="absolute inset-x-0 bottom-0 w-1.5 mx-auto bg-gray-300/60 rounded-full h-full" />
                            {/* Value bar */}
                            <div
                              className={`absolute inset-x-0 bottom-0 w-2 mx-auto rounded-full transition-all duration-500 ${
                                emotion.isCurrent 
                                  ? 'bg-indigo-500' 
                                  : 'bg-indigo-400/70'
                              }`}
                              style={{
                                height: `${Math.min((emotion.value / 100) * 100, 100)}%`
                              }}
                            />
                            {/* Highlight for current emotion */}
                            {emotion.isCurrent && (
                              <div className="absolute inset-0 bg-indigo-100/30 rounded-full" />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    {/* Emotion labels */}
                    <div className="h-12 flex justify-center gap-4">
                      {emotions.map((emotion, index) => (
                        <div
                          key={emotion.name}
                          className="w-10 flex items-end justify-center"
                        >
                          <span className={`text-xs leading-tight block transform -rotate-[30deg] origin-top-left text-right ${
                            emotion.isCurrent 
                              ? 'font-bold text-gray-800' 
                              : 'font-normal text-gray-600/70'
                          }`}>
                            {emotion.name}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

// Helper function to get emoji for emotions
const getEmotionEmoji = (emotion: string): string => {
  const emotionEmojis: { [key: string]: string } = {
      'Admiration': '👏',
      'Amusement': '😂',
      'Anger': '😠',
      'Annoyance': '😒',
      'Approval': '👍',
      'Caring': '🤗',
      'Confusion': '😕',
      'Curiosity': '🤔',
      'Desire': '😍',
      'Disappointment': '😞',
      'Disapproval': '👎',
      'Disgust': '🤢',
      'Embarrassment': '😳',
      'Excitement': '🤩',
      'Fear': '😨',
      'Gratitude': '🙏',
      'Grief': '💔',
      'Joy': '😁',
      'Love': '❤️',
      'Nervousness': '😟',
      'Optimism': '🌅',
      'Pride': '😌',
      'Realization': '💡',
      'Relief': '😌',
      'Remorse': '😣',
      'Sadness': '😢',
      'Surprise': '😲',
      'Neutral': '😐',  
  };
  
  return emotionEmojis[emotion] || '😐';
};
