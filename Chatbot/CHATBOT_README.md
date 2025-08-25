# Chatbot Integration Guide

This guide explains how to integrate your Python DistilBERT emotion detection chatbot with the React frontend.

## 🏗️ Architecture

- **Backend**: Flask API server (`api_server.py`) that serves your Python chatbot model
- **Frontend**: React application with real-time chat interface
- **Communication**: HTTP API calls between frontend and backend

## 📋 Prerequisites

1. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

2. **Node.js Dependencies**: Install frontend dependencies
   ```bash
   npm install
   ```

3. **Model Files**: Ensure your DistilBERT model files are in the `Emotion-Detector/model_directory` directory

## 🚀 Quick Start

### Option 1: Use the Batch Script (Windows)
1. Double-click `start_chatbot.bat`
2. This will automatically start both the API server and frontend

### Option 2: Manual Start

1. **Start Flask API Server** (Terminal 1):
   ```bash
   python api_server.py
   ```
   - Server will run on `http://localhost:5000`
   - You should see: "Starting Flask API server..."

2. **Start React Frontend** (Terminal 2):
   ```bash
   npm run dev
   ```
   - Frontend will run on `http://localhost:5173`

## 🔧 API Endpoints

- `POST /api/chat` - Send message and get response with emotion analysis
- `GET /api/health` - Health check endpoint

## 📱 Frontend Features

- **Real-time Chat**: Send messages and receive AI responses
- **Emotion Analysis**: View detected emotions with confidence scores
- **Interactive UI**: Toggle emotion analysis visualization
- **Auto-scroll**: Automatically scrolls to latest messages
- **Loading States**: Shows "Thinking..." while processing
- **Error Handling**: Displays errors gracefully

## 🧠 How It Works

1. **User Input**: User types a message in the frontend
2. **API Request**: Frontend sends message to Flask API
3. **Emotion Detection**: Python model analyzes text using DistilBERT
4. **Response Generation**: OpenAI GPT generates empathetic response
5. **Data Return**: API returns response + emotion data
6. **UI Update**: Frontend displays response with emotion visualization

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**:
   - Ensure Flask server is running on port 5000
   - Check if port 5000 is available

2. **Model Loading Error**:
   - Verify DistilBERT model files are in correct location
   - Check Python dependencies are installed

3. **CORS Issues**:
   - Flask-CORS is enabled, but if issues persist, check browser console

4. **OpenAI API Errors**:
   - Verify your API key in `chatbot.py`
   - Check API quota and billing

### Debug Mode

- Flask server runs in debug mode by default
- Check terminal output for detailed error messages
- Frontend console shows API request/response details

## 📁 File Structure

```
Chatbot/
├── api_server.py              # Flask API server
├── start_chatbot.bat         # Windows startup script
├── src/
│   ├── hooks/
│   │   └── useChatbot.ts     # Chat logic hook
│   ├── components/
│   │   └── ChatMessage.tsx   # Message component
│   └── screens/
│       └── DesktopSizeX/     # Main chat interface
└── package.json              # Node.js dependencies
Emotion-Detector/         
│   ├── model_directory/
│   │   └── distilbert/     # Python model
```

## 🔄 Customization

### Adding New Emotions
- Update the `getEmotionEmoji` function in `ChatMessage.tsx`
- Ensure your model supports the new emotion labels

### Modifying Response Style
- Edit the system prompt in `chatbot.py` `generate_response` function
- Adjust temperature and other OpenAI parameters

### UI Changes
- Modify `DesktopSizeX.tsx` for layout changes
- Update `ChatMessage.tsx` for message styling
- Customize colors in `tailwind.css`

## 📊 Performance Tips

1. **Model Loading**: DistilBERT loads once when API starts
2. **Caching**: Consider adding Redis for conversation caching
3. **Async Processing**: API handles requests asynchronously
4. **Error Recovery**: Graceful fallbacks for API failures

## 🔐 Security Notes

- API key is stored in `chatbot.py` (consider environment variables)
- CORS is enabled for development (restrict in production)
- Input validation on both frontend and backend

## 🚀 Production Deployment

1. **Environment Variables**: Move API keys to environment
2. **CORS Restrictions**: Limit allowed origins
3. **Rate Limiting**: Add request throttling
4. **Logging**: Implement proper logging
5. **Monitoring**: Add health checks and metrics

## 📞 Support

If you encounter issues:
1. Check terminal output for error messages
2. Verify all dependencies are installed
3. Ensure model files are in correct locations
4. Check API endpoints are accessible

---

**Happy Chatting! 🤖💬**
