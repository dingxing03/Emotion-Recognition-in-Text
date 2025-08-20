from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the Emotion-Detector directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../Emotion-Detector/model_directory'))

# Import the chatbot functions
from chatbot import preprocess, predict_emotions, generate_response, translate_mixed_language

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        print(f"🔍 Processing message: {user_message}")
        
        # Step 1: Translate mixed language 
        try:
            translated_text = translate_mixed_language(user_message)
            print(f"🌐 Translation result: {translated_text}")
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
            translated_text = user_message
        
        # Step 2: Text preprocessing
        try:
            cleaned_text = preprocess(translated_text)
            print(f"🧹 Preprocessed text: {cleaned_text}")
        except Exception as e:
            print(f"⚠️ Preprocessing failed: {e}")
            cleaned_text = translated_text
        
        # Step 3: Predict emotions using DistilBERT
        try:
            emotion_confidences = predict_emotions(cleaned_text)
            print(f"😊 Detected emotions: {emotion_confidences}")
        except Exception as e:
            print(f"⚠️ Emotion detection failed: {e}")
            emotion_confidences = []
        
        # Step 4: Generate empathetic response
        try:
            ai_response = generate_response(
                user_input=user_message,  # Use original message for context
                emotion_confidences=emotion_confidences,
                conversation_history=conversation_history
            )
            print(f"🤖 Generated response: {ai_response}")
        except Exception as e:
            print(f"⚠️ Response generation failed: {e}")
            ai_response = "I'm here to listen. Could you tell me more about how you're feeling?"
        
        # Format emotions for frontend
        emotions_data = []
        for emotion, confidence in emotion_confidences:
            emotions_data.append({
                'name': emotion.capitalize(),
                'value': int(confidence * 100),
                'confidence': confidence,
                'isCurrent': len(emotions_data) == 0  # First emotion is current
            })
        
        return jsonify({
            'response': ai_response,
            'emotions': emotions_data,
            'primary_emotion': emotion_confidences[0][0].capitalize() if emotion_confidences else None,
            'primary_confidence': emotion_confidences[0][1] if emotion_confidences else None,
            'debug_info': {
                'original_message': user_message,
                'translated_text': translated_text,
                'cleaned_text': cleaned_text,
                'emotion_count': len(emotion_confidences)
            }
        })
        
    except Exception as e:
        print(f"🚨 Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Chatbot API is running'})

if __name__ == '__main__':
    print("Starting Flask API server...")
    print("Make sure you have the required dependencies installed:")
    print("pip install flask flask-cors")
    app.run(debug=True, host='0.0.0.0', port=5000)
