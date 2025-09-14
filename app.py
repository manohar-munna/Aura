import os
import json
import logging
from datetime import datetime
from flask import Flask, request, session, jsonify, render_template, redirect, url_for, Response
from flask_session import Session
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from sqlalchemy import desc
from sqlalchemy.orm import joinedload

from gemini import client as gemini_client, analyze_sentiment
from send_message import send_twilio_message
from elevenlabs_integration import is_configured as elevenlabs_configured, create_conversation_with_agent, ELEVENLABS_CRITICAL_ALERT_AGENT_ID
from twilio.twiml.voice_response import VoiceResponse
from models import db, init_db, User, PatientProfile, DoctorPatient, Conversation, Message, SentimentSnapshot, Alert, NotificationLog, UserType, ConversationChannel, MessageSender, AlertSeverity, AlertType, NotificationChannel, NotificationStatus
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', secrets.token_hex(16))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME', '127.0.0.1:5000') 
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///aura.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
Session(app)

init_db(app)
migrate = Migrate(app, db)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_critical_mood_and_alert(user_id, sentiment_rating, conversation_text):
    try:
        user = db.session.get(User, user_id)
        if not user or user.user_type != UserType.PATIENT:
            return
        
        recent_sentiments = db.session.query(SentimentSnapshot).filter_by(
            patient_id=user_id
        ).order_by(desc(SentimentSnapshot.created_at)).limit(5).all()
        
        is_critical = False
        alert_type = None
        rationale = ""
        
        if sentiment_rating <= 1.5:
            is_critical = True
            alert_type = AlertType.SEVERE_DEPRESSION
            rationale = f"Very low mood detected (rating: {sentiment_rating})"
        
        if len(recent_sentiments) >= 3:
            avg_recent = sum(s.rating for s in recent_sentiments[:3]) / 3
            if avg_recent <= 2.0 and sentiment_rating < avg_recent - 1.0:
                is_critical = True
                alert_type = AlertType.SEVERE_DEPRESSION
                rationale = f"Rapid mood decline detected (current: {sentiment_rating}, recent avg: {avg_recent:.1f})"
        
        crisis_keywords = ['suicide', 'kill myself', 'end it all', 'hurt myself', 'no point', 'give up']
        text_lower = conversation_text.lower()
        if any(keyword in text_lower for keyword in crisis_keywords):
            is_critical = True
            alert_type = AlertType.SUICIDAL_IDEATION
            rationale = "Crisis language detected in conversation"
        
        if is_critical:
            doctor_assignment = DoctorPatient.query.filter_by(
                patient_id=user_id, is_active=True
            ).first()
            
            if doctor_assignment:
                alert = Alert(
                    patient_id=user_id,
                    severity=AlertSeverity.CRITICAL,
                    alert_type=alert_type,
                    rationale=rationale,
                    trigger_data={
                        'sentiment_rating': sentiment_rating,
                        'conversation_snippet': conversation_text[:200],
                        'recent_ratings': [s.rating for s in recent_sentiments]
                    }
                )
                db.session.add(alert)
                db.session.commit()
                
                call_doctor_for_critical_alert(alert, doctor_assignment.doctor)
                
                logger.critical(f"Critical alert triggered for patient {user_id}, doctor {doctor_assignment.doctor_id} notified")
            else:
                logger.warning(f"Critical patient {user_id} has no assigned doctor for emergency contact")
                
    except Exception as e:
        logger.error(f"Error in critical mood check: {e}")

def call_doctor_for_critical_alert(alert, doctor):
    try:
        if not doctor.phone:
            logger.error(f"Doctor {doctor.id} has no phone number for emergency contact")
            return
        
        patient = db.session.get(User, alert.patient_id)
        alert_type_text = alert.alert_type.value.replace('_', ' ').title()

        recent_messages = db.session.query(Message).join(Conversation).filter(
            Conversation.patient_id == patient.id
        ).order_by(Message.created_at.desc()).limit(10).all()
        
        conversation_history = "\n".join(
            [f"{'Patient' if msg.sender == MessageSender.PATIENT else 'Aura'}: {msg.text}" for msg in reversed(recent_messages)]
        )

        def build_snippet(messages, max_chars=1000):
            parts = []
            for m in messages:
                who = "Patient" if m.sender == MessageSender.PATIENT else "Aura"
                parts.append(f"{who}: {m.text}")
            joined = "\n".join(parts)
            if len(joined) > max_chars:
                return joined[:max_chars-3] + "..."
            return joined

        recent = recent_messages or []
        conversation_snippet = build_snippet(reversed(recent[:6]), max_chars=800) if recent else ""

        dynamic_vars = {
            "user_name": patient.name,
            "patient_id": str(patient.id),
            "alert_type": alert_type_text,
            "alert_rationale": alert.rationale,
            "conversation_snippet": conversation_snippet,
            "recent_ratings": alert.trigger_data.get('recent_ratings', []) if getattr(alert, "trigger_data", None) else [],
            "trigger_snippet": (alert.trigger_data.get('conversation_snippet', '')[:240] if getattr(alert, "trigger_data", None) else '')
        }

        if elevenlabs_configured():
            res = create_conversation_with_agent(
                phone_number=doctor.get_full_phone(),
                agent_id=ELEVENLABS_CRITICAL_ALERT_AGENT_ID,
                customer_id=f"doctor_alert_{alert.id}",
                dynamic_variables=dynamic_vars,
                status_callback_url=url_for('elevenlabs_call_status_webhook', _external=True)
            )
            if res:
                logger.info(f"ElevenLabs call initiation response: {res}")

        sms_summary = f"URGENT AURA ALERT: Patient {patient.name} requires immediate attention. A {alert_type_text} alert was triggered. A context-aware AI is calling you now. Please check your dashboard."
        send_twilio_message(doctor.get_full_phone(), sms_summary)

        logger.info(f"Context-aware alert call initiated to Dr. {doctor.id} for patient {patient.id}")
    except Exception as e:
        logger.error(f"Failed to notify doctor of critical alert: {e}")

def get_user_from_session():
    if 'user_id' not in session:
        return None
    return db.session.get(User, session['user_id'])

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('user_type') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    user_type = data.get('user_type', 'patient')
    name = data.get('name')
    phone = data.get('phone')
    country_code = data.get('country_code', '')
    
    if not all([email, password, name, phone]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if not isinstance(password, str) or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    try:
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'User already exists'}), 400
        
        user = User(
            name=name,
            email=email,
            phone=phone,
            country_code=country_code,
            user_type=UserType.PATIENT if user_type == 'patient' else UserType.DOCTOR
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.flush()
        
        if user.user_type == UserType.PATIENT:
            profile = PatientProfile(
                user_id=user.id,
                consent_analytics=True,
                consent_doctor_sharing=True,
                consent_emergency_contact=True
            )
            db.session.add(profile)
        
        db.session.commit()
        
        return jsonify({'success': True, 'user_id': user.id})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'error': 'Missing email or password'}), 400
    
    if not isinstance(password, str):
        return jsonify({'error': 'Invalid password format'}), 400
    
    try:
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['user_id'] = user.id
        session['user_email'] = email
        session['user_type'] = user.user_type.value
        session['user_name'] = user.name
        
        return jsonify({'success': True, 'user_type': user.user_type.value})
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/chat')
def chat():
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/doctor')
def doctor_dashboard():
    if 'user_id' not in session or session.get('user_type') != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor_dashboard.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized - patients only'}), 403
    
    data = request.get_json() or {}
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        recent_conversations = db.session.query(Conversation).filter_by(
            patient_id=session['user_id']
        ).order_by(desc(Conversation.started_at)).limit(3).all()

        context = ""
        for conv in reversed(recent_conversations):
            messages = db.session.query(Message).filter_by(
                conversation_id=conv.id
            ).order_by(Message.created_at.asc()).limit(2).all()
            for msg in messages:
                if msg.sender == MessageSender.PATIENT:
                    context += f"User: {msg.text}\n"
                else:
                    context += f"Aura: {msg.text}\n"
            context += "\n"

        system_prompt = """You are Aura, a compassionate and professional AI mental health companion. You are:

EMPATHETIC: Always validate feelings and show understanding
SUPPORTIVE: Provide comfort, hope, and encouragement
PROFESSIONAL: Maintain appropriate boundaries and suggest professional help when needed
ACTIVE LISTENER: Ask thoughtful follow-up questions and remember what users share
CRISIS-AWARE: If someone mentions self-harm, suicide, or immediate danger, gently but clearly encourage them to contact emergency services or a crisis hotline

Guidelines:
- Keep responses warm, conversational, and understanding
- Use "I" statements to show empathy ("I hear that you're feeling...")
- Ask one thoughtful follow-up question per response
- If the user seems in crisis, provide crisis resources while staying supportive
- Remember context from previous messages in this conversation
- Responses should be 1-3 paragraphs, conversational length

Crisis resources to mention if needed:
- Emergency: 911 or local emergency services
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741"""
        
        conversation_history = [f"Previous conversation context:\n{context}"] if context else []
        conversation_history.append(f"Current message: {message}")
        
        full_prompt = f"{system_prompt}\n\nConversation:\n" + "\n".join(conversation_history)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=full_prompt
        )
        
        ai_response = response.text or "I'm here to listen and support you."
        
        user = get_user_from_session()
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        try:
            sentiment = analyze_sentiment(message)
            sentiment_rating = sentiment.rating
            sentiment_confidence = sentiment.confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment_rating = 3.0
            sentiment_confidence = 0.5
        
        conversation = Conversation(
            patient_id=user.id,
            channel=ConversationChannel.CHAT
        )
        db.session.add(conversation)
        db.session.flush()
        
        user_message = Message(
            conversation_id=conversation.id,
            sender=MessageSender.PATIENT,
            text=message,
            message_metadata={'sentiment_rating': sentiment_rating, 'sentiment_confidence': sentiment_confidence}
        )
        db.session.add(user_message)
        db.session.flush()
        
        ai_message = Message(
            conversation_id=conversation.id,
            sender=MessageSender.AI,
            text=ai_response,
            message_metadata={'model': 'gemini-2.0-flash', 'response_time': datetime.now().isoformat()}
        )
        db.session.add(ai_message)
        
        sentiment_snapshot = SentimentSnapshot(
            patient_id=user.id,
            source=ConversationChannel.CHAT,
            rating=sentiment_rating,
            confidence=sentiment_confidence,
            message_id=user_message.id,
            analysis_details={'original_message': message[:100], 'ai_model': 'gemini-sentiment'}
        )
        db.session.add(sentiment_snapshot)
        
        db.session.commit()
        
        check_critical_mood_and_alert(user.id, sentiment_rating, message)
        
        return jsonify({
            'response': ai_response,
            'conversation_id': conversation.id,
            'sentiment_rating': sentiment_rating,
            'use_streaming': False
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'I apologize, but I encountered a technical issue.'}), 500

@app.route('/api/my-doctor-assignment')
def api_my_doctor_assignment():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    assignment = DoctorPatient.query.options(joinedload(DoctorPatient.doctor)).filter_by(patient_id=session['user_id'], is_active=True).first()
    if assignment:
        return jsonify({'assigned': True, 'doctor': {'name': assignment.doctor.name, 'email': assignment.doctor.email}})
    return jsonify({'assigned': False})

@app.route('/api/chat/stream', methods=['POST'])
def api_chat_stream():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized - patients only'}), 403
    
    data = request.get_json() or {}
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    user_id = session['user_id']
    
    def generate_streaming_response():
        try:
            response = gemini_client.models.generate_content(model="gemini-2.0-flash-lite", contents=f"User: {message}")
            ai_response = response.text or "I am here to listen."

            words = ai_response.split()
            for word in words:
                yield f"data: {json.dumps({'type': 'word', 'word': word})}\n\n"
                import time
                time.sleep(0.05)
            
            with app.app_context():
                user = db.session.get(User, user_id)
                if not user:
                    logger.error(f"User {user_id} not found in database for streaming context.")
                    return

                sentiment = analyze_sentiment(message)
                
                conversation = Conversation(patient_id=user.id, channel=ConversationChannel.CHAT)
                db.session.add(conversation)
                db.session.flush()

                user_msg = Message(conversation_id=conversation.id, sender=MessageSender.PATIENT, text=message)
                ai_msg = Message(conversation_id=conversation.id, sender=MessageSender.AI, text=ai_response)
                db.session.add_all([user_msg, ai_msg])
                db.session.flush()

                snapshot = SentimentSnapshot(
                    patient_id=user.id,
                    source=ConversationChannel.CHAT,
                    rating=sentiment.rating,
                    confidence=sentiment.confidence,
                    message_id=user_msg.id
                )
                db.session.add(snapshot)
                db.session.commit()
                
                check_critical_mood_and_alert(user.id, sentiment.rating, message)
                
                yield f"data: {json.dumps({'type': 'complete', 'conversation_id': conversation.id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed.'})}\n\n"

    return Response(generate_streaming_response(), mimetype='text/event-stream')

@app.route('/api/start_call', methods=['POST'])
def api_start_call():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized - patients only'}), 403
    
    try:
        user = get_user_from_session()
        if not user or not user.phone:
            return jsonify({'error': 'Phone number not found. Please update your profile.'}), 400
        
        phone_number = user.get_full_phone()
        
        call_response_data = create_conversation_with_agent(
            phone_number=phone_number,
            customer_id=session['user_id']
        )
        
        if call_response_data:
            voice_conversation = Conversation(
                patient_id=user.id,
                channel=ConversationChannel.VOICE,
                provider_session_id=json.dumps(call_response_data)
            )
            db.session.add(voice_conversation)
            db.session.commit()
            
            logger.info(f"ElevenLabs voice call initiated for user {user.id}: {call_response_data.get('call_sid')}")
            
            return jsonify({
                'success': True,
                'conversation_id': voice_conversation.id,
                'phone_number': phone_number,
                'message': f'AI voice call initiated to {phone_number}. You should receive a call from Aura shortly.',
                'agent_configured': True
            })
        else:
            return jsonify({'error': 'Failed to initiate voice call. Please try again later.'}), 500
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Call initiation error: {e}", exc_info=True)
        return jsonify({'error': 'Unable to start call at this time. Please try again later.'}), 500

@app.route('/voice/webhook', methods=['POST'])
def voice_webhook():
    try:
        call_id = request.args.get('call_id')
        
        if not call_id or f'call_context_{call_id}' not in session:
            return "Call context not found", 400
        
        call_context = session[f'call_context_{call_id}']
        user_id = call_context['user_id']
        context = call_context['context']
        
        response = VoiceResponse()
        
        welcome_message = f"""Hello! This is Aura, your AI mental health companion. 
        I'm here to continue our conversation and provide support. 
        {'I have the context from our recent chat, so we can pick up where we left off.' if context else 'How are you feeling today?'}
        Please speak, and I'll respond with empathy and understanding."""
        
        response.say(welcome_message, voice='Polly.Joanna', language='en-US')
        
        gather = response.gather(
            input='speech',
            timeout=10,
            speechTimeout='auto',
            action=f'/voice/process?call_id={call_id}',
            method='POST'
        )
        
        gather.say("Please share what's on your mind, and I'll listen with care.", 
                  voice='Polly.Joanna', language='en-US')
        
        response.say("I didn't hear anything. Please try speaking again.", 
                    voice='Polly.Joanna', language='en-US')
        response.redirect(f'/voice/webhook?call_id={call_id}')
        
        return str(response), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        logger.error(f"Voice webhook error: {e}")
        response = VoiceResponse()
        response.say("I apologize, but I'm experiencing technical difficulties. Please try calling back later.", 
                    voice='Polly.Joanna', language='en-US')
        return str(response), 200, {'Content-Type': 'text/xml'}

@app.route('/voice/process', methods=['POST'])
def voice_process():
    try:
        call_id = request.args.get('call_id')
        
        if not call_id or f'call_context_{call_id}' not in session:
            return "Call context not found", 400
        
        call_context = session[f'call_context_{call_id}']
        user_id = call_context['user_id']
        
        user_speech = request.form.get('SpeechResult', '')
        
        if not user_speech:
            response = VoiceResponse()
            response.say("I didn't catch that. Could you please repeat what you said?", 
                        voice='Polly.Joanna', language='en-US')
            response.redirect(f'/voice/webhook?call_id={call_id}')
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        system_prompt = """You are Aura, a compassionate AI mental health companion in a voice conversation. 
        Keep responses conversational, warm, and under 150 words since this is voice. 
        Ask one follow-up question to keep the conversation flowing naturally."""
        
        full_prompt = f"{system_prompt}\n\nUser said: {user_speech}\n\nRespond with empathy and support:"
        
        gemini_response = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt
        )
        
        ai_response = gemini_response.text if gemini_response.text else "I understand you're reaching out, and I'm here to listen and support you."
        
        user = db.session.get(User, user_id)
        if user:
            conversation = Conversation.query.filter(Conversation.provider_session_id.contains(call_id)).first()
            if not conversation:
                conversation = Conversation(
                    patient_id=user.id,
                    channel=ConversationChannel.VOICE,
                    provider_session_id=json.dumps({'call_sid': call_id})
                )
                db.session.add(conversation)
                db.session.flush()

            user_message = Message(
                conversation_id=conversation.id,
                sender=MessageSender.PATIENT,
                text=user_speech
            )
            db.session.add(user_message)

            ai_message = Message(
                conversation_id=conversation.id,
                sender=MessageSender.AI,
                text=ai_response
            )
            db.session.add(ai_message)
            db.session.commit()

        try:
            sentiment = analyze_sentiment(user_speech)
            sentiment_snapshot = SentimentSnapshot(
                patient_id=user_id,
                source=ConversationChannel.VOICE,
                rating=sentiment.rating,
                confidence=sentiment.confidence,
                message_id=user_message.id
            )
            db.session.add(sentiment_snapshot)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to analyze sentiment for voice: {e}")
        
        response = VoiceResponse()
        
        response.say(ai_response, voice='Polly.Joanna', language='en-US')
        
        gather = response.gather(
            input='speech',
            timeout=10,
            speechTimeout='auto',
            action=f'/voice/process?call_id={call_id}',
            method='POST'
        )
        
        gather.say("Please continue sharing. I'm here to listen.", 
                  voice='Polly.Joanna', language='en-US')
        
        response.say("Thank you for talking with me. Feel free to continue our conversation in the chat anytime. Take care.", 
                    voice='Polly.Joanna', language='en-US')
        
        return str(response), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        response = VoiceResponse()
        response.say("I apologize for the technical issue. Our conversation has been helpful, and you can always continue chatting with me online.", 
                    voice='Polly.Joanna', language='en-US')
        return str(response), 200, {'Content-Type': 'text/xml'}

@app.route('/api/available-doctors')
def api_available_doctors():
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized - patients only'}), 401
    
    try:
        doctors = User.query.filter_by(user_type=UserType.DOCTOR, is_active=True).all()
        
        doctor_list = []
        for doctor in doctors:
            patient_count = DoctorPatient.query.filter_by(
                doctor_id=doctor.id, is_active=True
            ).count()
            
            doctor_list.append({
                'id': doctor.id,
                'name': doctor.name,
                'email': doctor.email,
                'patient_count': patient_count,
                'specialization': 'Mental Health Specialist'
            })
        
        return jsonify({'doctors': doctor_list})
        
    except Exception as e:
        logger.error(f"Error fetching doctors: {e}")
        return jsonify({'error': 'Unable to fetch doctor list'}), 500

@app.route('/api/request-doctor', methods=['POST'])
def api_request_doctor():
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized - patients only'}), 401
    
    data = request.get_json() or {}
    doctor_id = data.get('doctor_id')
    
    if not doctor_id:
        return jsonify({'error': 'Doctor ID required'}), 400
    
    try:
        existing_assignment = DoctorPatient.query.filter_by(
            patient_id=session['user_id'], is_active=True
        ).first()
        
        if existing_assignment:
            return jsonify({'error': 'You already have an assigned doctor'}), 400
        
        doctor = User.query.filter_by(id=doctor_id, user_type=UserType.DOCTOR).first()
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        assignment = DoctorPatient(
            doctor_id=doctor_id,
            patient_id=session['user_id'],
            is_active=True,
            notes=f"Patient {session.get('user_name')} assigned to Dr. {doctor.name}"
        )
        
        db.session.add(assignment)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'doctor_name': doctor.name,
            'message': f'Successfully assigned to Dr. {doctor.name}'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error assigning doctor: {e}")
        return jsonify({'error': 'Failed to assign doctor'}), 500

@app.route('/api/my-patients')
def api_my_patients():
    if 'user_id' not in session or session.get('user_type') != 'doctor':
        return jsonify({'error': 'Unauthorized - doctors only'}), 401
    
    try:
        assignments = db.session.query(DoctorPatient).options(
            joinedload(DoctorPatient.patient)
        ).filter_by(doctor_id=session['user_id'], is_active=True).all()
        
        patients = []
        for assignment in assignments:
            patient = assignment.patient
            
            recent_sentiments = SentimentSnapshot.query.filter_by(
                patient_id=patient.id
            ).order_by(desc(SentimentSnapshot.created_at)).limit(5).all()
            
            active_alerts = Alert.query.filter_by(
                patient_id=patient.id, status='active'
            ).count()
            
            avg_mood = 3.0
            if recent_sentiments:
                avg_mood = sum(s.rating for s in recent_sentiments) / len(recent_sentiments)
            
            last_conversation = Conversation.query.filter_by(
                patient_id=patient.id
            ).order_by(desc(Conversation.started_at)).first()
            
            patients.append({
                'id': patient.id,
                'name': patient.name,
                'email': patient.email,
                'phone': patient.get_full_phone(),
                'avg_mood': round(avg_mood, 1),
                'active_alerts': active_alerts,
                'risk_level': 'high' if avg_mood < 2.0 or active_alerts > 0 else 'medium' if avg_mood < 3.0 else 'low',
                'last_interaction': last_conversation.started_at.isoformat() if last_conversation else None,
                'assigned_at': assignment.assigned_at.isoformat()
            })
        
        return jsonify({'patients': patients})
        
    except Exception as e:
        logger.error(f"Error fetching doctor's patients: {e}")
        return jsonify({'error': 'Unable to fetch patient data'}), 500

@app.route('/my-doctor')
def my_doctor():
    if 'user_id' not in session or session.get('user_type') != 'patient': return redirect(url_for('login'))
    return render_template('my_doctor.html')

@app.route('/api/patient/<int:patient_id>/analytics')
def api_patient_analytics(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401

    assignment = DoctorPatient.query.filter_by(
        doctor_id=session['user_id'],
        patient_id=patient_id,
        is_active=True
    ).first()

    if not assignment:
        return jsonify({'error': 'Patient not found or not assigned to you'}), 404

    try:
        total_interactions = Conversation.query.filter_by(patient_id=patient_id).count()

        sentiments = [s.rating for s in SentimentSnapshot.query.filter_by(patient_id=patient_id).all()]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        recent_conversations_db = db.session.query(Conversation).filter_by(
            patient_id=patient_id
        ).order_by(desc(Conversation.started_at)).limit(5).all()

        recent_conversations_list = []
        for convo in recent_conversations_db:
            user_msg = Message.query.filter_by(conversation_id=convo.id, sender=MessageSender.PATIENT).first()
            ai_msg = Message.query.filter_by(conversation_id=convo.id, sender=MessageSender.AI).first()
            recent_conversations_list.append({
                'timestamp': convo.started_at.isoformat(),
                'type': convo.channel.value,
                'user_message': user_msg.text if user_msg else "[No user message]",
                'ai_response': ai_msg.text if ai_msg else "[No AI response]"
            })
        
        return jsonify({
            'total_interactions': total_interactions,
            'avg_sentiment': avg_sentiment,
            'recent_conversations': recent_conversations_list
        })
        
    except Exception as e:
        logger.error(f"Error fetching patient analytics: {e}")
        return jsonify({'error': 'Unable to fetch analytics'}), 500

@app.route('/hooks/elevenlabs-status', methods=['POST'])
def elevenlabs_call_status_webhook():
    data = request.json
    logger.info(f"Received ElevenLabs status update: {data}")
    return jsonify(success=True), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
