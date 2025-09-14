"""
Database models for Aura mental health analytics system
Implements comprehensive data structure for patient monitoring, sentiment analysis,
and critical alert management with HIPAA-compliant design principles.
"""
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import enum

db = SQLAlchemy()

class UserType(enum.Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"

class MessageSender(enum.Enum):
    PATIENT = "patient"
    AI = "ai"

class ConversationChannel(enum.Enum):
    CHAT = "chat"
    VOICE = "voice"

class AlertSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(enum.Enum):
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    PANIC = "panic"
    SEVERE_DEPRESSION = "severe_depression"
    ABUSE = "abuse"

class NotificationChannel(enum.Enum):
    SMS = "sms"
    VOICE = "voice"
    PUSH = "push"

class NotificationStatus(enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"

class User(db.Model):
    """User accounts for both patients and doctors"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    phone = Column(String(20), nullable=True)
    country_code = Column(String(5), nullable=True)
    user_type = Column(Enum(UserType), nullable=False, default=UserType.PATIENT)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    patient_profile = relationship("PatientProfile", back_populates="user", uselist=False)
    doctor_patients = relationship("DoctorPatient", foreign_keys="DoctorPatient.doctor_id", back_populates="doctor")
    patient_doctors = relationship("DoctorPatient", foreign_keys="DoctorPatient.patient_id", back_populates="patient")
    conversations = relationship("Conversation", back_populates="patient")
    # FIX: Explicitly define the foreign key for the relationship to resolve ambiguity
    alerts = relationship("Alert", back_populates="patient", foreign_keys="Alert.patient_id")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_full_phone(self):
        if self.country_code and self.phone:
            return f"{self.country_code}{self.phone}"
        return self.phone

class PatientProfile(db.Model):
    """Extended profile information for patients"""
    __tablename__ = 'patient_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    emergency_contact_name = Column(String(100))
    emergency_contact_phone = Column(String(20))
    consent_analytics = Column(Boolean, default=False)
    consent_doctor_sharing = Column(Boolean, default=False)
    consent_emergency_contact = Column(Boolean, default=False)
    baseline_mood_rating = Column(Float, default=3.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="patient_profile")

class DoctorPatient(db.Model):
    """Patient-doctor assignment relationships"""
    __tablename__ = 'doctor_patients'
    __table_args__ = (db.UniqueConstraint('doctor_id', 'patient_id', name='unique_doctor_patient'),)
    
    id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    
    # Relationships
    doctor = relationship("User", foreign_keys=[doctor_id])
    patient = relationship("User", foreign_keys=[patient_id])

class Conversation(db.Model):
    """Chat and voice conversation sessions"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    channel = Column(Enum(ConversationChannel), nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)
    duration_seconds = Column(Integer)
    provider_session_id = Column(String(100))  # ElevenLabs conversation ID, etc.
    
    # Relationships
    patient = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    transcripts = relationship("Transcript", back_populates="conversation")

class Message(db.Model):
    """Individual messages within conversations"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    sender = Column(Enum(MessageSender), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    message_metadata = Column(JSON)  # Additional data like response time, model used, etc.
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sentiment_snapshots = relationship("SentimentSnapshot", back_populates="message")

class Transcript(db.Model):
    """Voice conversation transcripts from external providers"""
    __tablename__ = 'transcripts'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    provider = Column(String(50))  # 'twilio', 'elevenlabs'
    provider_transcript_id = Column(String(100))
    transcript_text = Column(Text)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="transcripts")

class SentimentSnapshot(db.Model):
    """Sentiment analysis results for messages and transcripts"""
    __tablename__ = 'sentiment_snapshots'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    source = Column(Enum(ConversationChannel), nullable=False)
    rating = Column(Float, nullable=False)  # 1.0 to 5.0 scale
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    message_id = Column(Integer, ForeignKey('messages.id'), nullable=True)
    transcript_id = Column(Integer, ForeignKey('transcripts.id'), nullable=True)
    analysis_details = Column(JSON)  # Raw LLM response, keywords detected, etc.
    
    # Relationships
    patient = relationship("User")
    message = relationship("Message", back_populates="sentiment_snapshots")
    transcript = relationship("Transcript")

class MoodDailyAggregate(db.Model):
    """Daily mood aggregations for trend analysis"""
    __tablename__ = 'mood_daily_aggregates'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    avg_rating = Column(Float, nullable=False)
    min_rating = Column(Float, nullable=False)
    max_rating = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)  # Standard deviation
    message_count = Column(Integer, default=0)
    voice_minutes = Column(Integer, default=0)
    
    # Relationships
    patient = relationship("User")

class Alert(db.Model):
    """Critical mental health alerts requiring immediate attention"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    triggered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    rationale = Column(Text, nullable=False)  # Why this alert was triggered
    acknowledged_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    acknowledged_at = Column(DateTime)
    status = Column(String(20), default='active')  # active, acknowledged, resolved
    trigger_data = Column(JSON)  # Conversation snippets, sentiment scores, etc.
    
    # Relationships
    patient = relationship("User", back_populates="alerts", foreign_keys=[patient_id])
    acknowledger = relationship("User", foreign_keys=[acknowledged_by])
    notifications = relationship("NotificationLog", back_populates="alert")

class NotificationLog(db.Model):
    """Log of all notifications sent for alerts"""
    __tablename__ = 'notification_logs'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey('alerts.id'), nullable=False)
    channel = Column(Enum(NotificationChannel), nullable=False)
    to_user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    status = Column(Enum(NotificationStatus), default=NotificationStatus.PENDING)
    provider_sid = Column(String(100))  # Twilio call SID, message SID, etc.
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    delivered_at = Column(DateTime)
    message_content = Column(Text)  # Redacted/sanitized notification content
    
    # Relationships
    alert = relationship("Alert", back_populates="notifications")
    recipient = relationship("User")

class AuditLog(db.Model):
    """Audit trail for HIPAA compliance"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    actor_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    action = Column(String(50), nullable=False)  # 'view_patient', 'send_alert', etc.
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    redacted_details = Column(JSON)  # Non-PHI context information
    
    # Relationships
    actor = relationship("User")

class Consent(db.Model):
    """Patient consent management for different data uses"""
    __tablename__ = 'consents'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    scope = Column(String(100), nullable=False)  # 'analytics', 'doctor_sharing', 'emergency_contact'
    granted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime)
    consent_text = Column(Text)  # The actual consent language shown to user
    
    # Relationships
    patient = relationship("User")

def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
def create_sample_data():
    """Create sample data for testing (development only)"""
    # This would be used only in development
    pass