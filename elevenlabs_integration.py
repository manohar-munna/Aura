import os
import logging
import requests
import json
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_AGENT_ID = os.environ.get('ELEVENLABS_AGENT_ID', 'NrwS4Bi2Qi8wsW1FhnKa')
ELEVENLABS_CRITICAL_ALERT_AGENT_ID = os.environ.get('ELEVENLABS_CRITICAL_ALERT_AGENT_ID')
ELEVENLABS_PHONE_ID = 'phnum_1301k52zn761ed4vrmvf0r6spv78'

try:
    from elevenlabs.client import ElevenLabs
    
    if ELEVENLABS_API_KEY:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("ElevenLabs client initialized successfully")
    else:
        client = None
        logger.warning("ElevenLabs API key not found. Voice features will be limited.")
except ImportError:
    client = None
    logger.warning("ElevenLabs not properly installed. Voice features will be limited.")

def create_conversation_with_agent(
    phone_number: str,
    agent_id: str = None,
    customer_id: str = None,
    system_prompt: str = None,
    dynamic_variables: dict = None,
    status_callback_url: str = None
):
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not configured")
        return None

    final_agent_id = agent_id or ELEVENLABS_AGENT_ID

    try:
        headers = {
            'xi-api-key': ELEVENLABS_API_KEY,
            'Content-Type': 'application/json'
        }

        payload = {
            'agent_id': final_agent_id,
            'agent_phone_number_id': ELEVENLABS_PHONE_ID,
            'to_number': phone_number,
        }

        if system_prompt:
            payload['system_prompt'] = system_prompt

        if dynamic_variables:
            payload['dynamic_variables'] = dynamic_variables
            logger.info(f"Sending dynamic variables: {list(dynamic_variables.keys())}")

        if customer_id:
            payload['conversation_initiation_client_data'] = {
                'customer_id': customer_id,
                'source': 'aura_mental_health_app'
            }

        if status_callback_url:
            payload['status_callback_url'] = status_callback_url

        debug_payload = payload.copy()
        if 'dynamic_variables' in debug_payload and 'conversation_context' in debug_payload['dynamic_variables']:
            debug_payload['dynamic_variables']['conversation_context'] = f"[{len(debug_payload['dynamic_variables']['conversation_context'])} chars]"
        logger.info(f"ElevenLabs API payload: {debug_payload}")

        response = requests.post(
            'https://api.elevenlabs.io/v1/convai/twilio/outbound-call',
            headers=headers,
            json=payload,
            timeout=20
        )

        if response.status_code in (200, 201):
            call_data = response.json()
            call_sid = call_data.get('callSid') or call_data.get('call_sid')
            conversation_id = call_data.get('conversation_id') or call_data.get('conversationId')
            logger.info(f"Started ElevenLabs call {call_sid} to {phone_number} with agent {final_agent_id}")
            if dynamic_variables:
                logger.info(f"Dynamic variables passed: {list(dynamic_variables.keys())}")
            return {'call_sid': call_sid, 'conversation_id': conversation_id, 'raw': call_data}
        else:
            logger.error(f"Failed to start call: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error creating ElevenLabs call: {e}")
        return None

def is_configured():
    return ELEVENLABS_API_KEY is not None and client is not None
