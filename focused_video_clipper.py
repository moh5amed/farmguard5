#!/usr/bin/env python3
"""
Focused Video Clipper Backend
A streamlined Flask backend for video clipping with AI-powered content selection.
Maintains the same input/output format and processing techniques as the original.
"""

import os
import json
import time
import re
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import concurrent.futures
from multiprocessing import Pool, cpu_count
import threading
import subprocess
import random
import shutil
import queue
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

# Flask and request handling
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Video processing
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, TextClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# AI and transcription
import google.generativeai as genai
import openai
from openai import OpenAI

# Audio processing
import librosa
import soundfile as sf

# Environment and configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request pacing and audio defaults
REQUEST_SUBMIT_DELAY_BASE = 0.01  # seconds - ultra-fast for speed
REQUEST_SUBMIT_DELAY_JITTER = 0.04  # seconds - minimal jitter
DEFAULT_AUDIO_SR = 8000  # 8 kHz mono - ultra-fast processing
DEFAULT_AUDIO_CHANNELS = 1
SILENCE_THRESHOLD_MEAN_ABS = 0.006  # more aggressive silence detection
CHUNK_TARGET_SIZE_MB = 4.0  # smaller chunks for faster transcription
MAX_CHUNK_DURATION = 30  # maximum 30 seconds per chunk
MIN_CHUNK_DURATION = 8   # minimum 8 seconds per chunk

# Render optimization constants
class RenderOptimizedConstants:
    MAX_WORKERS = 12  # More workers for faster processing
    CHUNK_SIZE_MB = 4.0  # Smaller chunks for faster transcription
    MAX_CHUNK_DURATION = 30

# Temporary directory for chunked uploads
TEMP_DIR = "/tmp" if os.name == 'posix' else os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Persistent processing system
class ProcessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingJob:
    job_id: str
    status: ProcessingStatus
    progress: int
    message: str
    start_time: float
    last_update: float
    video_path: str
    project_data: Dict[str, Any]
    current_step: str
    completed_steps: List[str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class PersistentProcessingManager:
    """Persistent processing manager that never resets and survives timeouts"""
    
    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.jobs_file = os.path.join(TEMP_DIR, "persistent_jobs.json")
        self.lock = threading.Lock()
        
        # Load existing jobs on startup
        self.load_jobs()
        
        # Start background worker
        self.start_worker()
        
        logger.info("🔄 Persistent Processing Manager initialized")
    
    def load_jobs(self):
        """Load existing jobs from disk"""
        try:
            if os.path.exists(self.jobs_file):
                with open(self.jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                    for job_id, job_data in jobs_data.items():
                        job_data['status'] = ProcessingStatus(job_data['status'])
                        self.jobs[job_id] = ProcessingJob(**job_data)
                logger.info(f"📂 Loaded {len(self.jobs)} existing jobs")
        except Exception as e:
            logger.error(f"❌ Failed to load jobs: {e}")
    
    def save_jobs(self):
        """Save jobs to disk"""
        try:
            with self.lock:
                jobs_data = {}
                for job_id, job in self.jobs.items():
                    jobs_data[job_id] = asdict(job)
                    jobs_data[job_id]['status'] = job.status.value
                
                with open(self.jobs_file, 'w') as f:
                    json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save jobs: {e}")
    
    def start_worker(self):
        """Start background worker thread"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("🚀 Background worker started")
    
    def _worker_loop(self):
        """Background worker loop that processes jobs"""
        while self.is_running:
            try:
                # Check for pending jobs
                pending_jobs = [job for job in self.jobs.values() 
                              if job.status == ProcessingStatus.PENDING]
                
                if pending_jobs:
                    # Process the oldest pending job
                    job = min(pending_jobs, key=lambda x: x.start_time)
                    self._process_job(job)
                
                # Save jobs periodically
                self.save_jobs()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                time.sleep(5)
    
    def _process_job(self, job: ProcessingJob):
        """Process a single job"""
        try:
            logger.info(f"🎬 Starting job {job.job_id}: {job.project_data.get('projectName', 'Unknown')}")
            
            # Update status
            job.status = ProcessingStatus.IN_PROGRESS
            job.message = "Starting video processing..."
            job.progress = 0
            job.last_update = time.time()
            self.save_jobs()
            
            # Process video using the video clipper
            video_clipper = FocusedVideoClipper()
            
            # Step 1: Extract audio
            job.current_step = "audio_extraction"
            job.message = "Extracting audio from video..."
            job.progress = 10
            job.last_update = time.time()
            self.save_jobs()
            
            viral_segments, full_transcript = video_clipper.extract_audio_segments(job.video_path)
            
            # Step 2: AI clip selection
            job.current_step = "ai_selection"
            job.message = "AI selecting best clips..."
            job.progress = 50
            job.last_update = time.time()
            self.save_jobs()
            
            num_clips = job.project_data.get('numClips', 3)
            frontend_inputs = {
                'projectName': job.project_data.get('projectName', ''),
                'description': job.project_data.get('description', ''),
                'aiPrompt': job.project_data.get('aiPrompt', ''),
                'targetPlatforms': job.project_data.get('targetPlatforms', ['tiktok']),
                'processingOptions': job.project_data.get('processingOptions', {})
            }
            
            viral_moments = video_clipper.ai_select_best_clips(
                viral_segments, full_transcript, num_clips, frontend_inputs
            )
            
            # Step 3: Generate clips
            job.current_step = "clip_generation"
            job.message = "Generating video clips..."
            job.progress = 70
            job.last_update = time.time()
            self.save_jobs()
            
            generated_clips = []
            for i, moment in enumerate(viral_moments):
                start_time = moment['start_time']
                duration = moment['duration']
                
                # Create descriptive filename
                safe_caption = re.sub(r'[^\w\s-]', '', moment['caption'])[:30]
                clip_name = f"viral_clip_{i+1}_{moment['viral_score']}_{safe_caption}.mp4"
                clip_name = clip_name.replace(' ', '_')
                
                # Extract processing options
                aspect_ratio_options = None
                watermark_options = None
                processing_options = job.project_data.get('processingOptions', {})
                
                if processing_options:
                    if 'targetAspectRatio' in processing_options:
                        aspect_ratio_options = {
                            'targetAspectRatio': processing_options.get('targetAspectRatio', '16:9'),
                            'preserveOriginal': processing_options.get('preserveOriginalAspectRatio', False),
                            'enableSmartCropping': processing_options.get('enableSmartCropping', True),
                            'enableLetterboxing': processing_options.get('enableLetterboxing', True)
                        }
                    
                    if 'watermarkOptions' in processing_options:
                        watermark_options = processing_options['watermarkOptions']
                
                clip_path = video_clipper.create_clip(
                    job.video_path, start_time, duration, clip_name, 
                    aspect_ratio_options, watermark_options
                )
                generated_clips.append(clip_path)
                
                # Update progress
                job.progress = 70 + int((i + 1) / len(viral_moments) * 20)
                job.message = f"Generated clip {i+1}/{len(viral_moments)}"
                job.last_update = time.time()
                self.save_jobs()
            
            # Complete job
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100
            job.message = f"Successfully generated {len(generated_clips)} clips"
            job.current_step = "completed"
            job.completed_steps = ["audio_extraction", "ai_selection", "clip_generation", "completed"]
            job.result = {
                'success': True,
                'clips_generated': len(generated_clips),
                'clips': [
                    {
                        'filename': os.path.basename(clip_path),
                        'filepath': clip_path,
                        'download_url': f'/api/download/{os.path.basename(clip_path)}'
                    } for clip_path in generated_clips
                ],
                'transcription': full_transcript,
                'processing_options': processing_options
            }
            job.last_update = time.time()
            self.save_jobs()
            
            logger.info(f"✅ Job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Job {job.job_id} failed: {e}")
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.message = f"Processing failed: {str(e)}"
            job.last_update = time.time()
            self.save_jobs()
    
    def create_job(self, video_path: str, project_data: Dict[str, Any]) -> str:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            progress=0,
            message="Job created, waiting to start...",
            start_time=time.time(),
            last_update=time.time(),
            video_path=video_path,
            project_data=project_data,
            current_step="pending",
            completed_steps=[]
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.save_jobs()
        
        logger.info(f"📝 Created job {job_id} for {project_data.get('projectName', 'Unknown')}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, ProcessingJob]:
        """Get all jobs"""
        return self.jobs.copy()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.IN_PROGRESS]:
                job.status = ProcessingStatus.CANCELLED
                job.message = "Job cancelled by user"
                job.last_update = time.time()
                self.save_jobs()
                return True
        return False

# Global persistent processing manager
persistent_manager = PersistentProcessingManager()

class FocusedVideoClipper:
    """Focused video clipper with AI-powered content selection"""
    
    def __init__(self, output_dir: str = "viral_clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize AI services
        self._setup_gemini()
        self._setup_whisper()
        
        # Processing settings
        self.min_clip_duration = 15
        self.max_clip_duration = 60
        self.default_num_clips = 3
        
        logger.info("🎬 Focused Video Clipper initialized")
    
    def _setup_gemini(self):
        """Setup Gemini AI configuration"""
        try:
            # For development: hardcoded API key
            # For production: use environment variable
            api_key = os.getenv('VITE_GEMINI_API_KEY') or "AIzaSyCIJC5xQbc6TXCLL_sJOhAr9UxkA4puZRM"
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Safety settings (same as original)
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            logger.info("✅ Gemini AI configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Gemini: {e}")
            raise
    
    def _setup_whisper(self):
        """Setup Whisper API configuration"""
        try:
            # For development: hardcoded API key
            # For production: use environment variable
            api_key = os.getenv('OPENAI_API_KEY') or "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"
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✅ Whisper API configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Whisper: {e}")
            raise
    
    def transcribe_with_whisper_api(self, audio_file_path: str) -> dict:
        """Transcribe audio using OpenAI Whisper API (same as original)"""
        try:
            logger.info("🤖 Using OpenAI Whisper API for transcription...")
            
            # Check file size (Whisper API has 25MB limit)
            file_size = os.path.getsize(audio_file_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"📊 File size: {file_size_mb:.2f} MB")
            
            # If file is too large, use chunking approach
            if file_size > 20 * 1024 * 1024:  # 20MB limit for safety (reduced from 23MB)
                logger.info("⚠️ File too large for single Whisper API request")
                return self._transcribe_large_audio_with_chunking(audio_file_path)
            
            # Precompress small/medium files for faster upload if not already small
            precompressed_path = None
            try:
                if (file_size_mb > 4) or (not audio_file_path.lower().endswith('.m4a')):  # More aggressive compression
                    precompressed_path = self._precompress_audio_for_whisper(audio_file_path)
                    if precompressed_path and os.path.exists(precompressed_path):
                        compressed_size_mb = os.path.getsize(precompressed_path) / (1024 * 1024)
                        logger.info(f"📦 Precompressed audio to {compressed_size_mb:.2f} MB for faster upload")
                        audio_file_path = precompressed_path
            except Exception as e:
                logger.warning(f"⚠️ Precompression skipped due to error: {e}")
                precompressed_path = None

            # Single file transcription
            try:
                return self._transcribe_single_audio_file(audio_file_path)
            finally:
                # Cleanup temp precompressed file
                if precompressed_path and os.path.exists(precompressed_path):
                    try:
                        os.remove(precompressed_path)
                    except Exception:
                        pass
                
        except Exception as e:
            logger.error(f"❌ Whisper API transcription failed: {str(e)}")
            raise e
    
    def _calculate_chunk_duration(self, chunk_path: str) -> float:
        """Calculate chunk duration in parallel"""
        try:
            chunk_audio_data, chunk_sr = librosa.load(chunk_path, sr=None, mono=True)
            return len(chunk_audio_data) / chunk_sr
        except:
            return 3 * 60  # Default to 3 minutes if calculation fails

    def _transcribe_single_audio_file(self, audio_file_path: str) -> dict:
        """Transcribe a single audio file with Whisper API with timeout and retry logic"""
        max_retries = 3
        timeout_seconds = 300  # 5 minute timeout per request for large files
        
        for attempt in range(max_retries):
            try:
                # Final size check before sending to Whisper API
                file_size = os.path.getsize(audio_file_path)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size > 25 * 1024 * 1024:  # 25MB hard limit
                    raise ValueError(f"File size {file_size_mb:.2f}MB exceeds Whisper API 25MB limit")
                
                logger.info(f"📊 Final file size check: {file_size_mb:.2f}MB (under 25MB limit)")
                
                # Open the audio file and send to Whisper API
                with open(audio_file_path, 'rb') as audio_file:
                    logger.info(f"🚀 Sending audio to Whisper API... (attempt {attempt + 1}/{max_retries})")
                    
                    # Use Whisper API with speed-optimized settings and timeout
                    response = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",  # Get detailed response with timestamps
                        temperature=0.0,  # Deterministic results
                        language="en",  # You can make this configurable
                        prompt="",  # Empty prompt for faster processing
                        timeout=timeout_seconds  # Add timeout for faster failure handling
                    )
                
                logger.info("✅ Whisper API transcription completed!")
                logger.info(f"📝 Transcription length: {len(response.text)} characters")
                
                # Convert API response to match local Whisper format
                result = {
                    'text': response.text,
                    'segments': response.segments if hasattr(response, 'segments') else [],
                    'words': []
                }
                
                # Extract word-level timestamps if available
                if hasattr(response, 'segments') and response.segments:
                    for segment in response.segments:
                        if hasattr(segment, 'words') and segment.words:
                            for word in segment.words:
                                result['words'].append({
                                    'word': word.word,
                                    'start': word.start,
                                    'end': word.end,
                                    'probability': getattr(word, 'probability', 1.0)
                                })
                
                logger.info(f"🎯 Extracted {len(result['words'])} word timestamps")
                return result
                
            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ All {max_retries} attempts failed for {audio_file_path}")
                    raise
                else:
                    # Exponential backoff with jitter; slightly longer on rate limits
                    base = 1.5 ** attempt
                    jitter = random.random() * 0.5
                    backoff = min(8.0, base + jitter)
                    # If looks like rate limit, back off a bit more
                    err_text = str(e).lower()
                    if 'rate limit' in err_text or '429' in err_text:
                        backoff = min(12.0, backoff + 1.0)
                    logger.info(f"🔄 Retrying in {backoff:.2f}s... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(backoff)

    def _precompress_audio_for_whisper(self, audio_file_path: str) -> Optional[str]:
        """Precompress input audio to mono 16 kHz m4a (AAC 32k) for faster upload.
        Returns path to temp file, or None if compression failed/not available.
        """
        try:
            if shutil.which('ffmpeg') is None:
                # Fallback: return None; caller will use original
                return None
            timestamp = int(time.time() * 1000)
            temp_out = f"temp_precomp_{timestamp}.m4a"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', audio_file_path,
                '-ac', str(DEFAULT_AUDIO_CHANNELS),
                '-ar', str(DEFAULT_AUDIO_SR),
                '-c:a', 'aac', '-b:a', '24k', '-strict', '-2',  # ultra-low bitrate for maximum speed
                temp_out
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            return temp_out
        except Exception as e:
            logger.warning(f"⚠️ Precompression failed: {e}")
            return None
    
    def _transcribe_large_audio_with_chunking(self, audio_file_path: str) -> dict:
        """Transcribe large audio file by chunking it into smaller pieces (same as original)"""
        try:
            logger.info("🔄 Starting chunked transcription for large audio file...")
            
            # Create audio chunks
            audio_chunks = self._create_audio_chunks(audio_file_path)
            logger.info(f"✅ Created {len(audio_chunks)} audio chunks")
            
            # Transcribe each chunk
            all_transcriptions = []
            all_segments = []
            all_words = []
            chunk_offset = 0.0  # Track time offset for each chunk
            
            # Use parallel processing for maximum speed
            logger.info(f"🚀 Starting parallel transcription of {len(audio_chunks)} chunks...")
            start_time = time.time()
            
            # Use ThreadPoolExecutor for parallel processing with aggressive concurrency
            # Use more workers for faster processing (API can handle it with our pacing)
            aggressive_max = min(cpu_count() * 4, 16)  # Even more aggressive parallelization
            max_workers = min(len(audio_chunks), aggressive_max)
            logger.info(f"🔧 Using {max_workers} parallel workers (aggressive parallelization)")
            
            # Preprocess chunks in parallel for faster timestamp calculation
            logger.info("🔧 Preprocessing chunks in parallel for faster processing...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(audio_chunks), 8)) as preprocess_executor:
                # Submit all chunk duration calculations
                future_to_chunk = {}
                for chunk_path in audio_chunks:
                    future = preprocess_executor.submit(self._calculate_chunk_duration, chunk_path)
                    future_to_chunk[future] = chunk_path
                
                # Collect results
                chunk_durations = []
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_durations.append(future.result())
            
            # Calculate cumulative offsets for each chunk
            chunk_offsets = [0.0]
            for duration in chunk_durations[:-1]:  # Exclude last chunk
                chunk_offsets.append(chunk_offsets[-1] + duration)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk transcription tasks with small delays to avoid rate limits
                future_to_chunk = {}
                for i, chunk_path in enumerate(audio_chunks):
                    # Add jittered delay between submissions to avoid thundering herd
                    if i > 0:
                        delay = REQUEST_SUBMIT_DELAY_BASE + random.random() * REQUEST_SUBMIT_DELAY_JITTER
                        time.sleep(delay)
                    future = executor.submit(self._transcribe_single_audio_file, chunk_path)
                    future_to_chunk[future] = (i, chunk_path, chunk_offsets[i])
                
                # Process completed tasks as they finish
                completed_chunks = 0
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_index, chunk_path, chunk_offset = future_to_chunk[future]
                    completed_chunks += 1
                    
                    try:
                        chunk_result = future.result()
                        logger.info(f"✅ Completed chunk {chunk_index + 1}/{len(audio_chunks)} ({completed_chunks}/{len(audio_chunks)} total)")
                        
                        # Adjust timestamps for chunk offset
                        adjusted_segments = []
                        adjusted_words = []
                        
                        if chunk_result['segments']:
                            for segment in chunk_result['segments']:
                                # Handle both dict and TranscriptionSegment object
                                if hasattr(segment, 'start'):
                                    # TranscriptionSegment object
                                    adjusted_segment = {
                                        'start': segment.start + chunk_offset,
                                        'end': segment.end + chunk_offset,
                                        'text': segment.text
                                    }
                                else:
                                    # Dictionary
                                    adjusted_segment = {
                                        'start': segment['start'] + chunk_offset,
                                        'end': segment['end'] + chunk_offset,
                                        'text': segment['text']
                                    }
                                adjusted_segments.append(adjusted_segment)
                        
                        if chunk_result['words']:
                            for word in chunk_result['words']:
                                # Handle both dict and Word object
                                if hasattr(word, 'word'):
                                    # Word object
                                    adjusted_word = {
                                        'word': word.word,
                                        'start': word.start + chunk_offset,
                                        'end': word.end + chunk_offset,
                                        'probability': getattr(word, 'probability', 1.0)
                                    }
                                else:
                                    # Dictionary
                                    adjusted_word = {
                                        'word': word['word'],
                                        'start': word['start'] + chunk_offset,
                                        'end': word['end'] + chunk_offset,
                                        'probability': word.get('probability', 1.0)
                                    }
                                adjusted_words.append(adjusted_word)
                        
                        all_transcriptions.append(chunk_result['text'])
                        all_segments.extend(adjusted_segments)
                        all_words.extend(adjusted_words)
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to transcribe chunk {chunk_index + 1}: {e}")
                    
                    finally:
                        # Clean up chunk file (with retry for Windows file locking)
                        try:
                            os.remove(chunk_path)
                        except PermissionError:
                            # Windows file locking issue - try again after a short delay
                            time.sleep(0.1)
                            try:
                                os.remove(chunk_path)
                            except:
                                logger.warning(f"⚠️ Could not delete chunk file: {chunk_path}")
                        except:
                            pass
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Parallel transcription completed in {elapsed_time:.1f} seconds")
            
            # Combine all results
            combined_result = {
                'text': ' '.join(all_transcriptions),
                'segments': all_segments,
                'words': all_words
            }
            
            logger.info(f"✅ Chunked transcription completed: {len(all_segments)} segments, {len(all_words)} words")
            return combined_result
            
        except Exception as e:
            logger.error(f"❌ Chunked transcription failed: {str(e)}")
            raise e
    
    def _create_audio_chunks(self, audio_file_path: str) -> list:
        """Create audio chunks for large file processing with strong compression (mono 16 kHz).

        Strategy:
        - Load once, convert to mono 16 kHz to minimize size.
        - Slice into short chunks (10–45s) targeting ~1–2 MB per chunk.
        - Save as m4a (AAC) via ffmpeg for best size/quality; fallback to WAV PCM16 if ffmpeg missing.
        """
        try:
            # Clean up any leftover temp files first
            import glob
            temp_files = glob.glob("temp_chunk_*.wav")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    logger.info(f"🧹 Cleaned up leftover temp file: {temp_file}")
                except Exception:
                    pass
            # Load audio with librosa and convert to mono 16kHz (fast upload/transcription)
            audio_data, sample_rate = librosa.load(audio_file_path, sr=DEFAULT_AUDIO_SR, mono=True)
            duration = len(audio_data) / sample_rate
            
            # Get original file size to calculate optimal chunk duration
            original_file_size = os.path.getsize(audio_file_path)
            logger.info(f"📊 Original file size: {original_file_size / (1024*1024):.2f} MB")
            
            # Calculate target chunk duration based on file size
            # Target ~0.8MB per chunk (post-compression) for ultra-fast processing
            target_chunk_size_mb = CHUNK_TARGET_SIZE_MB
            
            # Estimate compression ratio based on file type and size
            if original_file_size > 100 * 1024 * 1024:  # Large files (>100MB)
                estimated_compression_ratio = 0.25  # More compression for large files
            elif original_file_size > 50 * 1024 * 1024:  # Medium files (50-100MB)
                estimated_compression_ratio = 0.3
            else:  # Smaller files
                estimated_compression_ratio = 0.4
            
            # Calculate target duration: (target_size / original_size) * duration / compression_ratio
            target_chunk_duration = (target_chunk_size_mb * 1024 * 1024) / (original_file_size / duration) / estimated_compression_ratio

            # Clamp between 8 seconds and 30 seconds for ultra-fast processing
            chunk_duration = max(MIN_CHUNK_DURATION, min(MAX_CHUNK_DURATION, target_chunk_duration))
            
            logger.info(f"🎯 Calculated chunk duration: {chunk_duration/60:.1f} minutes")
            
            # Create chunks in parallel for maximum speed
            chunks = []
            chunk_count = 0
            
            chunk_duration_int = int(chunk_duration)
            chunk_tasks = []
            
            # Prepare chunk creation tasks
            for i in range(0, int(duration), chunk_duration_int):
                start_sample = i * sample_rate
                end_sample = min((i + chunk_duration_int) * sample_rate, len(audio_data))
                chunk_tasks.append((start_sample, end_sample, i))
            
            # Process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunk_tasks), 8)) as executor:
                future_to_chunk = {}
                
                for start_sample, end_sample, i in chunk_tasks:
                    future = executor.submit(self._create_single_chunk, 
                                          audio_data[start_sample:end_sample], 
                                          i, chunk_count)
                    future_to_chunk[future] = (start_sample, end_sample, i)
                    chunk_count += 1
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_path = future.result()
                        if chunk_path:  # Skip None results (silent chunks)
                            chunks.append(chunk_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Chunk creation failed: {e}")
                        continue
            
            logger.info(f"✅ Created {len(chunks)} audio chunks with {chunk_duration:.1f}s duration each (mono {DEFAULT_AUDIO_SR//1000}kHz)")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to create audio chunks: {e}")
            raise e
    
    def _create_single_chunk(self, chunk_data, chunk_index, chunk_count):
        """Create a single audio chunk with compression and silence detection"""
        try:
            # Skip mostly-silent chunks to avoid unnecessary API calls
            if len(chunk_data) == 0:
                return None
            mean_abs = float(np.mean(np.abs(chunk_data)))
            if mean_abs < SILENCE_THRESHOLD_MEAN_ABS and len(chunk_data) > (DEFAULT_AUDIO_SR * 3):
                logger.info(f"🔇 Skipping silent chunk {chunk_count} (mean|x|={mean_abs:.4f})")
                return None
            
            # Save chunk with strong compression using ffmpeg (m4a). Fallback to wav PCM16 if ffmpeg unavailable.
            use_ffmpeg = shutil.which('ffmpeg') is not None
            if use_ffmpeg:
                wav_tmp_path = f"temp_chunk_{chunk_count}.wav"
                m4a_path = f"temp_chunk_{chunk_count}.m4a"
                # Write PCM WAV @8k mono first (fast)
                sf.write(wav_tmp_path, chunk_data, DEFAULT_AUDIO_SR, subtype='PCM_16')
                # Encode to AAC LC with ultra-low bitrate for maximum speed
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-i', wav_tmp_path,
                    '-ac', str(DEFAULT_AUDIO_CHANNELS),
                    '-ar', str(DEFAULT_AUDIO_SR),
                    '-c:a', 'aac', '-b:a', '24k', '-strict', '-2',  # ultra-low bitrate for maximum speed
                    m4a_path
                ]
                try:
                    subprocess.run(ffmpeg_cmd, check=True)
                    os.remove(wav_tmp_path)
                    chunk_path = m4a_path
                except Exception as ffm_err:
                    logger.warning(f"⚠️ ffmpeg encode failed, falling back to WAV: {ffm_err}")
                    chunk_path = wav_tmp_path
            else:
                chunk_path = f"temp_chunk_{chunk_count}.wav"
                sf.write(chunk_path, chunk_data, DEFAULT_AUDIO_SR, subtype='PCM_16')

            # Validate size
            file_size = os.path.getsize(chunk_path)
            logger.info(f"📦 Chunk {chunk_count} size: {file_size / (1024*1024):.2f} MB")
            return chunk_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create chunk {chunk_count}: {e}")
            return None
    
    def extract_audio_segments(self, video_path: str):
        """Extract and analyze audio segments for viral content (same as original)"""
        try:
            logger.info("🎵 Extracting audio from video...")
            
            # Extract audio from video (prefer mono 16kHz for speed)
            video = VideoFileClip(video_path)
            audio_path = "temp_audio.wav"
            try:
                video.audio.write_audiofile(
                    audio_path,
                    fps=DEFAULT_AUDIO_SR,
                    nbytes=2,
                    verbose=False,
                    logger=None
                )
            except Exception:
                # Fallback to default parameters if not supported
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            # Transcribe audio
            whisper_result = self.transcribe_with_whisper_api(audio_path)
            
            # Create viral segments from transcription
            viral_segments = self._create_viral_segments_from_transcription(whisper_result)
            
            # Clean up temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"✅ Extracted {len(viral_segments)} viral segments")
            return viral_segments, whisper_result['text']
            
        except Exception as e:
            logger.error(f"❌ Failed to extract audio segments: {e}")
            raise e
    
    def _create_viral_segments_from_transcription(self, whisper_result: dict) -> list:
        """Create viral segments from transcription data (same as original)"""
        try:
            segments = []
            words = whisper_result.get('words', [])
            
            if not words:
                # Fallback: create segments from text
                text = whisper_result.get('text', '')
                if text:
                    # Create basic segments every 30 seconds
                    duration = 30  # Default duration
                    for i in range(0, len(text), 500):  # Every 500 characters
                        start_time = (i // 500) * duration
                        end_time = start_time + duration
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text[i:i+500],
                            'viral_score': 7  # Default score
                        })
                return segments
            
            # Create segments based on word timestamps
            current_segment = []
            segment_start = 0
            segment_duration = 30  # 30 seconds per segment
            
            for word in words:
                if not current_segment:
                    segment_start = word['start']
                
                current_segment.append(word)
                
                # Check if we should end this segment
                if word['end'] - segment_start >= segment_duration:
                    # Create segment
                    segment_text = ' '.join([w['word'] for w in current_segment])
                    segments.append({
                        'start': segment_start,
                        'end': word['end'],
                        'text': segment_text,
                        'viral_score': self._calculate_viral_score(segment_text)
                    })
                    
                    # Start new segment
                    current_segment = []
                    segment_start = word['end']
            
            # Add remaining words as final segment
            if current_segment:
                segment_text = ' '.join([w['word'] for w in current_segment])
                segments.append({
                    'start': segment_start,
                    'end': current_segment[-1]['end'],
                    'text': segment_text,
                    'viral_score': self._calculate_viral_score(segment_text)
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Failed to create viral segments: {e}")
            return []
    
    def _calculate_viral_score(self, text: str) -> int:
        """Calculate viral score for text segment (same as original)"""
        try:
            score = 5  # Base score
            
            # Viral keywords and phrases
            viral_keywords = [
                'amazing', 'incredible', 'unbelievable', 'shocking', 'mind-blowing',
                'you won\'t believe', 'wait for it', 'this is crazy', 'insane',
                'viral', 'trending', 'hack', 'secret', 'trick', 'tip',
                'game changer', 'life changing', 'must see', 'watch this'
            ]
            
            text_lower = text.lower()
            
            # Check for viral keywords
            for keyword in viral_keywords:
                if keyword in text_lower:
                    score += 1
            
            # Check for emotional words
            emotional_words = ['love', 'hate', 'angry', 'excited', 'scared', 'surprised']
            for word in emotional_words:
                if word in text_lower:
                    score += 1
            
            # Check for questions (engagement)
            if '?' in text:
                score += 1
            
            # Check for exclamations (excitement)
            if '!' in text:
                score += 1
            
            # Cap the score
            return min(score, 10)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate viral score: {e}")
            return 5
    
    def create_ultra_advanced_ai_prompt(self, condensed_transcript, viral_segments, num_clips_requested, frontend_inputs=None):
        """Create ultra-advanced AI prompt with strict user instruction adherence (same as original)"""
        
        # Get user's specific AI instructions
        user_ai_prompt = frontend_inputs.get('aiPrompt', '') if frontend_inputs else ''
        
        # Build strict user instruction context
        user_instructions = self._build_strict_user_instructions(frontend_inputs)
        
        prompt = f"""
        🚨 **STRICT USER INSTRUCTION COMPLIANCE SYSTEM**
        
        You are an AI content analyzer that MUST follow user instructions EXACTLY.
        **USER INSTRUCTIONS ARE ABSOLUTE AND OVERRIDE ALL DEFAULT BEHAVIORS.**
        
        📋 **USER'S SPECIFIC AI INSTRUCTIONS (MANDATORY TO FOLLOW):**
        {user_ai_prompt if user_ai_prompt.strip() else 'No specific AI instructions provided - use standard viral content selection.'}
        
        📊 **USER CONTEXT & REQUIREMENTS:**
        {user_instructions}
        
        📏 **VIDEO DURATION:** {frontend_inputs.get('video_duration', 'Unknown')} seconds
        ⚠️ **IMPORTANT:** Ensure all clip durations fit within the video length!
        
        🎯 **PRIMARY MISSION:**
        Extract EXACTLY {num_clips_requested} clips that follow the user's instructions above.
        If user instructions conflict with viral content best practices, USER INSTRUCTIONS WIN.
        
        📝 **TRANSCRIPT DATA:**
        {condensed_transcript}

        ---  

        🚨 **STRICT COMPLIANCE RULES:**
        
        1. **USER INSTRUCTIONS ARE PRIORITY #1** - Follow them exactly as specified
        2. **IGNORE DEFAULT VIRAL RULES** if they conflict with user instructions
        3. **ADAPT SELECTION CRITERIA** to match user's specific requirements
        4. **PRIORITIZE USER PREFERENCES** over generic viral content formulas
        
        ---
        
        🧠 **SELECTION PROCESS (USER-CENTRIC):**
        
        STEP 1: **ANALYZE USER INSTRUCTIONS**
        - What specific content type does the user want?
        - What style, tone, or focus areas are mentioned?
        - What platforms or audiences are targeted?
        
        STEP 2: **APPLY USER CRITERIA TO TRANSCRIPT**
        - Filter segments based on user's specific requirements
        - Ignore segments that don't match user instructions
        - Prioritize content that aligns with user's vision
        
        STEP 3: **VALIDATE AGAINST USER REQUIREMENTS**
        - Does each selected clip meet user's criteria?
        - Are the clips the type of content the user requested?
        - Does the selection match user's stated preferences?
        
        STEP 4: **QUALITY CONTROL FOR USER SATISFACTION**
        - Ensure clips start/end at natural speech boundaries
        - Verify content quality meets user's standards
        - Confirm selection aligns with user's goals
        
        ---
        
        🔍 **SELECTION PRIORITY (USER-FIRST):**
        
        1. **USER INSTRUCTIONS COMPLIANCE** - Must follow exactly what user requested
        2. **CONTENT RELEVANCE** - Must match user's specified content type/style
        3. **TECHNICAL QUALITY** - Clean audio cuts, proper timing
        4. **USER SATISFACTION** - Content that meets user's stated goals
        
        ---
        
        🚨 **MANDATORY COMPLIANCE CHECKS:**
        
        - ✅ Does each clip follow user's specific instructions?
        - ✅ Is the content type/style what the user requested?
        - ✅ Are the clips relevant to user's stated goals?
        - ✅ Does the selection prioritize user preferences over generic rules?
        
        ---
        
        ✅ **REQUIRED OUTPUT FORMAT:**
        
        {{
            "selected_clips": [
                {{
                    "start_time": <float, 2 decimals>,
                    "end_time": <float, 2 decimals>,
                    "duration": <float>,
                    "viral_score": <int 1-10>,
                    "content_type": "<string - based on user instructions>",
                    "viral_factor": "<string - why this matches user requirements>",
                    "engagement_potential": "<string - how it serves user's goals>",
                    "caption": "<string - optimized for user's target audience>",
                    "hashtags": ["<string>", "<string>", ...],
                    "hook_line": "<string - based on user's style preferences>",
                    "call_to_action": "<string - aligned with user's goals>",
                    "thumbnail_suggestion": "<string - matches user's aesthetic>",
                    "target_audience": "<string - from user's specifications>",
                    "platforms": ["<string>", "<string>", ...],
                    "optimal_posting_time": "<string>",
                    "cross_platform_adaptation": "<string>",
                    "segment_text": "<string>",
                    "reasoning": "<string - explain how this follows user instructions>",
                    "confidence_score": <float 0.0-1.0>,
                    "user_compliance_score": <int 1-10 - how well it follows user instructions>
                }}
            ]
        }}

        ---  

        🧠 **FINAL COMPLIANCE VERIFICATION:**
        
        Before responding, verify:
        - 🔴 **EVERY clip follows user's specific instructions**
        - 🔴 **Content type/style matches user's requirements**
        - 🔴 **Selection criteria prioritize user preferences**
        - 🔴 **Output format is 100% valid JSON**
        
        **REMEMBER: User instructions are LAW. Follow them exactly, even if they conflict with viral content best practices.**
        """

        return prompt
    
    def _build_strict_user_instructions(self, frontend_inputs):
        """Build strict, directive user instructions that override default behaviors (same as original)"""
        if not frontend_inputs:
            return "**NO USER INSTRUCTIONS PROVIDED** - Use standard viral content selection criteria."
        
        strict_instructions = []
        
        # Project-specific requirements
        if frontend_inputs.get('projectName'):
            strict_instructions.append(f"**PROJECT NAME**: {frontend_inputs['projectName']}")
        
        if frontend_inputs.get('description'):
            strict_instructions.append(f"**PROJECT DESCRIPTION**: {frontend_inputs['description']}")
        
        # Processing options that override defaults
        processing_options = frontend_inputs.get('processingOptions', {})
        
        if processing_options.get('targetDuration'):
            strict_instructions.append(f"**TARGET DURATION**: {processing_options['targetDuration']} seconds - CLIPS MUST BE THIS LENGTH")
        
        if processing_options.get('minDuration'):
            strict_instructions.append(f"**MINIMUM DURATION**: {processing_options['minDuration']} seconds - NO CLIPS SHORTER THAN THIS")
        
        if processing_options.get('maxDuration'):
            strict_instructions.append(f"**MAXIMUM DURATION**: {processing_options['maxDuration']} seconds - NO CLIPS LONGER THAN THIS")
        
        if processing_options.get('quality'):
            strict_instructions.append(f"**QUALITY REQUIREMENT**: {processing_options['quality']} quality processing - prioritize content that works with this setting")
        
        # Target platforms
        if frontend_inputs.get('targetPlatforms'):
            platforms = frontend_inputs['targetPlatforms']
            if isinstance(platforms, list):
                strict_instructions.append(f"**TARGET PLATFORMS**: {', '.join(platforms)} - OPTIMIZE CONTENT FOR THESE SPECIFIC PLATFORMS")
            else:
                strict_instructions.append(f"**TARGET PLATFORM**: {platforms} - OPTIMIZE CONTENT FOR THIS SPECIFIC PLATFORM")
        
        # Style and tone preferences
        if frontend_inputs.get('style'):
            style = frontend_inputs['style'].lower()
            if style == 'funny':
                strict_instructions.append("**STYLE REQUIREMENT**: HUMOROUS CONTENT ONLY - Focus on comedy, jokes, funny situations, and laugh-out-loud moments")
            elif style == 'dramatic':
                strict_instructions.append("**STYLE REQUIREMENT**: DRAMATIC CONTENT ONLY - Focus on emotional impact, suspense, intense moments, and powerful storytelling")
            elif style == 'educational':
                strict_instructions.append("**STYLE REQUIREMENT**: EDUCATIONAL CONTENT ONLY - Focus on learning, insights, knowledge sharing, and valuable information")
            elif style == 'inspirational':
                strict_instructions.append("**STYLE REQUIREMENT**: INSPIRATIONAL CONTENT ONLY - Focus on motivation, uplifting messages, positive energy, and life-changing insights")
        
        if frontend_inputs.get('tone'):
            tone = frontend_inputs['tone'].lower()
            if tone == 'professional':
                strict_instructions.append("**TONE REQUIREMENT**: PROFESSIONAL TONE ONLY - Formal, business-like, authoritative, and polished content")
            elif tone == 'casual':
                strict_instructions.append("**TONE REQUIREMENT**: CASUAL TONE ONLY - Relaxed, conversational, friendly, and approachable content")
            elif tone == 'energetic':
                strict_instructions.append("**TONE REQUIREMENT**: ENERGETIC TONE ONLY - High-energy, enthusiastic, dynamic, and exciting content")
            elif tone == 'calm':
                strict_instructions.append("**TONE REQUIREMENT**: CALM TONE ONLY - Peaceful, soothing, relaxed, and tranquil content")
        
        # Target audience specifications
        if frontend_inputs.get('target_audience'):
            audience = frontend_inputs['target_audience'].lower()
            if audience == 'gen_z':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: GEN Z ONLY - Content must appeal to 16-24 year olds: trend-aware, authentic, social justice focused, meme culture")
            elif audience == 'millennials':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: MILLENNIALS ONLY - Content must appeal to 25-40 year olds: nostalgic, career-focused, work-life balance, practical solutions")
            elif audience == 'gen_x':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: GEN X ONLY - Content must appeal to 41-56 year olds: practical, family-oriented, value-conscious, established professionals")
        
        # Content focus areas
        if frontend_inputs.get('content_focus'):
            focus = frontend_inputs['content_focus'].lower()
            if focus == 'entertainment':
                strict_instructions.append("**CONTENT FOCUS**: ENTERTAINMENT ONLY - Prioritize fun, engaging, and enjoyable content over educational or informational")
            elif focus == 'education':
                strict_instructions.append("**CONTENT FOCUS**: EDUCATION ONLY - Prioritize learning, insights, and knowledge sharing over pure entertainment")
            elif focus == 'business':
                strict_instructions.append("**CONTENT FOCUS**: BUSINESS ONLY - Prioritize professional insights, career advice, and business strategies")
            elif focus == 'lifestyle':
                strict_instructions.append("**CONTENT FOCUS**: LIFESTYLE ONLY - Prioritize personal development, health, relationships, and daily life improvements")
        
        # Duration preferences
        if frontend_inputs.get('duration_preference'):
            duration = frontend_inputs['duration_preference'].lower()
            if duration == 'short':
                strict_instructions.append("**DURATION REQUIREMENT**: SHORT FORMAT ONLY - 15-30 seconds, quick impact, scroll-stopping content")
            elif duration == 'medium':
                strict_instructions.append("**DURATION REQUIREMENT**: MEDIUM FORMAT ONLY - 30-60 seconds, story development, engagement building")
            elif duration == 'long':
                strict_instructions.append("**DURATION REQUIREMENT**: LONG FORMAT ONLY - 60+ seconds, deep dive, comprehensive content")
        
        # Custom instructions
        if frontend_inputs.get('custom_instructions'):
            strict_instructions.append(f"**CUSTOM REQUIREMENT**: {frontend_inputs['custom_instructions']}")
        
        if strict_instructions:
            return "\n".join(strict_instructions)
        else:
            return "**NO SPECIFIC USER REQUIREMENTS** - Use standard viral content selection criteria."
    
    def ai_select_best_clips(self, viral_segments, full_transcript, num_clips_requested, frontend_inputs=None):
        """AI-powered clip selection using Gemini (same as original)"""
        try:
            logger.info(f"🤖 AI selecting {num_clips_requested} best clips...")
            
            # Preprocess transcription for AI
            condensed_transcript = self._preprocess_transcription_for_ai(full_transcript, viral_segments)
            
            # Create AI prompt
            prompt = self.create_ultra_advanced_ai_prompt(condensed_transcript, viral_segments, num_clips_requested, frontend_inputs)
            
            # Generate content with Gemini
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,  # Creative generation
                    top_p=0.95,      # High quality
                    top_k=50,        # Diverse options
                    max_output_tokens=5000
                )
            )
            
            # Parse response
            try:
                cleaned_response = self._clean_ai_response(response.text)
                
                if isinstance(cleaned_response, dict):
                    selected_clips = cleaned_response.get('selected_clips', [])
                else:
                    selected_clips = json.loads(cleaned_response).get('selected_clips', [])
                
                logger.info(f"✅ AI selected {len(selected_clips)} clips")
                
                # If AI didn't select any clips, use fallback
                if not selected_clips:
                    logger.warning("⚠️ AI selected no clips, using fallback selection")
                    return self._fallback_clip_selection(viral_segments, num_clips_requested)
                
                return selected_clips
                
            except Exception as e:
                logger.error(f"❌ Failed to parse AI response: {e}")
                # Fallback to simple selection
                return self._fallback_clip_selection(viral_segments, num_clips_requested)
                
        except Exception as e:
            logger.error(f"❌ AI clip selection failed: {e}")
            # Fallback to simple selection
            return self._fallback_clip_selection(viral_segments, num_clips_requested)
    
    def _preprocess_transcription_for_ai(self, full_transcript, viral_segments):
        """Preprocess transcription for AI analysis (same as original)"""
        try:
            # Create condensed version with key segments
            condensed_parts = []
            
            # Add high-scoring segments
            high_score_segments = [seg for seg in viral_segments if seg.get('viral_score', 0) >= 7]
            for seg in high_score_segments[:10]:  # Top 10 segments
                condensed_parts.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] Score:{seg['viral_score']} - {seg['text']}")
            
            # Add full transcript summary
            condensed_parts.append(f"\nFULL TRANSCRIPT:\n{full_transcript}")
            
            return "\n".join(condensed_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess transcription: {e}")
            return full_transcript
    
    def _clean_ai_response(self, response_text):
        """Clean AI response to extract JSON (same as original)"""
        try:
            # Remove markdown formatting
            cleaned = response_text.strip()
            
            # Find JSON block
            if '```json' in cleaned:
                start = cleaned.find('```json') + 7
                end = cleaned.find('```', start)
                cleaned = cleaned[start:end].strip()
            elif '```' in cleaned:
                start = cleaned.find('```') + 3
                end = cleaned.find('```', start)
                cleaned = cleaned[start:end].strip()
            
            # Try to parse as JSON
            return json.loads(cleaned)
            
        except Exception as e:
            logger.error(f"❌ Failed to clean AI response: {e}")
            return response_text
    
    def _fallback_clip_selection(self, viral_segments, num_clips_requested):
        """Fallback clip selection when AI fails (same as original)"""
        try:
            # Check if we have any segments
            if not viral_segments:
                logger.warning("⚠️ No viral segments available, creating default clips")
                # Create default clips from the beginning of the video
                default_clips = []
                for i in range(num_clips_requested):
                    start_time = i * 30  # 30 seconds apart
                    end_time = start_time + 30
                    clip = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': 30,
                        'viral_score': 6,
                        'content_type': 'default',
                        'viral_factor': 'Default clip selection',
                        'segment_text': f"Clip {i+1} - Default selection",
                        'caption': f"Interesting moment {i+1}",
                        'hashtags': ['content', 'video', 'clip'],
                        'target_audience': 'general',
                        'platforms': ['tiktok', 'instagram', 'youtube'],
                        'viral_potential': 6,
                        'engagement': 6,
                        'story_value': 6,
                        'audio_impact': 6
                    }
                    default_clips.append(clip)
                return default_clips
            
            # Sort segments by viral score
            sorted_segments = sorted(viral_segments, key=lambda x: x.get('viral_score', 0), reverse=True)
            
            selected_clips = []
            for i, segment in enumerate(sorted_segments[:num_clips_requested]):
                clip = {
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'duration': segment['end'] - segment['start'],
                    'viral_score': segment.get('viral_score', 7),
                    'content_type': 'viral',
                    'viral_factor': 'High viral potential based on content analysis',
                    'engagement_potential': 'Strong engagement potential',
                    'caption': f"Amazing moment from the video! #{i+1}",
                    'hashtags': ['viral', 'amazing', 'trending'],
                    'hook_line': 'You won\'t believe this!',
                    'call_to_action': 'Follow for more!',
                    'thumbnail_suggestion': 'Exciting moment capture',
                    'target_audience': 'general',
                    'platforms': ['tiktok', 'instagram', 'youtube'],
                    'optimal_posting_time': 'Evening',
                    'cross_platform_adaptation': 'Optimized for all platforms',
                    'segment_text': segment['text'],
                    'reasoning': 'Selected based on viral score analysis',
                    'confidence_score': 0.8,
                    'user_compliance_score': 7
                }
                selected_clips.append(clip)
            
            logger.info(f"✅ Fallback selection: {len(selected_clips)} clips")
            return selected_clips
            
        except Exception as e:
            logger.error(f"❌ Fallback selection failed: {e}")
            return []
    
    def create_clip(self, video_path, start_time, duration, output_name, aspect_ratio_options=None, watermark_options=None):
        """Create video clip using MoviePy with smart aspect ratio processing and watermarking (same as original)"""
        try:
            output_path = self.output_dir / output_name
            
            logger.info(f"   Creating clip with MoviePy...")
            logger.info(f"   Time: {start_time:.1f}s - {start_time + duration:.1f}s")
            
            # Load video with MoviePy
            video = VideoFileClip(video_path)
            
            # Validate duration against actual video length
            video_duration = video.duration
            if start_time + duration > video_duration:
                logger.warning(f"   ⚠️ Requested clip duration ({start_time + duration:.1f}s) exceeds video duration ({video_duration:.1f}s)")
                duration = max(1, video_duration - start_time)  # Ensure at least 1 second
                logger.info(f"   🔧 Adjusted duration to: {duration:.1f}s")
            
            # Extract the clip
            clip = video.subclip(start_time, start_time + duration)
            
            # Apply smart aspect ratio processing if options provided
            if aspect_ratio_options:
                clip = self._apply_aspect_ratio_processing(clip, aspect_ratio_options)
            
            # Apply watermark if options provided
            if watermark_options and watermark_options.get('enableWatermark', False):
                logger.info(f"   💧 Applying watermark to clip...")
                clip = self._apply_watermark(clip, watermark_options)
            
            # Write the clip
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            clip.close()
            video.close()
            
            logger.info(f"   MoviePy clip created successfully: {os.path.basename(output_path)}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Failed to create clip with MoviePy: {str(e)}"
            logger.error(f"   ERROR: {error_msg}")
            raise Exception(error_msg)
    
    def _apply_aspect_ratio_processing(self, clip, aspect_ratio_options):
        """Apply smart aspect ratio processing to video clip (same as original)"""
        try:
            target_ratio = aspect_ratio_options.get('targetAspectRatio', '16:9')
            preserve_original = aspect_ratio_options.get('preserveOriginal', False)
            enable_smart_cropping = aspect_ratio_options.get('enableSmartCropping', True)
            enable_letterboxing = aspect_ratio_options.get('enableLetterboxing', True)
            
            logger.info(f"   🎬 Applying SMART aspect ratio processing: {target_ratio}")
            
            # Parse target aspect ratio
            if ':' in target_ratio:
                width_ratio, height_ratio = map(float, target_ratio.split(':'))
                target_aspect = width_ratio / height_ratio
            else:
                target_aspect = float(target_ratio)
            
            # Get current clip dimensions
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            
            logger.info(f"   📏 Current: {current_width}x{current_height} ({current_aspect:.3f})")
            logger.info(f"   🎯 Target: {target_aspect:.3f}")
            
            # If preserving original aspect ratio, just add letterboxing if needed
            if preserve_original:
                logger.info(f"   📦 Preserving original aspect ratio: {current_aspect:.3f}")
                if enable_letterboxing and abs(current_aspect - target_aspect) > 0.01:
                    logger.info(f"   📦 Adding letterboxing to preserve original ratio")
                    clip = self._add_letterboxing(clip, target_aspect)
                return clip
            
            # Calculate target dimensions
            if current_aspect > target_aspect:
                # Video is wider than target - crop width
                target_width = int(current_height * target_aspect)
                target_height = current_height
                x_offset = (current_width - target_width) // 2
                y_offset = 0
            else:
                # Video is taller than target - crop height
                target_width = current_width
                target_height = int(current_width / target_aspect)
                x_offset = 0
                y_offset = (current_height - target_height) // 2
            
            # Apply cropping
            if enable_smart_cropping:
                clip = clip.crop(x1=x_offset, y1=y_offset, x2=x_offset + target_width, y2=y_offset + target_height)
                logger.info(f"   ✂️ Applied smart cropping: {target_width}x{target_height}")
            
            return clip
            
        except Exception as e:
            logger.error(f"❌ Aspect ratio processing failed: {e}")
            return clip
    
    def _add_letterboxing(self, clip, target_aspect):
        """Add letterboxing to preserve aspect ratio (same as original)"""
        try:
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            
            if current_aspect > target_aspect:
                # Add vertical letterboxing
                target_height = int(current_width / target_aspect)
                y_offset = (target_height - current_height) // 2
                
                # Create black background
                background = ImageClip(size=(current_width, target_height), color=(0, 0, 0), duration=clip.duration)
                
                # Composite clip on background
                clip = CompositeVideoClip([background, clip.set_position(('center', y_offset))])
            else:
                # Add horizontal letterboxing
                target_width = int(current_height * target_aspect)
                x_offset = (target_width - current_width) // 2
                
                # Create black background
                background = ImageClip(size=(target_width, current_height), color=(0, 0, 0), duration=clip.duration)
                
                # Composite clip on background
                clip = CompositeVideoClip([background, clip.set_position((x_offset, 'center'))])
            
            logger.info(f"   📦 Added letterboxing")
            return clip
            
        except Exception as e:
            logger.error(f"❌ Letterboxing failed: {e}")
            return clip
    
    def _apply_watermark(self, clip, watermark_options):
        """Apply watermark to video clip (same as original)"""
        try:
            use_logo = watermark_options.get('useLogo', True)
            watermark_text = watermark_options.get('watermarkText', 'Made by Zuexis')
            position = watermark_options.get('watermarkPosition', 'top-left')
            size = watermark_options.get('watermarkSize', 'extra-large')
            opacity = watermark_options.get('watermarkOpacity', 0.4)
            
            logger.info(f"   💧 Applying watermark: {watermark_text}")
            
            # Size mapping
            size_map = {
                'small': 20,
                'medium': 30,
                'large': 40,
                'extra-large': 50
            }
            text_size = size_map.get(size, 30)
            
            # Create text watermark
            if use_logo and os.path.exists('logo.png'):
                # Use logo if available
                logo_clip = ImageClip('logo.png', duration=clip.duration)
                logo_clip = logo_clip.resize(height=text_size).set_opacity(opacity)
            else:
                # Create text watermark
                text_clip = TextClip(
                    watermark_text,
                    fontsize=text_size,
                    color='white',
                    font='Arial-Bold'
                ).set_duration(clip.duration).set_opacity(opacity)
                logo_clip = text_clip
            
            # Position watermark
            if position == 'top-left':
                logo_clip = logo_clip.set_position(('left', 'top'))
            elif position == 'top-right':
                logo_clip = logo_clip.set_position(('right', 'top'))
            elif position == 'bottom-left':
                logo_clip = logo_clip.set_position(('left', 'bottom'))
            elif position == 'bottom-right':
                logo_clip = logo_clip.set_position(('right', 'bottom'))
            else:
                logo_clip = logo_clip.set_position(('center', 'center'))
            
            # Composite watermark with video
            clip = CompositeVideoClip([clip, logo_clip])
            
            logger.info(f"   ✅ Watermark applied successfully")
            return clip
            
        except Exception as e:
            logger.error(f"❌ Watermark application failed: {e}")
            return clip
    
    def generate_viral_clips(self, video_path, num_clips=3, frontend_inputs=None):
        """Generate viral clips using AI-powered selection and MoviePy (same as original)"""
        try:
            logger.info(f"🎬 Starting viral clip generation for: {video_path}")
            
            # Get video duration for validation
            video_clip = VideoFileClip(video_path)
            video_duration = video_clip.duration
            video_clip.close()
            logger.info(f"📏 Video duration: {video_duration:.1f} seconds")
            
            # Extract and analyze audio segments
            viral_segments, full_transcript = self.extract_audio_segments(video_path)
            
            # AI-powered clip selection with captions and hashtags
            # Add video duration to frontend inputs for AI awareness
            if frontend_inputs:
                frontend_inputs['video_duration'] = video_duration
            viral_moments = self.ai_select_best_clips(viral_segments, full_transcript, num_clips, frontend_inputs)
            
            # Generate clips from AI-selected moments
            generated_clips = []
            clip_details = []
            
            logger.info(f"Generating {len(viral_moments)} AI-selected viral clips...")
            
            for i, moment in enumerate(viral_moments):
                start_time = moment['start_time']
                duration = moment['duration']
                
                # Create descriptive filename
                safe_caption = re.sub(r'[^\w\s-]', '', moment['caption'])[:30]
                clip_name = f"viral_clip_{i+1}_{moment['viral_score']}_{safe_caption}.mp4"
                clip_name = clip_name.replace(' ', '_')
                
                logger.info(f"Creating clip {i+1}/{len(viral_moments)}: {clip_name}")
                logger.info(f"   Time: {start_time:.1f}s - {start_time + duration:.1f}s")
                logger.info(f"   Score: {moment['viral_score']}/10")
                logger.info(f"   Caption: {moment['caption'][:50]}...")
                logger.info(f"   Hashtags: {', '.join(moment['hashtags'][:3])}...")
                
                try:
                    # Extract aspect ratio and watermark options from frontend inputs
                    aspect_ratio_options = None
                    watermark_options = None
                    
                    if frontend_inputs and 'processingOptions' in frontend_inputs:
                        processing_options = frontend_inputs['processingOptions']
                        
                        # Extract aspect ratio options
                        if 'targetAspectRatio' in processing_options or 'aspectRatioOptions' in processing_options:
                            aspect_ratio_options = {
                                'targetAspectRatio': processing_options.get('targetAspectRatio', '16:9'),
                                'preserveOriginal': processing_options.get('preserveOriginalAspectRatio', False),
                                'enableSmartCropping': processing_options.get('enableSmartCropping', True),
                                'enableLetterboxing': processing_options.get('enableLetterboxing', True),
                                'enableQualityPreservation': processing_options.get('enableQualityPreservation', True)
                            }
                            logger.info(f"   🎬 Using aspect ratio options: {aspect_ratio_options}")
                        
                        # Extract watermark options
                        if 'watermarkOptions' in processing_options:
                            watermark_options = processing_options['watermarkOptions']
                            logger.info(f"   💧 Using watermark options: {watermark_options}")
                        elif 'enableWatermark' in processing_options:
                            # Fallback to individual watermark properties
                            watermark_options = {
                                'enableWatermark': processing_options.get('enableWatermark', True),
                                'useLogo': processing_options.get('useLogo', True),
                                'watermarkText': processing_options.get('watermarkText', 'Made by Zuexis'),
                                'watermarkPosition': processing_options.get('watermarkPosition', 'top-left'),
                                'watermarkSize': processing_options.get('watermarkSize', 'extra-large'),
                                'watermarkOpacity': processing_options.get('watermarkOpacity', 0.4)
                            }
                            logger.info(f"   💧 Using fallback watermark options: {watermark_options}")
                    
                    clip_path = self.create_clip(video_path, start_time, duration, clip_name, aspect_ratio_options, watermark_options)
                    generated_clips.append(clip_path)
                    
                    # Save detailed clip information
                    clip_info = {
                        'clip_number': i + 1,
                        'filename': clip_name,
                        'filepath': clip_path,
                        'start_time': start_time,
                        'end_time': moment['end_time'],
                        'duration': duration,
                        'viral_score': moment['viral_score'],
                        'content_type': moment['content_type'],
                        'caption': moment['caption'],
                        'hashtags': moment['hashtags'],
                        'target_audience': moment['target_audience'],
                        'platforms': moment['platforms'],
                        'segment_text': moment['segment_text'],
                        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    clip_details.append(clip_info)
                    
                    # Save individual clip analysis
                    analysis_path = self.output_dir / f"clip_{i+1}_analysis.json"
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(clip_info, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"   Clip created: {os.path.basename(clip_path)}")
                    
                except Exception as e:
                    error_msg = f"Failed to create clip {i+1}: {e}"
                    logger.error(f"   ERROR: {error_msg}")
            
            # Save comprehensive analysis report
            report_data = {
                'video_path': video_path,
                'total_segments': len(viral_segments),
                'selected_clips': len(viral_moments),
                'clips_generated': len(generated_clips),
                'clip_details': clip_details,
                'full_transcript': full_transcript,
                'viral_segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'],
                        'viral_score': seg['viral_score']
                    } for seg in viral_segments
                ],
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report_path = self.output_dir / "generation_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Save full transcription
            transcription_path = self.output_dir / "transcription.txt"
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(f"Full Video Transcription\n{'='*50}\n\n")
                f.write(f"Total Duration: {max(seg['end'] for seg in viral_segments):.1f} seconds\n")
                f.write(f"Viral Segments Found: {len(viral_segments)}\n\n")
                f.write("Viral Segments:\n")
                for seg in viral_segments:
                    f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] Score: {seg['viral_score']}\n")
                    f.write(f"Text: {seg['text']}\n\n")
                f.write("Full Transcript:\n")
                f.write(full_transcript)
            
            logger.info(f"Generated {len(generated_clips)} clips with AI-powered selection")
            logger.info(f"Report saved to: {report_path}")
            
            return generated_clips, full_transcript
            
        except Exception as e:
            logger.error(f"❌ Viral clip generation failed: {e}")
            raise e


# Flask Application
app = Flask(__name__)
CORS(app)
# Ensure CORS headers are present on all responses (including errors)
@app.after_request
def add_cors_headers(response):
    try:
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    except Exception:
        pass
    return response


# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize video clipper
video_clipper = FocusedVideoClipper()

# Global processing status tracking
processing_status = {
    'is_processing': False,
    'current_task': None,
    'progress': 0,
    'message': '',
    'start_time': None,
    'last_checkpoint': None,
    'completed_steps': []
}

# Progress persistence file
PROGRESS_FILE = '/tmp/processing_progress.json'

def save_progress():
    """Save current processing progress to disk"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(processing_status, f, indent=2)
        logger.info(f"💾 Progress saved: {processing_status['progress']}% - {processing_status['message']}")
    except Exception as e:
        logger.error(f"❌ Failed to save progress: {e}")

def load_progress():
    """Load processing progress from disk"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                saved_progress = json.load(f)
                processing_status.update(saved_progress)
                logger.info(f"📂 Progress loaded: {processing_status['progress']}% - {processing_status['message']}")
                return True
    except Exception as e:
        logger.error(f"❌ Failed to load progress: {e}")
    return False

def clear_progress():
    """Clear saved progress"""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        processing_status.update({
            'is_processing': False,
            'current_task': None,
            'progress': 0,
            'message': '',
            'start_time': None,
            'last_checkpoint': None,
            'completed_steps': []
        })
        logger.info("🧹 Progress cleared")
    except Exception as e:
        logger.error(f"❌ Failed to clear progress: {e}")

def update_progress(progress: int, message: str, checkpoint: str = None):
    """Update progress and save to disk"""
    processing_status['progress'] = progress
    processing_status['message'] = message
    if checkpoint:
        processing_status['last_checkpoint'] = checkpoint
        if checkpoint not in processing_status['completed_steps']:
            processing_status['completed_steps'].append(checkpoint)
    save_progress()

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """🚀 RENDER OPTIMIZED: Health check endpoint with memory monitoring"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Health check preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Get memory information (simple fallback)
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        memory_usage_mb = round(memory_info.used / (1024 * 1024), 2)
        memory_percent = round(memory_info.percent, 1)
    except ImportError:
        memory_usage_mb = 0
        memory_percent = 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'Focused Video Clipper (Render Optimized)',
        'memory_usage_mb': memory_usage_mb,
        'memory_percent': memory_percent,
        'render_optimizations': {
            'chunk_size_mb': CHUNK_TARGET_SIZE_MB,
            'max_workers': RenderOptimizedConstants.MAX_WORKERS,
            'max_chunk_duration': MAX_CHUNK_DURATION,
            'audio_sample_rate': DEFAULT_AUDIO_SR
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check_simple():
    """🚀 RENDER OPTIMIZATION: Simple health check for Render platform"""
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        memory_usage_mb = round(memory_info.used / (1024 * 1024), 2)
        memory_percent = round(memory_info.percent, 1)
    except ImportError:
        memory_usage_mb = 0
        memory_percent = 0
    
    return jsonify({
        'status': 'healthy',
        'memory_usage_mb': memory_usage_mb,
        'memory_percent': memory_percent,
        'timestamp': time.time()
    })

@app.route('/api/process-chunk', methods=['POST', 'OPTIONS'])
def process_chunk():
    """🚀 SUPER FAST: Process a single video chunk for chunked uploads"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Chunk processing preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    try:
        logger.info("🚀 [ChunkedUpload] Processing video chunk...")
        
        # Get chunk data
        chunk = request.files.get('chunk')
        chunk_id = request.form.get('chunkId')
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        is_last_chunk = request.form.get('isLastChunk', 'false').lower() == 'true'
        project_data = json.loads(request.form.get('projectData', '{}'))
        
        if not chunk:
            return jsonify({'success': False, 'error': 'No chunk provided'}), 400
        
        logger.info(f"📦 [ChunkedUpload] Processing chunk {chunk_index + 1}/{total_chunks} (ID: {chunk_id})")
        
        # Save chunk temporarily
        chunk_filename = f"chunk_{chunk_id}_{chunk_index}.tmp"
        chunk_path = os.path.join(TEMP_DIR, chunk_filename)
        chunk.save(chunk_path)
        
        # Process chunk based on type
        if is_last_chunk:
            # Last chunk - combine all chunks and process
            result = process_combined_chunks(chunk_id, total_chunks, project_data)
        else:
            # Regular chunk - just store for later combination
            result = store_chunk_for_combination(chunk_id, chunk_index, chunk_path)
        
        # Clean up temporary chunk file
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ [ChunkedUpload] Error processing chunk: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def store_chunk_for_combination(chunk_id: str, chunk_index: int, chunk_path: str) -> Dict[str, Any]:
    """Store chunk for later combination"""
    try:
        # Create chunk storage directory
        chunk_dir = os.path.join(TEMP_DIR, f"chunks_{chunk_id}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Move chunk to storage directory
        stored_chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index}.tmp")
        shutil.move(chunk_path, stored_chunk_path)
        
        logger.info(f"📦 [ChunkedUpload] Stored chunk {chunk_index} for {chunk_id}")
        
        return {
            'success': True,
            'message': f'Chunk {chunk_index} stored successfully',
            'chunkId': chunk_id,
            'chunkIndex': chunk_index,
            'stored': True
        }
        
    except Exception as e:
        logger.error(f"❌ [ChunkedUpload] Error storing chunk: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def process_combined_chunks(chunk_id: str, total_chunks: int, project_data: Dict[str, Any]) -> Dict[str, Any]:
    """Combine all chunks and process the complete video"""
    try:
        logger.info(f"🔄 [ChunkedUpload] Combining {total_chunks} chunks for {chunk_id}")
        
        # Get chunk storage directory
        chunk_dir = os.path.join(TEMP_DIR, f"chunks_{chunk_id}")
        
        if not os.path.exists(chunk_dir):
            return {
                'success': False,
                'error': 'Chunk storage directory not found'
            }
        
        # Combine chunks into single video file
        combined_video_path = os.path.join(TEMP_DIR, f"combined_{chunk_id}.mp4")
        
        with open(combined_video_path, 'wb') as combined_file:
            for i in range(total_chunks):
                chunk_path = os.path.join(chunk_dir, f"chunk_{i}.tmp")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as chunk_file:
                        combined_file.write(chunk_file.read())
                else:
                    logger.warning(f"⚠️ [ChunkedUpload] Missing chunk {i} for {chunk_id}")
        
        logger.info(f"✅ [ChunkedUpload] Combined video created: {combined_video_path}")
        
        # Process the combined video using persistent processing system
        try:
            logger.info(f"🎬 [ChunkedUpload] Starting persistent processing for combined video...")
            
            # Extract project information
            project_name = project_data.get('projectName', 'Chunked Video')
            description = project_data.get('description', '')
            target_platforms = project_data.get('targetPlatforms', ['tiktok'])
            ai_prompt = project_data.get('aiPrompt', '')
            processing_options = project_data.get('processingOptions', {})
            num_clips = project_data.get('numClips', 3)
            
            # Create persistent processing job
            job_id = persistent_manager.create_job(
                combined_video_path,
                {
                    'projectName': project_name,
                    'description': description,
                    'targetPlatforms': target_platforms,
                    'aiPrompt': ai_prompt,
                    'processingOptions': processing_options,
                    'numClips': num_clips
                }
            )
            
            result = {
                'success': True,
                'message': 'Video processing started in background - will never reset!',
                'job_id': job_id,
                'project_name': project_name,
                'status': 'processing_started',
                'progress_url': f'/api/job-status/{job_id}',
                'processing_options': processing_options,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ [ChunkedUpload] Error processing video: {e}")
            result = {
                'success': False,
                'error': str(e)
            }
        
        # Clean up chunk storage
        shutil.rmtree(chunk_dir, ignore_errors=True)
        if os.path.exists(combined_video_path):
            os.remove(combined_video_path)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ [ChunkedUpload] Error combining chunks: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/process-video', methods=['POST', 'OPTIONS'])
def process_video_direct():
    """🎬 DIRECT: Process a complete video file directly (no chunking)"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Direct video processing preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    try:
        logger.info("🎬 [DirectUpload] Processing complete video file...")
        
        # Get video file and project data
        video_file = request.files.get('video')
        project_data = json.loads(request.form.get('projectData', '{}'))
        
        if not video_file:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        logger.info(f"📁 [DirectUpload] Processing video: {video_file.filename}")
        
        # Save video file temporarily
        video_filename = f"direct_upload_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(TEMP_DIR, video_filename)
        video_file.save(video_path)
        
        logger.info(f"✅ [DirectUpload] Video saved: {video_path}")
        
        # Process the video using persistent processing system
        try:
            logger.info(f"🎬 [DirectUpload] Starting persistent processing for direct upload...")
            
            # Extract project information
            project_name = project_data.get('projectName', 'Direct Upload Video')
            description = project_data.get('description', '')
            target_platforms = project_data.get('targetPlatforms', ['tiktok'])
            ai_prompt = project_data.get('aiPrompt', '')
            processing_options = project_data.get('processingOptions', {})
            num_clips = project_data.get('numClips', 3)
            
            # Create persistent processing job
            job_id = persistent_manager.create_job(
                video_path,
                {
                    'projectName': project_name,
                    'description': description,
                    'targetPlatforms': target_platforms,
                    'aiPrompt': ai_prompt,
                    'processingOptions': processing_options,
                    'numClips': num_clips
                }
            )
            
            result = {
                'success': True,
                'message': 'Video processing started in background - will never reset!',
                'job_id': job_id,
                'project_name': project_name,
                'status': 'processing_started',
                'progress_url': f'/api/job-status/{job_id}',
                'processing_options': processing_options,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ [DirectUpload] Error processing video: {e}")
            result = {
                'success': False,
                'error': str(e)
            }
        
        # Clean up video file after processing starts
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ [DirectUpload] Error processing direct upload: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/progress/<task_id>', methods=['GET', 'OPTIONS'])
def get_progress(task_id):
    """Get processing progress for a task"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Progress check preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Return current processing status
    return jsonify({
        'task_id': task_id,
        'is_processing': processing_status['is_processing'],
        'current_task': processing_status['current_task'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'start_time': processing_status['start_time'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status', methods=['GET', 'OPTIONS'])
def get_status():
    """Get current processing status"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Status check preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Load latest progress from disk
    load_progress()
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'current_task': processing_status['current_task'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'start_time': processing_status['start_time'],
        'last_checkpoint': processing_status['last_checkpoint'],
        'completed_steps': processing_status['completed_steps'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/progress', methods=['GET', 'OPTIONS'])
def get_detailed_progress():
    """Get detailed progress information"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Progress check preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    load_progress()
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'last_checkpoint': processing_status['last_checkpoint'],
        'completed_steps': processing_status['completed_steps'],
        'start_time': processing_status['start_time'],
        'elapsed_time': time.time() - processing_status['start_time'] if processing_status['start_time'] else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/process-video', methods=['POST', 'OPTIONS'])
def process_video():
    """Process video and generate viral clips (same input/output format as original)"""
    if request.method == 'OPTIONS':
        logger.info("🔄 [API] OPTIONS request received for /api/process-video")
        response = jsonify({'message': 'Process video preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Check if already processing
    if processing_status['is_processing']:
        logger.warning("⚠️ [API] Video processing already in progress")
        return jsonify({
            'success': False,
            'error': 'Video processing already in progress. Please wait for current task to complete.'
        }), 409
    
    try:
        logger.info("🚀 [API] POST request received for /api/process-video")
        logger.info(f"📊 [API] Request content type: {request.content_type}")
        logger.info(f"📊 [API] Request method: {request.method}")
        logger.info(f"📊 [API] Request headers: {dict(request.headers)}")
        logger.info("🚀 Processing video request received")
        
        # Check for existing progress and resume if possible
        if load_progress() and processing_status['is_processing']:
            logger.info(f"🔄 [API] Resuming previous processing from {processing_status['last_checkpoint']}")
            return jsonify({
                'success': False,
                'error': 'Video processing already in progress. Please wait for current task to complete.',
                'progress': processing_status['progress'],
                'message': processing_status['message']
            }), 409
        
        # Set processing status
        processing_status['is_processing'] = True
        processing_status['current_task'] = 'video_processing'
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting video processing...'
        processing_status['start_time'] = time.time()
        processing_status['last_checkpoint'] = 'start'
        processing_status['completed_steps'] = []
        save_progress()
        
        # Handle both JSON and FormData requests
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle FormData request
            data = {
                'projectName': request.form.get('projectName'),
                'sourceType': request.form.get('sourceType'),
                'description': request.form.get('description', ''),
                'aiPrompt': request.form.get('aiPrompt', ''),
                'targetPlatforms': json.loads(request.form.get('targetPlatforms', '["tiktok"]')),
                'numClips': int(request.form.get('numClips', 3)),
                'processingOptions': json.loads(request.form.get('processingOptions', '{}')),
                'videoFile': request.files.get('videoFile') if 'videoFile' in request.files else None
            }
        else:
            # Handle JSON request
            data = request.get_json()
        
        # Validate required fields
        required_fields = ['projectName', 'sourceType']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_name = data['projectName']
        source_type = data['sourceType']
        logger.info(f"🔎 [API] Parsed fields: project={project_name}, sourceType={source_type}")
        description = data.get('description', '')
        ai_prompt = data.get('aiPrompt', '')
        target_platforms = data.get('targetPlatforms', ['tiktok'])
        num_clips = data.get('numClips', 3)
        processing_options = data.get('processingOptions', {})
        
        # Handle video source
        if source_type == 'file':
            if 'videoFile' not in data:
                return jsonify({'error': 'Video file required for file upload'}), 400
            
            video_file = data['videoFile']
            
            # Save uploaded file
            if hasattr(video_file, 'filename') and video_file.filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_project_name = safe_project_name.replace(' ', '_')[:30]
                safe_filename = f"{timestamp}_{safe_project_name}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                
                video_file.save(filepath)
                logger.info(f"✅ Video saved: {filepath}")
                update_progress(10, 'Video file uploaded successfully', 'file_uploaded')
            else:
                return jsonify({'error': 'Invalid video file format'}), 400
        elif source_type in ('cloud', 'drive', 'google-drive'):
            provider = data.get('provider') or 'google-drive'
            file_id = data.get('fileId')
            access_token = data.get('googleAccessToken')
            if not (provider == 'google-drive' and file_id and access_token):
                return jsonify({'error': 'Missing provider/fileId/access token for cloud source'}), 400
            logger.info(f"🌐 [API] Cloud source: provider={provider}, fileId={file_id[:8]}..., token={access_token[:12]}...")
            # Download from Google Drive
            import requests
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_project_name = safe_project_name.replace(' ', '_')[:30]
            safe_filename = f"{timestamp}_{safe_project_name}_gdrive.mp4"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            logger.info(f"🌐 Downloading from Google Drive fileId={file_id} -> {filepath}")
            update_progress(5, 'Downloading video from Google Drive...', 'downloading')
            with requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                              headers={'Authorization': f'Bearer {access_token}'},
                              stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('Content-Length') or 0)
                written = 0
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                            if total:
                                progress = 5 + int((written / total) * 10)  # 5-15% for download
                                logger.info(f"⬇️ [API] Download progress: {written/1024/1024:.1f}MB / {total/1024/1024:.1f}MB")
                                update_progress(progress, f'Downloading video... {written/1024/1024:.1f}MB / {total/1024/1024:.1f}MB', 'downloading')
            logger.info(f"✅ Video downloaded: {filepath}")
            update_progress(15, 'Video downloaded successfully', 'download_complete')
        else:
            return jsonify({'error': f'Unsupported source type: {source_type}'}), 400
        
        # Prepare frontend inputs for processing
        frontend_inputs = {
            'projectName': project_name,
            'description': description,
            'aiPrompt': ai_prompt,
            'targetPlatforms': target_platforms,
            'processingOptions': processing_options
        }
        
        # Generate viral clips
        logger.info(f"🎬 Starting clip generation for {num_clips} clips...")
        update_progress(20, 'Starting video processing...', 'processing_started')
        generated_clips, full_transcript = video_clipper.generate_viral_clips(
            filepath, 
            num_clips=num_clips, 
            frontend_inputs=frontend_inputs
        )
        update_progress(90, 'Video processing completed', 'processing_complete')
        
        # Prepare response (same format as original)
        response_data = {
            'success': True,
            'message': f'Successfully generated {len(generated_clips)} viral clips',
            'project_name': project_name,
            'clips_generated': len(generated_clips),
            'clips': [
                {
                    'filename': os.path.basename(clip_path),
                    'filepath': clip_path,
                    'download_url': f'/api/download/{os.path.basename(clip_path)}'
                } for clip_path in generated_clips
            ],
            'transcription': full_transcript,
            'processing_options': processing_options,
            'timestamp': datetime.now().isoformat()
        }
        
        # Complete processing and clear progress
        update_progress(100, 'Video processing completed successfully', 'completed')
        clear_progress()
        
        logger.info(f"✅ Video processing completed: {len(generated_clips)} clips generated")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Video processing failed: {e}")
        clear_progress()  # Clear progress on error
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_clip(filename):
    """Download generated clip"""
    try:
        clip_path = video_clipper.output_dir / filename
        if clip_path.exists():
            return send_file(str(clip_path), as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcription/<filename>', methods=['GET'])
def get_transcription(filename):
    """Get transcription for processed video"""
    try:
        transcription_path = video_clipper.output_dir / "transcription.txt"
        if transcription_path.exists():
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = f.read()
            return jsonify({
                'success': True,
                'transcription': transcription
            })
        else:
            return jsonify({'error': 'Transcription not found'}), 404
    except Exception as e:
        logger.error(f"❌ Transcription retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<filename>', methods=['GET'])
def get_analysis(filename):
    """Get analysis report for processed video"""
    try:
        report_path = video_clipper.output_dir / "generation_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        else:
            return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        logger.error(f"❌ Analysis retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-status/<job_id>', methods=['GET', 'OPTIONS'])
def get_job_status(job_id):
    """Get status of a persistent processing job"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Job status preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    try:
        job = persistent_manager.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        response_data = {
            'job_id': job.job_id,
            'status': job.status.value,
            'progress': job.progress,
            'message': job.message,
            'current_step': job.current_step,
            'completed_steps': job.completed_steps,
            'start_time': job.start_time,
            'last_update': job.last_update,
            'elapsed_time': time.time() - job.start_time,
            'project_name': job.project_data.get('projectName', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add result if completed
        if job.result:
            response_data['result'] = job.result
        
        # Add error if failed
        if job.error:
            response_data['error'] = job.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Job status retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['GET', 'OPTIONS'])
def get_all_jobs():
    """Get all persistent processing jobs"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Jobs list preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    try:
        jobs = persistent_manager.get_all_jobs()
        jobs_data = []
        
        for job in jobs.values():
            job_data = {
                'job_id': job.job_id,
                'status': job.status.value,
                'progress': job.progress,
                'message': job.message,
                'current_step': job.current_step,
                'start_time': job.start_time,
                'last_update': job.last_update,
                'elapsed_time': time.time() - job.start_time,
                'project_name': job.project_data.get('projectName', 'Unknown')
            }
            jobs_data.append(job_data)
        
        # Sort by start time (newest first)
        jobs_data.sort(key=lambda x: x['start_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'jobs': jobs_data,
            'total_jobs': len(jobs_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Jobs list retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel-job/<job_id>', methods=['POST', 'OPTIONS'])
def cancel_job(job_id):
    """Cancel a persistent processing job"""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Cancel job preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    try:
        success = persistent_manager.cancel_job(job_id)
        if success:
            return jsonify({
                'success': True,
                'message': f'Job {job_id} cancelled successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Job not found or cannot be cancelled'
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Job cancellation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Focused Video Clipper Backend (Persistent Processing)',
        'version': '2.0.0',
        'features': [
            '🔄 Persistent Processing - Never resets, survives timeouts',
            '📦 Chunked Upload - Direct backend upload in 1MB chunks',
            '🎬 AI-Powered Clip Generation - Gemini AI selection',
            '⏱️ Real-time Progress Tracking - Live status updates',
            '💾 Progress Persistence - Survives server restarts'
        ],
        'endpoints': [
            'POST /api/process-chunk - Chunked video upload (NEW)',
            'GET /api/job-status/<job_id> - Get job status (NEW)',
            'GET /api/jobs - List all jobs (NEW)',
            'POST /api/cancel-job/<job_id> - Cancel job (NEW)',
            'POST /api/process-video - Process video and generate clips',
            'GET /api/download/<filename> - Download generated clip',
            'GET /api/transcription/<filename> - Get transcription',
            'GET /api/analysis/<filename> - Get analysis report',
            'GET /api/health - Health check'
        ],
        'persistent_processing': {
            'enabled': True,
            'timeout_immune': True,
            'auto_resume': True,
            'progress_persistence': True
        }
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Focused Video Clipper Backend...")
    # Disable auto-reload to prevent interruption during video processing
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
